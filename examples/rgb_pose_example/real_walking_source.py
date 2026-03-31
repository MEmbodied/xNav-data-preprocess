from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from utils.rgb_pose_dataset import (
    RGBPosePoint,
    RGBPoseSourceInfo,
    RGBPoseTrajectory,
    RGBPoseTrajectorySource,
    transform_to_pose_vector,
)


def _body_from_camera_from_coordinate_system(coordinate_system: dict[str, Any]) -> np.ndarray:
    convention = coordinate_system.get("camera_convention", "")
    axes = coordinate_system.get("axes", "")

    if convention == "OpenCV" or axes == "+X right, +Y down, +Z forward":
        rotation = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
    elif convention == "OpenGL" or axes == "+X right, +Y up, +Z backward":
        rotation = np.array(
            [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unsupported camera convention: convention={convention!r}, axes={axes!r}")

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    return transform


def _normalize_fps(value: float) -> int:
    fps = int(round(float(value)))
    if abs(float(value) - fps) > 1e-6:
        raise ValueError(f"LeRobot export expects integer fps, got {value}")
    return fps


def _mean_intrinsics(frames: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["fx", "fy", "cx", "cy"]
    return {
        key: float(np.mean([float(frame["intrinsics"][key]) for frame in frames], dtype=np.float64))
        for key in keys
    }


class RealWalkingTrajectory(RGBPoseTrajectory):
    def __init__(self, manifest_path: Path, manifest: dict[str, Any], video_path: Path, body_from_camera: np.ndarray):
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.video_path = video_path
        self.body_from_camera = body_from_camera.astype(np.float32)
        self.frames = manifest["frames"]
        self._camera_intrinsics_mean = _mean_intrinsics(self.frames)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "camera_convention": self.manifest["coordinate_system"]["camera_convention"],
            "camera_axes": self.manifest["coordinate_system"]["axes"],
            "camera_pose_type": self.manifest["coordinate_system"]["camera_pose_type"],
            "K": self._camera_intrinsics_mean,
            "T_b_c": self.body_from_camera,
        }

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[RGBPosePoint]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        current_index = -1
        current_bgr = None
        try:
            for frame in self.frames:
                target_index = int(frame["source_frame_index"])
                if target_index < current_index:
                    raise ValueError("source_frame_index must be non-decreasing within one trajectory")

                while current_index < target_index:
                    ok, current_bgr = cap.read()
                    current_index += 1
                    if not ok or current_bgr is None:
                        raise RuntimeError(
                            f"Video ended before frame {target_index} in trajectory {self.manifest['video_id']}"
                        )

                rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
                camera_pose = transform_to_pose_vector(np.asarray(frame["pose"]["camera_pose"], dtype=np.float32))
                yield RGBPosePoint(rgb=rgb, camera_pose=camera_pose)
        finally:
            cap.release()


class RealWalkingRGBPoseSource(RGBPoseTrajectorySource):
    def __init__(self, data_path: str | Path, limit: int | None = None):
        self.info_dir = Path(data_path)
        if not self.info_dir.exists():
            raise FileNotFoundError(f"Info directory does not exist: {self.info_dir}")

        manifest_paths = sorted(self.info_dir.glob("*-camera.json"))
        if not manifest_paths:
            raise FileNotFoundError(f"No '*-camera.json' files found under {self.info_dir}")
        if limit is not None:
            manifest_paths = manifest_paths[: max(0, int(limit))]

        self.records: list[tuple[Path, dict[str, Any], Path]] = []
        for manifest_path in manifest_paths:
            with open(manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
            video_path = self._resolve_video_path(manifest)
            self.records.append((manifest_path, manifest, video_path))

        first_manifest = self.records[0][1]
        first_frames = first_manifest["frames"]
        width, height = first_frames[0]["image_size"]
        body_from_camera = _body_from_camera_from_coordinate_system(first_manifest["coordinate_system"])

        self._info = RGBPoseSourceInfo(
            fps=_normalize_fps(first_manifest["sample_fps"]),
            image_size=(int(height), int(width)),
            body_from_camera=body_from_camera,
            robot_type="lerobot",
            task="",
            source_name="sekai_real_walking_hq_video",
        )

        self._validate_records()

    def _resolve_video_path(self, manifest: dict[str, Any]) -> Path:
        candidates = [Path(manifest["source_video"])]

        sibling_video_dir = self.info_dir.parent / self.info_dir.name.replace("-info", "")
        candidates.append(sibling_video_dir / f"{manifest['video_id']}.mp4")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Failed to resolve video for {manifest['video_id']} from candidates: {candidates}")

    def _validate_records(self):
        expected_fps = self._info.fps
        expected_size_wh = (self._info.image_size[1], self._info.image_size[0])
        expected_extrinsic = self._info.body_from_camera

        for manifest_path, manifest, video_path in self.records:
            fps = _normalize_fps(manifest["sample_fps"])
            if fps != expected_fps:
                raise ValueError(f"sample_fps mismatch in {manifest_path}: {fps} != {expected_fps}")

            frames = manifest["frames"]
            if not frames:
                raise ValueError(f"No frames in manifest: {manifest_path}")

            size_wh = tuple(frames[0]["image_size"])
            if size_wh != expected_size_wh:
                raise ValueError(f"image_size mismatch in {manifest_path}: {size_wh} != {expected_size_wh}")

            extrinsic = _body_from_camera_from_coordinate_system(manifest["coordinate_system"])
            if not np.allclose(extrinsic, expected_extrinsic):
                raise ValueError(f"camera convention mismatch in {manifest_path}")

            if not video_path.exists():
                raise FileNotFoundError(f"Missing video file for {manifest_path}: {video_path}")

    @property
    def info(self) -> RGBPoseSourceInfo:
        return self._info

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[RealWalkingTrajectory]:
        for manifest_path, manifest, video_path in self.records:
            yield RealWalkingTrajectory(
                manifest_path=manifest_path,
                manifest=manifest,
                video_path=video_path,
                body_from_camera=self._info.body_from_camera,
            )
