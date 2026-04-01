"""CityWalker dataset source adapter for the RGB-Pose LeRobot export pipeline.

CityWalker stores trajectories as:
- ``pose_label/pose_traj_XX.txt``  (3 lines per frame: GPS, pose, category)
- ``obs/traj_nav_XX/forward_XXXX.jpg``  (RGB images)

Pose format (per frame, line 2 of each 3-line group):
    timestamp, tx, ty, tz, rx, ry, rz, image_idx

- Translation [tx, ty, tz] is in **meters**.
- Rotation [rx, ry, rz] is a **rotation vector (axis-angle)** in radians.
- Poses come from DPVO (monocular visual-inertial odometry) and represent
  ``T_world<-camera`` in an **OpenCV camera convention**
  (+X right, +Y down, +Z forward).
- Pose labels are sampled at ~1 fps; the raw images are captured at ~10 fps.
- Image resolution: 400 x 400.
- No camera intrinsics are provided in this dataset.

Usage::

    uv run rgb_pose_to_lerobot.py \\
        --raw_dir /data-25T/CityWalker \\
        --source_cls examples.rgb_pose_example.citywalker_source:CityWalkerRGBPoseSource \\
        --output_dir ./tmp --dataset_name citywalker-test
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from utils.rgb_pose_dataset import (
    RGBPosePoint,
    RGBPoseSourceInfo,
    RGBPoseTrajectory,
    RGBPoseTrajectorySource,
)

logger = logging.getLogger(__name__)

# OpenCV camera convention: +X right, +Y down, +Z forward
# Body convention used by the export pipeline: +X forward, +Y left, +Z up
_BODY_FROM_CAMERA_OPENCV = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector (axis-angle) to quaternion in [qx, qy, qz, qw]."""
    return Rotation.from_rotvec(rotvec).as_quat().astype(np.float32)


def _parse_pose_file(path: Path) -> list[dict[str, Any]]:
    """Parse a ``pose_traj_XX.txt`` file.

    Returns a list of dicts, one per frame, with keys:
        - ``tx``, ``ty``, ``tz``: translation (meters)
        - ``qx``, ``qy``, ``qz``, ``qw``: quaternion (xyzw)
        - ``image_idx``: integer frame index into ``forward_XXXX.jpg``
        - ``timestamp``: pose timestamp offset (seconds)
    """
    lines = path.read_text().splitlines()
    if len(lines) % 3 != 0:
        raise ValueError(f"Pose file {path} has {len(lines)} lines, expected a multiple of 3")

    frames: list[dict[str, Any]] = []
    for i in range(0, len(lines), 3):
        pose_tokens = lines[i + 1].strip().split(",")
        timestamp = float(pose_tokens[0])
        tx, ty, tz = float(pose_tokens[1]), float(pose_tokens[2]), float(pose_tokens[3])
        rx, ry, rz = float(pose_tokens[4]), float(pose_tokens[5]), float(pose_tokens[6])
        image_idx = int(pose_tokens[7])

        quat = _rotvec_to_quat_xyzw(np.array([rx, ry, rz], dtype=np.float64))

        frames.append(
            {
                "timestamp": timestamp,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "qx": float(quat[0]),
                "qy": float(quat[1]),
                "qz": float(quat[2]),
                "qw": float(quat[3]),
                "image_idx": image_idx,
            }
        )
    return frames


class CityWalkerTrajectory(RGBPoseTrajectory):
    """One CityWalker trajectory (one ``pose_traj_XX.txt`` + ``traj_nav_XX/``)."""

    def __init__(self, pose_frames: list[dict[str, Any]], image_dir: Path, traj_id: str):
        self._frames = pose_frames
        self._image_dir = image_dir
        self._traj_id = traj_id

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "video.ego_view.K": None,  # no intrinsics available
            "video.ego_view.body_from_camera": _BODY_FROM_CAMERA_OPENCV,
        }

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[RGBPosePoint]:
        for frame in self._frames:
            image_path = self._image_dir / f"forward_{frame['image_idx']:04d}.jpg"
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")

            camera_pose = np.array(
                [frame["tx"], frame["ty"], frame["tz"],
                 frame["qx"], frame["qy"], frame["qz"], frame["qw"]],
                dtype=np.float32,
            )
            yield RGBPosePoint(rgb=image_path, camera_pose=camera_pose)


class CityWalkerRGBPoseSource(RGBPoseTrajectorySource):
    """Dataset-level adapter for CityWalker.

    Expects ``raw_dir`` to contain:
    - ``pose_label/pose_traj_01.txt`` .. ``pose_traj_18.txt``
    - ``obs/traj_nav_01/`` .. ``traj_nav_18/``
    """

    def __init__(self, data_path: str | Path, limit: int | None = None):
        root = Path(data_path)
        pose_dir = root / "pose_label"
        obs_dir = root / "obs"

        if not pose_dir.exists():
            raise FileNotFoundError(f"Pose label directory not found: {pose_dir}")
        if not obs_dir.exists():
            raise FileNotFoundError(f"Observation directory not found: {obs_dir}")

        pose_files = sorted(pose_dir.glob("pose_traj_*.txt"))
        if not pose_files:
            raise FileNotFoundError(f"No pose_traj_*.txt files found in {pose_dir}")
        if limit is not None:
            pose_files = pose_files[: max(0, int(limit))]

        self._trajectories: list[tuple[list[dict[str, Any]], Path, str]] = []
        for pose_file in pose_files:
            # Extract trajectory number: pose_traj_01.txt -> 01
            traj_num = pose_file.stem.replace("pose_traj_", "")
            image_dir = obs_dir / f"traj_nav_{traj_num}"
            if not image_dir.exists():
                logger.warning("Skipping %s: image dir %s not found", pose_file.name, image_dir)
                continue

            frames = _parse_pose_file(pose_file)
            if not frames:
                logger.warning("Skipping %s: no frames parsed", pose_file.name)
                continue

            self._trajectories.append((frames, image_dir, f"traj_nav_{traj_num}"))
            logger.info("Loaded %s: %d frames", pose_file.name, len(frames))

        if not self._trajectories:
            raise RuntimeError("No valid trajectories found")

        # Verify image size from the first image of the first trajectory
        first_frame = self._trajectories[0][0][0]
        first_image_dir = self._trajectories[0][1]
        first_image = Image.open(first_image_dir / f"forward_{first_frame['image_idx']:04d}.jpg")
        width, height = first_image.size

        self._info = RGBPoseSourceInfo(
            fps=1,
            image_size=(height, width),
            body_from_camera=_BODY_FROM_CAMERA_OPENCV,
            robot_type="lerobot",
            task="",
            source_name="citywalker",
        )

    @property
    def info(self) -> RGBPoseSourceInfo:
        return self._info

    def __len__(self) -> int:
        return len(self._trajectories)

    def __iter__(self) -> Iterator[CityWalkerTrajectory]:
        for frames, image_dir, traj_id in self._trajectories:
            yield CityWalkerTrajectory(frames, image_dir, traj_id)
