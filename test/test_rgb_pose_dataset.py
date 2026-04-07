import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils.rgb_pose_dataset import (
    RGBPosePoint,
    RGBPoseSourceInfo,
    RGBPoseTrajectory,
    RGBPoseTrajectorySource,
    camera_pose_to_body_pose,
    export_rgb_pose_source,
    load_rgb_image,
    select_video_pixel_format,
)


class MockRGBPoseTrajectory(RGBPoseTrajectory):
    def __init__(self, trajectory_id: str, points: list[RGBPosePoint], task: str = ""):
        self.trajectory_id = trajectory_id
        self.points = points
        self._task = task

    @property
    def task(self) -> str:
        return self._task

    @property
    def metadata(self) -> dict:
        return {"trajectory_id": self.trajectory_id}

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        yield from self.points


class MockRGBPoseSource(RGBPoseTrajectorySource):
    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        self._info = RGBPoseSourceInfo(
            fps=12,
            image_size=(4, 5),
            body_from_camera=np.eye(4, dtype=np.float32),
            robot_type="mock_uav",
            task="",
        )

        self._trajectories = [
            MockRGBPoseTrajectory(
                "traj_0",
                points=[
                    RGBPosePoint(
                        rgb=np.full((4, 5, 3), 10, dtype=np.uint8),
                        camera_pose=np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
                    ),
                    RGBPosePoint(
                        rgb=np.full((4, 5, 3), 20, dtype=np.uint8),
                        camera_pose=np.array([1, 2, 3, 0, 0, 0, 1], dtype=np.float32),
                    ),
                ],
            ),
            MockRGBPoseTrajectory(
                "traj_1",
                points=[
                    RGBPosePoint(
                        rgb=np.full((4, 5, 3), 30, dtype=np.uint8),
                        camera_pose=np.array([4, 5, 6, 0, 0, 0, 1], dtype=np.float32),
                    ),
                    RGBPosePoint(
                        rgb=np.full((4, 5, 3), 40, dtype=np.uint8),
                        camera_pose=np.array([7, 8, 9, 0, 0, 0, 1], dtype=np.float32),
                    ),
                ],
            ),
        ]

    @property
    def info(self) -> RGBPoseSourceInfo:
        return self._info

    def __len__(self) -> int:
        return len(self._trajectories)

    def __iter__(self):
        yield from self._trajectories


class RGBPoseDatasetTests(unittest.TestCase):
    def test_camera_pose_to_body_pose_uses_body_from_camera(self):
        body_from_camera = np.eye(4, dtype=np.float32)
        body_from_camera[:3, 3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        camera_pose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        body_pose = camera_pose_to_body_pose(camera_pose, body_from_camera)

        expected = np.array([0, 0, -1, 0, 0, 0, 1], dtype=np.float32)
        np.testing.assert_allclose(body_pose, expected, atol=1e-6)

    def test_export_rgb_pose_source(self):
        with tempfile.TemporaryDirectory(prefix="rgb_pose_export_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            root = tmp_path / "mock_dataset"

            export_rgb_pose_source(
                raw_dir=raw_dir,
                repo_id="mock_dataset",
                root=root,
                source_cls=MockRGBPoseSource,
                num_processes=1,
                codec="h264",
            )

            with open(root / "meta" / "info.json", "r", encoding="utf-8") as file:
                info = json.load(file)
            with open(root / "meta" / "tasks.jsonl", "r", encoding="utf-8") as file:
                tasks = [json.loads(line) for line in file if line.strip()]
            with open(root / "meta" / "episodes_extras.jsonl", "r", encoding="utf-8") as file:
                extras = [json.loads(line) for line in file if line.strip()]

            self.assertEqual(info["fps"], 12)
            self.assertEqual(info["robot_type"], "mock_uav")
            self.assertEqual(info["features"]["video.front"]["shape"], [4, 5, 3])
            self.assertEqual(info["features"]["video.front"]["info"]["video.pix_fmt"], "yuv444p")
            self.assertEqual(tasks, [{"task_index": 0, "task": ""}])
            self.assertEqual(len(extras), 2)
            self.assertEqual(extras[0]["trajectory_id"], "traj_0")
            self.assertEqual(extras[1]["trajectory_id"], "traj_1")

            parquet_files = sorted((root / "data").glob("**/*.parquet"))
            self.assertEqual(len(parquet_files), 2)

            first_episode = pd.read_parquet(parquet_files[0])
            state = np.stack(first_episode["observation.state"].to_numpy())
            action = np.stack(first_episode["action"].to_numpy())

            expected = np.array(
                [
                    [0, 0, 0, 0, 0, 0, 1],
                    [1, 2, 3, 0, 0, 0, 1],
                ],
                dtype=np.float32,
            )

            np.testing.assert_allclose(state, expected, atol=1e-6)
            np.testing.assert_allclose(action, expected, atol=1e-6)

    def test_load_rgb_image_scales_normalized_float_arrays(self):
        image = np.full((2, 3, 3), 0.5, dtype=np.float32)
        loaded = load_rgb_image(image)

        self.assertEqual(loaded.dtype, np.uint8)
        self.assertTrue(np.all(loaded == 127))

    def test_select_video_pixel_format_keeps_hevc_odd_resolution_encodable(self):
        self.assertEqual(select_video_pixel_format((4, 5), codec="hevc", pix_fmt="auto"), "yuv444p")


if __name__ == "__main__":
    unittest.main()
