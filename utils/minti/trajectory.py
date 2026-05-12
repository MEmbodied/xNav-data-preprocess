"""Minti dataset trajectory adapter.

Source data layout (one scene):
    scene_dir/
        episode_XXXXXX/
            episode_meta.json
            events.jsonl
            frames.jsonl         # per-frame body pose & camera pose
            task_info.csv
            rgb/front/NNNNN.png  # first-person camera image
            depth/...            # ignored

Conventions in source data (UE):
    - World/body/camera frames are all LEFT-HANDED, UE convention:
        body:   +x forward, +y right, +z up
        camera: +x camera-forward, +y right, +z up
    - frames.jsonl `pose` is body in world: [x, y, z, roll, pitch, yaw]
      with xyz in cm, rpy in degrees, ZYX intrinsic rotation order.
    - `camera_pose_front` follows the same convention.

Target dataset conventions (matching vln_ce):
    - body:   +x forward, +y left, +z up (right-handed)
    - camera: OpenCV: +x right, +y down, +z forward (right-handed)
    - units: meters, degrees (for yaw output only)
    - observation.state: [x, y, z, yaw] relative to first frame's body
    - action: [dx, dy, dz, dyaw, 0, 0, 0, 0]
"""

from pathlib import Path
from typing import Callable, Iterable, List
import json

import numpy as np
import pandas as pd  # noqa: F401 (kept for symmetry with vln_ce)
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils import Traj, Trajectories
from utils.coordinate import homogeneous_inv


# Mirror matrix that flips the y-axis. UE left-handed (+x前 +y右 +z上)
# becomes target right-handed (+x前 +y左 +z上) by negating y.
_M = np.diag([1.0, -1.0, 1.0]).astype(np.float64)

# Rotation that maps OpenCV camera axes (in target convention) to body axes.
# +x_cam (right)   -> -y_body
# +y_cam (down)    -> -z_body
# +z_cam (forward) -> +x_body
_R_BODY_FROM_CAM_OPENCV = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

# Rotation that maps UE-camera axes (+x cam-forward, +y right, +z up) to
# UE-body axes (+x forward, +y right, +z up). When body and camera have the
# same orientation (as is the case in this dataset), this is identity.
_R_UE_BODY_FROM_CAM = np.eye(3, dtype=np.float64)


def _ue_pose_to_T_world(pose: List[float]) -> np.ndarray:
    """Convert a UE pose [x, y, z, roll, pitch, yaw] (cm, deg, ZYX intrinsic)
    into a 4x4 transform in the UE (left-handed) world frame."""
    x, y, z, roll, pitch, yaw = pose
    # scipy Rotation 'ZYX' intrinsic expects [yaw, pitch, roll]
    R = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def _ue_world_T_to_target(T_ue: np.ndarray, cm_to_m: bool = True) -> np.ndarray:
    """Apply axis remap M=diag(1,-1,1) to convert a homogeneous transform from
    UE left-handed frame to target right-handed frame.

    For a transform T = [[R, t], [0, 1]] in UE, the equivalent transform in
    target coordinates is:
        R' = M @ R @ M
        t' = M @ t
    """
    R = T_ue[:3, :3]
    t = T_ue[:3, 3]
    R_t = _M @ R @ _M
    t_t = _M @ t
    if cm_to_m:
        t_t = t_t / 100.0
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_t
    T[:3, 3] = t_t
    return T


def _poses_to_xyzyaw(T: np.ndarray) -> np.ndarray:
    """Extract [x, y, z, yaw_deg] from a batch of [N, 4, 4] transforms."""
    R = T[:, :3, :3]
    yaw, _, _ = Rotation.from_matrix(R).as_euler("ZYX", degrees=True).T
    pos = T[:, :3, 3]
    return np.concatenate([pos, yaw[:, None]], axis=1).astype(np.float32)


class MintiTraj(Traj):
    # Filled in __init__ from per-episode data
    def __init__(
        self,
        episode_dir: Path,
        frames: List[dict],
        images: List[Path],
        task: str,
        task_idx: int,
    ):
        self.episode_dir = episode_dir
        self.frames = frames
        self.images = images
        self.task = task
        self.task_idx = task_idx

        assert len(self.frames) == len(self.images), (
            f"Length mismatch between frames.jsonl ({len(self.frames)}) and "
            f"rgb/front ({len(self.images)}) in {episode_dir}"
        )
        self.length = len(self.frames)

        self._T_b_c = None  # set in _process_traj

    @property
    def metadata(self) -> dict:
        if self._T_b_c is None:
            self._process_traj()
        return {
            "T_b_c": self._T_b_c.astype(np.float32),
        }

    def _process_traj(self):
        N = self.length

        # Build per-frame T_w_b and T_w_c in target frame
        T_w_b = np.zeros((N, 4, 4), dtype=np.float64)
        T_w_c = np.zeros((N, 4, 4), dtype=np.float64)
        for i, f in enumerate(self.frames):
            T_w_b_ue = _ue_pose_to_T_world(f["pose"])
            T_w_c_ue = _ue_pose_to_T_world(f["camera_pose_front"])
            T_w_b[i] = _ue_world_T_to_target(T_w_b_ue)
            T_w_c[i] = _ue_world_T_to_target(T_w_c_ue)

        # Express everything relative to first frame's body
        T_body0_w = homogeneous_inv(T_w_b[0])
        T_body_b = np.einsum("ij,njk->nik", T_body0_w, T_w_b)
        T_body_c = np.einsum("ij,njk->nik", T_body0_w, T_w_c)

        # T_b_c (camera in body frame) — compute from first frame for metadata.
        # Rotation part should match the standard OpenCV->body rotation; the
        # translation captures the static mounting offset.
        T_b_c_first = homogeneous_inv(T_w_b[0]) @ T_w_c[0]
        # Convert the source camera frame (UE: +x cam-forward, +y right, +z up)
        # to OpenCV camera frame (+x right, +y down, +z forward) by composing
        # with the appropriate axis remap on the right.
        # In target world coords, the source camera frame becomes UE-body-like
        # (same axes as body). To re-express as an OpenCV camera frame we
        # right-multiply by R_body_from_camera (i.e. interpret the existing
        # 'camera' frame's axes via OpenCV convention).
        R_cam_remap = np.eye(4, dtype=np.float64)
        R_cam_remap[:3, :3] = _R_BODY_FROM_CAM_OPENCV
        T_b_c = T_b_c_first @ R_cam_remap
        self._T_b_c = T_b_c

        # observation.state
        self.poses_body = _poses_to_xyzyaw(T_body_b)

        # action deltas (i -> i+1)
        T_b_cur_body = homogeneous_inv(T_body_b[:-1])
        T_cur_next = np.einsum("nij,njk->nik", T_b_cur_body, T_body_b[1:])
        deltas = _poses_to_xyzyaw(T_cur_next)
        last = np.zeros((1, 4), dtype=np.float32)
        self.action_deltas = np.concatenate([deltas, last], axis=0)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterable[tuple[dict, str]]:
        self._process_traj()
        zeros4 = np.zeros(4, dtype=np.float32)
        for i in range(self.length):
            frame = {
                "annotation.human.action.task_description": np.array(
                    [self.task_idx], dtype=np.int32
                ),
                "observation.state": self.poses_body[i],
                "video.ego_view": np.array(Image.open(self.images[i]).convert("RGB")),
                "action": np.concatenate(
                    [self.action_deltas[i], zeros4]
                ).astype(np.float32),
                "extra.cot": "",
            }
            yield frame, self.task


class MintiTrajectories(Trajectories):
    FPS: int = 10
    ROBOT_TYPE: str = "lerobot"
    INSTRUCTION_KEY: str = "annotation.human.action.task_description"

    # Match vln_ce schema. Image size 640x480 from episode_meta.json
    # capture_width / capture_height.
    FEATURES = {
        "annotation.human.action.task_description": {
            "dtype": "int32",
            "shape": (1,),
            "names": None,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"axes": ["x", "y", "z", "yaw"]},
        },
        "video.ego_view": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "axes": [
                    "x", "y", "z", "yaw",
                    "farthest_x", "farthest_y", "farthest_z", "farthest_yaw",
                ]
            },
        },
        "extra.cot": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
    }

    def __init__(self, data_path: str, get_task_idx: Callable[[str], int]):
        self.data_path = Path(data_path)
        self.get_task_idx = get_task_idx

        # All episode_XXXXXX directories with frames.jsonl + rgb/front
        self.episode_dirs: List[Path] = []
        for ep_dir in sorted(self.data_path.glob("episode_*")):
            if not ep_dir.is_dir():
                continue
            if not (ep_dir / "frames.jsonl").exists():
                continue
            if not (ep_dir / "rgb" / "front").is_dir():
                continue
            self.episode_dirs.append(ep_dir)

    def __len__(self) -> int:
        return len(self.episode_dirs)

    def __iter__(self) -> Iterable[Traj]:
        for ep_dir in tqdm(self.episode_dirs, desc="Episodes"):
            try:
                frames = []
                with open(ep_dir / "frames.jsonl", "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        frames.append(json.loads(line))

                if len(frames) == 0:
                    print(f"Skipping empty episode {ep_dir}")
                    continue

                images = sorted(
                    (ep_dir / "rgb" / "front").glob("*.png"),
                    key=lambda p: int(p.stem),
                )

                # Trim to min length, just in case images and frames disagree.
                n = min(len(frames), len(images))
                frames = frames[:n]
                images = images[:n]

                task = ""
                task_idx = self.get_task_idx(task)

                yield MintiTraj(
                    episode_dir=ep_dir,
                    frames=frames,
                    images=images,
                    task=task,
                    task_idx=task_idx,
                )
            except Exception:
                import traceback
                print(f"Failed to load trajectory from {ep_dir}")
                traceback.print_exc()
                continue

    @property
    def schema(self) -> dict:
        return MintiTrajectories.FEATURES
