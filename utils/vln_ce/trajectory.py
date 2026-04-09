from pathlib import Path
from typing import List, Callable
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pyparsing import Iterable
from utils import Traj, Trajectories

from scipy.spatial.transform import Rotation
from utils.coordinate import homogeneous_inv

class VLN_CE_Traj(Traj):
    # 0deg reference: body +x forward/+y left/+z up, camera +x right/+y down/+z forward.
    R_B_C_0 = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ], dtype=np.float32)

    SUPPORTED_VIEWPOINTS = (
        "125cm_0deg",
        "125cm_30deg",
        "125cm_45deg",
        "60cm_15deg",
        "60cm_30deg",
    )

    @property
    def metadata(self) -> dict:
        return {
            "T_b_c": self.T_b_c,
        }

    @staticmethod
    def _extract_pitch_deg(viewpoint: str) -> float:
        try:
            return float(viewpoint.split("_")[1].removesuffix("deg"))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid viewpoint format: {viewpoint}") from exc

    @classmethod
    def _build_camera_body_transforms(cls, viewpoint: str) -> tuple[np.ndarray, np.ndarray]:
        pitch_deg = cls._extract_pitch_deg(viewpoint)
        pitch_rad = np.deg2rad(pitch_deg)
        cos_theta = np.cos(pitch_rad)
        sin_theta = np.sin(pitch_rad)

        # Positive pitch means the camera tilts downward around body +y.
        R_pitch_body = np.array([
            [cos_theta, 0.0, sin_theta],
            [0.0, 1.0, 0.0],
            [-sin_theta, 0.0, cos_theta],
        ], dtype=np.float32)
        R_b_c = R_pitch_body @ cls.R_B_C_0
        R_c_b = R_b_c.T

        T_b_c = np.eye(4, dtype=np.float32)
        T_b_c[:3, :3] = R_b_c

        T_c_b = np.eye(4, dtype=np.float32)
        T_c_b[:3, :3] = R_c_b
        return T_b_c, T_c_b

    def __init__(self, parquet_path: Path, images: List[Path], task: str, task_idx: int, viewpoint: str):
        self.parquet_path = parquet_path
        self.images = images
        self.task = task
        self.task_idx = task_idx
        self.viewpoint = viewpoint
        self.T_b_c, self.T_c_b = self._build_camera_body_transforms(viewpoint)
        self.pose_col = f"pose.{viewpoint}"
        self.goal_key = f"relative_goal_frame_id.{viewpoint}"
        self.reason_col = f"{viewpoint}_reason"

        # Read parquet
        self.df = pd.read_parquet(self.parquet_path)

        if self.pose_col not in self.df.columns:
            raise KeyError(f"Missing pose column {self.pose_col} in {self.parquet_path}")

        assert len(self.df) == len(self.images), \
            f"Length mismatch between parquet {len(self.df)} and images {len(self.images)} in {self.parquet_path}"

        self.length = len(self.df)

    def _process_traj(self):
        # Process Poses
        # Ensure it's a stacked numpy array [N, 4, 4]
        # df[f"pose.{viewpoint}"] contains per-frame camera poses
        pose_col = self.df[self.pose_col].to_numpy()
        T_w_c = np.array([np.stack(p) for p in pose_col])
        
        # Calculate Body Poses
        T_w_b = T_w_c @ self.T_c_b
        
        # First frame as body frame origin for state
        self.T_w_body_0 = T_w_b[0]
        T_body_w = homogeneous_inv(self.T_w_body_0) # body means first frame's body coordinate
        
        # Transform all frames to be relative to the first frame (body frame)
        # [4, 4] @ [N, 4, 4] -> [N, 4, 4]
        T_body_b = np.einsum('ij,njk->nik', T_body_w, T_w_b)
        self.poses_body = self.get_poses(T_body_b) # observation.state
        
        # Calculate Action Deltas (first 4 values)
        # Delta is relative pose from current to next frame
        T_body_b_current = T_body_b[:-1]
        T_b_current_body = homogeneous_inv(T_body_b_current)
        T_body_b_next = T_body_b[1:]
        
        # [N-1, 4, 4]
        T_current_next = T_b_current_body @ T_body_b_next
        deltas = self.get_poses(T_current_next)
        
        # Pad last frame with identity (no movement)
        last_delta = np.zeros((1, 4), dtype=np.float32) 
        self.action_deltas = np.concatenate([deltas, last_delta], axis=0) # [N, 4]
        
        # Calculate Goal Actions (last 4 values) using a fixed lookahead window.
        N = self.length
        target_indices = np.minimum(np.arange(N) + 20, N - 1)
        
        T_body_goal = T_body_b[target_indices] # [N, 4, 4]
        T_b_body = homogeneous_inv(T_body_b) # [N, 4, 4]
        T_b_goal = T_b_body @ T_body_goal # [N, 4, 4]
        self.action_goals = self.get_poses(T_b_goal) # [N, 4]

    def get_poses(self, T: np.ndarray) -> np.ndarray:
        """
        Get poses in [x, y, z, yaw] format.
        Args:
            T: [N, 4, 4]
        Returns:
            poses: [N, 4]
        """
        # Yaw extraction (ZYX euler)
        R = T[:, :3, :3]
        yaw, pitch, roll = Rotation.from_matrix(R).as_euler('ZYX', degrees=True).T
        pos = T[:, :3, 3]
        return np.concatenate([pos, yaw[:, None]], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return self.length
    
    def __iter__(self) -> Iterable[tuple[dict, str]]:
        self._process_traj()
        has_reason = self.reason_col in self.df.columns
        for i in range(self.length):
            reason = str(self.df.iloc[i][self.reason_col]) if has_reason else ""
            if pd.isna(reason):
                reason = ""
            frame = {
                "annotation.human.action.task_description": np.array([self.task_idx], dtype=np.int32),
                "observation.state": self.poses_body[i],
                "video.ego_view": np.array(Image.open(self.images[i]).convert("RGB")),
                "action": np.concatenate([self.action_deltas[i], self.action_goals[i]]).astype(np.float32),
                "extra.cot": reason,
            }
            yield frame, self.task


class VLN_CE_Trajectories(Trajectories):
    FPS: int = 10
    ROBOT_TYPE: str = "lerobot"
    INSTRUCTION_KEY: str = "annotation.human.action.task_description"


    FEATURES = {
        # The language instruction for the task.
        "annotation.human.action.task_description": {
            "dtype": "int32", # index of task
            "shape": (1,),
            "names": None,
        },
        # The drone's pose in the first frame of the trajectory.
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": {
                "axes": ["x", "y", "z", "yaw"],
            },
        },
        # The primary video feed from the drone's ego-centric camera.
        "video.ego_view": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        },
        # The action command sent to the drone.
        # first 4 values are [dx, dy, dz, dyaw] to the next frame
        # last 4 values are goal pose of the to the current frame
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "axes": ["x", "y", "z", "yaw", "farthest_x", "farthest_y", "farthest_z", "farthest_yaw"],
            },
        },
        # Per-frame chain-of-thought reasoning from CoT annotations.
        "extra.cot": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
    }

    def __init__(self, data_path: str, get_task_idx: Callable[[str], int], viewpoint: str = "125cm_0deg"):
        self.data_path = Path(data_path)
        self.get_task_idx = get_task_idx
        if viewpoint not in VLN_CE_Traj.SUPPORTED_VIEWPOINTS:
            raise ValueError(
                f"Unsupported viewpoint {viewpoint}. Supported: {VLN_CE_Traj.SUPPORTED_VIEWPOINTS}"
            )
        self.viewpoint = viewpoint
        self.parquet_files = []
        
        # Find all 'data' directories.
        # Structure is assumed to be .../scene_id/data/chunk-XXX/episode.parquet
        # We verify that 'data' is a sibling of 'meta' and 'videos' to confirm it's a valid scene directory.
        for data_dir in tqdm(self.data_path.rglob("data"), desc="Scanning data directories"):
            if not data_dir.is_dir():
                continue
            
            scene_dir = data_dir.parent
            if (scene_dir / "meta").exists() and (scene_dir / "videos").exists():
                for chunk_dir in data_dir.glob("chunk-*"):
                    if chunk_dir.is_dir():
                        self.parquet_files.extend(chunk_dir.glob("*.parquet"))

    def __len__(self) -> int:
        return len(self.parquet_files)

    def __iter__(self) -> Iterable[Traj]:
        # Cache for metadata: path -> dict of episode_index -> task
        metadata_cache = {}

        for parquet_file in self.parquet_files:
            # We enforce the structure: .../scene_id/data/chunk-XXX/episode_XXXXXX.parquet
            # So traversing up 3 levels gives us the scene directory.
            chunk_dir = parquet_file.parent
            data_dir = chunk_dir.parent
            scene_dir = data_dir.parent 

            # Load metadata
            meta_file = scene_dir / "meta" / "episodes.jsonl"
            meta_key = str(meta_file)
            if meta_key not in metadata_cache:
                episodes_map = {}
                if meta_file.exists():
                    with open(meta_file, "r") as f:
                        for line in f:
                            try:
                                item = json.loads(line)
                                # task is a list of strings, we take the first one
                                if "tasks" in item and len(item["tasks"]) > 0:
                                    episodes_map[item["episode_index"]] = item["tasks"][0]
                            except Exception:
                                continue
                metadata_cache[meta_key] = episodes_map

            # Get episode index from filename (episode_000000.parquet)
            try:
                # Format is episode_XXXXXX.parquet
                episode_idx_str = parquet_file.stem.split("_")[1]
                episode_idx = int(episode_idx_str)
            except (IndexError, ValueError):
                continue

            task = metadata_cache[meta_key].get(episode_idx, "")
            task_idx = self.get_task_idx(task)

            # Get images
            # .../videos/chunk-XXX/
            chunk_name = chunk_dir.name
            video_chunk_dir = scene_dir / "videos" / chunk_name
            
            # Primary path as per instruction
            images_dir = video_chunk_dir / f"observation.images.rgb.{self.viewpoint}"

            # Image pattern: episode_000000_0.jpg
            # The last number is the frame index
            image_prefix = f"episode_{episode_idx_str}_"
            images = sorted(
                list(images_dir.glob(f"{image_prefix}*.jpg")),
                key=lambda p: int(p.stem.split("_")[-1])
            )

            try:
                traj = VLN_CE_Traj(
                    parquet_file,
                    images,
                    task=task,
                    task_idx=task_idx,
                    viewpoint=self.viewpoint,
                )
                yield traj
            except Exception:
                print(f"Failed to load trajectory from {parquet_file}")
                import traceback
                traceback.print_exc()
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"Failed to load trajectory from {parquet_file}\n")
                    log_file.write(traceback.format_exc())
                    log_file.write("\n")
                continue

    @property
    def schema(self) -> dict:
        return VLN_CE_Trajectories.FEATURES

if __name__ == "__main__":
    def get_task_idx_mock(task: str) -> int:
        return 0
    
    trajs = VLN_CE_Trajectories("/data-10T/InternData-N1/r2r", get_task_idx=get_task_idx_mock)

    i = 0
    for traj in trajs:
        i += 1
        if i >= 5:
            break
