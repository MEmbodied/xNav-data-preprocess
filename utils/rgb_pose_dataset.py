"""RGB + pose data source abstraction and LeRobot export pipeline.

This module exists to separate two concerns that were previously coupled:

1. Source-specific parsing
   Different datasets may store trajectories in very different layouts:
   nested folders, parquet files, JSON manifests, image sequences, databases,
   or anything else. That variability should stay inside a small source
   adapter.

2. Uniform LeRobot serialization
   Once a trajectory point is normalized into:
   - one RGB image
   - one camera pose as a 7D vector [tx, ty, tz, qx, qy, qz, qw]
   - dataset-level metadata such as fps, resolution and T_body<-camera
   the rest of the export logic should be shared.

Design overview
---------------

The module defines a three-layer abstraction:

- `RGBPosePoint`
  One trajectory point. It carries the RGB frame and the camera pose in the
  world frame. This is the minimum unit a source adapter needs to yield.

- `RGBPoseTrajectory`
  One trajectory, i.e. an iterable of `RGBPosePoint`. A source may optionally
  provide per-trajectory metadata and task text here.

- `RGBPoseTrajectorySource`
  A dataset-level adapter. It exposes:
  - `info`: shared metadata for the whole source
  - `__iter__`: all trajectories
  - `__len__`: number of trajectories

`RGBPoseSourceInfo` contains the information needed by the shared exporter:

- `fps`
  Stored into LeRobot metadata.
- `image_size`
  The raw RGB resolution `(H, W)`. Images are expected to be exported without
  resizing.
- `body_from_camera`
  The fixed extrinsic transform `T_body<-camera`.
  If a point provides `T_world<-camera`, the exporter derives:
  `T_world<-body = T_world<-camera @ T_camera<-body`,
  where `T_camera<-body = inv(T_body<-camera)`.

Export flow
-----------

`export_rgb_pose_source()` is the shared entry point:

1. Instantiate the source adapter.
2. Read source-level metadata (`fps`, resolution, extrinsic, robot type).
3. Build the LeRobot feature schema.
4. Convert each point from camera pose to body pose.
5. Write frames through `LeRobotCreator`.

Current field mapping
---------------------

For each point, the exporter writes:

- `annotation.human.action.task_description`
  Task index, resolved through LeRobot task metadata.
- `video.ego_view`
  Raw RGB image.
- `observation.state`
  Body pose as a 7D vector `[tx, ty, tz, qx, qy, qz, qw]`.
- `action`
  Currently identical to `observation.state`, matching the task requirement in
  `task.md`.

How to extend
-------------

To support a new raw dataset, the expected implementation pattern is:

1. Implement one `RGBPoseTrajectorySource` subclass.
2. Return `RGBPoseSourceInfo` from `info`.
3. Yield custom `RGBPoseTrajectory` objects from `__iter__`.
4. In each trajectory, yield normalized `RGBPosePoint` objects.

That keeps all source-specific logic isolated while reusing one common
LeRobot export pipeline.
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from utils.coordinate import homogeneous_inv
from utils.lerobot.lerobot_creater import LeRobotCreator

TASK_DESCRIPTION_KEY = "annotation.human.action.task_description"
VIDEO_KEY = "video.ego_view"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
POSE_AXES = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]


def _json_default(value: Any):
    """Serialize common numpy scalars/arrays for source metadata JSON files."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_numpy_float32(array: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Convert arbitrary input into a float32 numpy array with a strict shape.

    Important for this module because source adapters may return lists, tuples,
    numpy arrays or other array-like objects. The shared exporter wants all pose
    math to happen on predictable `float32` tensors with known dimensions.
    """
    value = np.asarray(array, dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return value


def _normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Normalize a quaternion in `[qx, qy, qz, qw]` convention.

    The exporter assumes all poses use the xyzw convention required by
    `scipy.spatial.transform.Rotation.from_quat`.

    We also flip the sign when `qw < 0`. Quaternions `q` and `-q` encode the
    same rotation; forcing a consistent hemisphere avoids unnecessary sign
    flips in exported data.
    """
    norm = np.linalg.norm(quaternion)
    if norm <= 0:
        raise ValueError("Quaternion norm must be positive.")

    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion.astype(np.float32)


def pose_vector_to_transform(pose: Any) -> np.ndarray:
    """Convert a 7D pose vector `[tx, ty, tz, qx, qy, qz, qw]` to a 4x4 transform.

    Convention used throughout this file:
    - translation is expressed in meters
    - quaternion is xyzw
    - the returned matrix is a standard homogeneous transform
    """
    pose = _to_numpy_float32(pose, (7,), "pose")
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rotation.from_quat(_normalize_quaternion_xyzw(pose[3:])).as_matrix().astype(np.float32)
    transform[:3, 3] = pose[:3]
    return transform


def transform_to_pose_vector(transform: Any) -> np.ndarray:
    """Convert a 4x4 homogeneous transform back to a 7D pose vector.

    This is the inverse helper of `pose_vector_to_transform()`. The returned
    quaternion is normalized and canonicalized through `_normalize_quaternion_xyzw`.
    """
    transform = _to_numpy_float32(transform, (4, 4), "transform")
    translation = transform[:3, 3]
    quaternion = _normalize_quaternion_xyzw(Rotation.from_matrix(transform[:3, :3]).as_quat().astype(np.float32))
    return np.concatenate([translation, quaternion], axis=0).astype(np.float32)


def camera_pose_to_body_pose(camera_pose: Any, body_from_camera: Any) -> np.ndarray:
    """Convert camera pose to body pose using a fixed extrinsic transform.

    Args:
        camera_pose:
            `T_world<-camera` represented as `[tx, ty, tz, qx, qy, qz, qw]`.
        body_from_camera:
            Fixed extrinsic `T_body<-camera`.

    Returns:
        Body pose `T_world<-body` in the same 7D vector format.

    Derivation:
        T_world<-body = T_world<-camera @ T_camera<-body
        T_camera<-body = inv(T_body<-camera)
    """
    world_from_camera = pose_vector_to_transform(camera_pose)
    body_from_camera = _to_numpy_float32(body_from_camera, (4, 4), "body_from_camera")
    camera_from_body = homogeneous_inv(body_from_camera)
    world_from_body = world_from_camera @ camera_from_body
    return transform_to_pose_vector(world_from_body)


def load_rgb_image(rgb: np.ndarray | Image.Image | str | Path) -> np.ndarray:
    """Normalize an RGB input into a `uint8` numpy array of shape `[H, W, 3]`.

    Accepted inputs:
    - already loaded numpy arrays
    - PIL images
    - filesystem paths

    Notes:
    - File paths are loaded lazily here so source adapters may return image
      paths instead of fully decoded arrays.
    - Float images in `[0, 1]` are automatically scaled to `[0, 255]`.
    - Images are not resized; resolution consistency is validated later against
      `RGBPoseSourceInfo.image_size`.
    """
    if isinstance(rgb, np.ndarray):
        image = rgb
    elif isinstance(rgb, Image.Image):
        image = np.array(rgb.convert("RGB"))
    else:
        with Image.open(rgb) as pil_image:
            image = np.array(pil_image.convert("RGB"))

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB image must have shape [H, W, 3], got {image.shape}")

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0)
        image = image.astype(np.uint8)

    return image


@dataclass(frozen=True)
class RGBPosePoint:
    """Normalized point-level payload produced by a source adapter.

    Attributes:
        rgb:
            One RGB frame. Can be a numpy array, PIL image or a local path.
        camera_pose:
            7D camera pose in world frame `[tx, ty, tz, qx, qy, qz, qw]`.
        metadata:
            Optional per-point metadata. The shared exporter does not currently
            write it, but keeping the slot here avoids re-design when a source
            wants to expose additional point-level information later.
    """
    rgb: np.ndarray | Image.Image | str | Path
    camera_pose: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "camera_pose", _to_numpy_float32(self.camera_pose, (7,), "camera_pose"))


@dataclass(frozen=True)
class RGBPoseSourceInfo:
    """Dataset-level metadata shared by all trajectories in one source adapter.

    This object is the contract between a concrete source implementation and the
    shared LeRobot exporter. It intentionally contains only the metadata that
    should stay constant across the whole export job.
    """
    fps: int
    image_size: tuple[int, int]
    body_from_camera: np.ndarray
    robot_type: str = "lerobot"
    task: str = ""
    source_name: str = ""

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if len(self.image_size) != 2 or any(v <= 0 for v in self.image_size):
            raise ValueError(f"image_size must be (height, width), got {self.image_size}")

        object.__setattr__(self, "image_size", (int(self.image_size[0]), int(self.image_size[1])))
        object.__setattr__(
            self,
            "body_from_camera",
            _to_numpy_float32(self.body_from_camera, (4, 4), "body_from_camera"),
        )


class RGBPoseTrajectory(ABC):
    """Abstract interface for one trajectory.

    A concrete implementation should encapsulate only source-specific reading
    logic, e.g. how to iterate frames inside one episode directory or one row
    group in a parquet file.
    """

    @property
    def task(self) -> str:
        """Optional task string for this trajectory.

        If empty, the exporter falls back to `RGBPoseSourceInfo.task`.
        """
        return ""

    @property
    def metadata(self) -> dict[str, Any]:
        """Optional per-trajectory metadata written to `episodes_extras.jsonl`."""
        return {}

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[RGBPosePoint]:
        raise NotImplementedError


class RGBPoseTrajectorySource(ABC):
    """Abstract dataset-level adapter.

    A new raw dataset format should usually implement this class and nothing
    else outside its own module. The exporter only depends on this interface.
    """

    @property
    @abstractmethod
    def info(self) -> RGBPoseSourceInfo:
        """Shared metadata for the whole source, such as fps and extrinsic."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[RGBPoseTrajectory]:
        raise NotImplementedError


def build_lerobot_rgb_pose_features(image_size: tuple[int, int]) -> dict[str, dict[str, Any]]:
    """Build the LeRobot feature schema for the RGB+pose task.

    The schema is intentionally centralized here so all source adapters export
    exactly the same field layout.
    """
    height, width = image_size
    return {
        TASK_DESCRIPTION_KEY: {
            "dtype": "int32",
            "shape": (1,),
            "names": None,
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "axes": POSE_AXES,
            },
        },
        VIDEO_KEY: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "axes": POSE_AXES,
            },
        },
    }


def select_video_pixel_format(image_size: tuple[int, int], codec: str, pix_fmt: str) -> str:
    """Choose a video pixel format compatible with the raw image resolution.

    Why this helper exists:
    - many codecs with `yuv420p` require even width and height
    - the task explicitly requires storing RGB at original resolution
    - therefore odd-sized inputs should switch to a compatible format instead of
      being silently resized

    When `pix_fmt='auto'`, odd-sized H.264/HEVC exports use `yuv444p`.
    """
    if pix_fmt != "auto":
        return pix_fmt

    if codec in {"h264", "hevc"} and any(size % 2 != 0 for size in image_size):
        logging.warning(
            "Image size %s is not divisible by 2, using yuv444p to keep the original resolution.",
            image_size,
        )
        return "yuv444p"

    return "yuv420p"


class LeRobotRGBPoseEpisode:
    """Bridge one normalized trajectory into the iterator format expected by `LeRobotCreator`.

    `LeRobotCreator.submit_episode()` expects an iterable yielding `(frame, task)`
    tuples. This class performs the last-mile conversion from source-level points
    to concrete LeRobot frame dictionaries.
    """

    def __init__(self, trajectory: RGBPoseTrajectory, source_info: RGBPoseSourceInfo, task_idx: int):
        self.trajectory = trajectory
        self.source_info = source_info
        self.task = trajectory.task if trajectory.task else source_info.task
        self.task_idx = task_idx

    @property
    def metadata(self) -> dict[str, Any]:
        """Expose trajectory metadata so `LeRobotCreator` can persist extras."""
        return dict(self.trajectory.metadata)

    def __iter__(self) -> Iterator[tuple[dict[str, Any], str]]:
        """Yield LeRobot-ready frames for one trajectory.

        Per point, this method:
        1. loads or normalizes the RGB image
        2. validates image resolution against the source-level contract
        3. converts camera pose to body pose using the fixed extrinsic
        4. re-expresses the body pose relative to the first frame's coordinate
           system so that `observation.state` and `action` are in a local frame
           where the first frame is the identity
        5. formats the final frame dict for `LeRobotCreator`
        """
        expected_height, expected_width = self.source_info.image_size
        inv_first_body: np.ndarray | None = None

        for point in self.trajectory:
            rgb = load_rgb_image(point.rgb)
            if rgb.shape[:2] != (expected_height, expected_width):
                raise ValueError(
                    f"RGB image shape mismatch, expected {(expected_height, expected_width)}, got {rgb.shape[:2]}"
                )

            body_pose = camera_pose_to_body_pose(point.camera_pose, self.source_info.body_from_camera)

            # Re-express in the first frame's coordinate system:
            #   T_first<-body_i = inv(T_world<-first) @ T_world<-body_i
            world_from_body = pose_vector_to_transform(body_pose)
            if inv_first_body is None:
                inv_first_body = homogeneous_inv(world_from_body)
            local_pose = transform_to_pose_vector(inv_first_body @ world_from_body)

            frame = {
                TASK_DESCRIPTION_KEY: np.array([self.task_idx], dtype=np.int32),
                STATE_KEY: local_pose,
                VIDEO_KEY: rgb,
                ACTION_KEY: local_pose.copy(),
            }
            yield frame, self.task


def validate_lerobot_dataset(repo_id: str, root: str | Path):
    """Run a minimal post-export integrity check on generated LeRobot files."""
    meta = LeRobotDatasetMetadata(repo_id, root=root)

    if meta.total_episodes == 0:
        raise ValueError("Number of episodes is 0.")

    for episode_index in range(meta.total_episodes):
        data_path = meta.root / meta.get_data_file_path(episode_index)
        if not data_path.exists():
            raise ValueError(f"Parquet file is missing in: {data_path}")

        for video_key in meta.video_keys:
            video_path = meta.root / meta.get_video_file_path(episode_index, video_key)
            if not video_path.exists():
                raise ValueError(f"Video file is missing in: {video_path}")


# def write_source_info(root: str | Path, source_info: RGBPoseSourceInfo):
#     """Persist source-level metadata beside the standard LeRobot metadata.

#     LeRobot metadata stores fps and video shape, but this file additionally
#     keeps source-specific export context such as the fixed body-camera extrinsic.
#     """
#     meta_dir = Path(root) / "meta"
#     meta_dir.mkdir(parents=True, exist_ok=True)
#     with open(meta_dir / "source_info.json", "w", encoding="utf-8") as file:
#         json.dump(source_info.to_jsonable(), file, indent=2, default=_json_default)


def export_rgb_pose_source(
    raw_dir: str | Path,
    repo_id: str,
    root: str | Path,
    source_cls: type[RGBPoseTrajectorySource],
    num_processes: int,
    codec: str = "h264",
    pix_fmt: str = "auto",
    *args,
    **kwargs,
):
    """Shared end-to-end exporter from `RGBPoseTrajectorySource` to LeRobot.

    This is the main API of the module. Typical usage:

    ```python
    export_rgb_pose_source(
        raw_dir="/path/to/raw",
        repo_id="my_dataset",
        root="/path/to/output",
        source_cls=MySource,
        num_processes=8,
    )
    ```

    Responsibilities:
    - instantiate the concrete source adapter
    - initialize LeRobot schema and metadata
    - map every source trajectory into a `LeRobotRGBPoseEpisode`
    - submit all episodes to `LeRobotCreator`
    - validate that parquet and video outputs exist after export
    """
    logging.info("Loading RGB pose source %s from %s", source_cls.__name__, raw_dir)
    source = source_cls(raw_dir, *args, **kwargs)
    source_info = source.info
    # write_source_info(root, source_info)
    resolved_pix_fmt = select_video_pixel_format(source_info.image_size, codec=codec, pix_fmt=pix_fmt)

    creator = LeRobotCreator(
        root=str(root),
        robot_type=source_info.robot_type,
        fps=source_info.fps,
        features=build_lerobot_rgb_pose_features(source_info.image_size),
        num_workers=max(1, num_processes),
        num_video_encoders=max(1, int(max(1, num_processes) * 1.75)),
        codec=codec,
        pix_fmt=resolved_pix_fmt,
        has_extras=True,
    )

    start_time = time.time()
    num_episodes = len(source)
    logging.info("Number of episodes %s", num_episodes)

    for episode_index, trajectory in enumerate(source, start=1):
        task = trajectory.task if trajectory.task else source_info.task
        task_idx = creator.add_task(task)
        creator.submit_episode(LeRobotRGBPoseEpisode(trajectory, source_info, task_idx))

        if episode_index % 10 == 0:
            elapsed = time.time() - start_time
            logging.info("Submitted %s / %s episodes (elapsed %.3f s)", episode_index, num_episodes, elapsed)

    creator.wait()
    validate_lerobot_dataset(repo_id=repo_id, root=root)


def load_source_cls(source_cls_path: str) -> type[RGBPoseTrajectorySource]:
    """Load a source class from import path notation.

    Supported formats:
    - `package.module:ClassName`
    - `package.module.ClassName`

    This is used by the CLI so users can plug in a new source adapter without
    modifying the common exporter script.
    """
    if ":" in source_cls_path:
        module_name, class_name = source_cls_path.split(":", maxsplit=1)
    else:
        module_name, class_name = source_cls_path.rsplit(".", maxsplit=1)

    module = importlib.import_module(module_name)
    source_cls = getattr(module, class_name)
    if not issubclass(source_cls, RGBPoseTrajectorySource):
        raise TypeError(f"{source_cls_path} is not a RGBPoseTrajectorySource subclass.")
    return source_cls
