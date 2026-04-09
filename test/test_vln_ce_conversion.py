import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from utils.coordinate import homogeneous_inv
from utils.vln_ce.trajectory import VLN_CE_Trajectories


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_SAMPLE_ROOT = REPO_ROOT / "tmp" / "r2r-test"
VIEWPOINTS = [
    "125cm_0deg",
    "125cm_30deg",
    "125cm_45deg",
    "60cm_15deg",
    "60cm_30deg",
]
ATOL = 1e-4


def _extract_pitch_deg(viewpoint: str) -> float:
    return float(viewpoint.split("_")[1].removesuffix("deg"))


def _expected_camera_body_transforms(viewpoint: str) -> tuple[np.ndarray, np.ndarray]:
    pitch_deg = _extract_pitch_deg(viewpoint)
    pitch_rad = np.deg2rad(pitch_deg)
    cos_theta = np.cos(pitch_rad)
    sin_theta = np.sin(pitch_rad)

    r_b_c_0 = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float32)
    r_pitch_body = np.array([
        [cos_theta, 0.0, sin_theta],
        [0.0, 1.0, 0.0],
        [-sin_theta, 0.0, cos_theta],
    ], dtype=np.float32)
    r_b_c = r_pitch_body @ r_b_c_0
    r_c_b = r_b_c.T

    t_b_c = np.eye(4, dtype=np.float32)
    t_b_c[:3, :3] = r_b_c

    t_c_b = np.eye(4, dtype=np.float32)
    t_c_b[:3, :3] = r_c_b
    return t_b_c, t_c_b


def _poses_from_transform(t: np.ndarray) -> np.ndarray:
    rotation = t[:, :3, :3]
    position = t[:, :3, 3]
    quat = Rotation.from_matrix(rotation).as_quat().astype(np.float32)
    return np.concatenate([position, quat], axis=1).astype(np.float32)


def _identity_pose() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _expected_intrinsics(width: int, height: int) -> np.ndarray:
    hfov_rad = np.deg2rad(79.0)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    vfov_rad = 2.0 * np.arctan((height / width) * np.tan(hfov_rad / 2.0))
    fy = (height / 2.0) / np.tan(vfov_rad / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def _load_raw_episode(sample_root: Path, viewpoint: str) -> tuple[pd.DataFrame, np.ndarray]:
    scene_dir = next(p for p in sample_root.iterdir() if p.is_dir())
    parquet_path = next(scene_dir.glob("data/chunk-*/*.parquet"))
    df = pd.read_parquet(parquet_path)
    t_w_c = np.stack([np.stack(p) for p in df[f"pose.{viewpoint}"].to_numpy()]).astype(np.float32)
    return df, t_w_c


def _manual_expected(sample_root: Path, viewpoint: str) -> dict[str, np.ndarray]:
    _, t_w_c = _load_raw_episode(sample_root, viewpoint)
    _, t_c_b = _expected_camera_body_transforms(viewpoint)

    t_w_b = t_w_c @ t_c_b
    t_body_w = homogeneous_inv(t_w_b[0])
    t_body_b = np.einsum("ij,njk->nik", t_body_w, t_w_b)

    t_current_next = homogeneous_inv(t_body_b[:-1]) @ t_body_b[1:]
    action_deltas = _poses_from_transform(t_current_next)
    action_deltas = np.concatenate(
        [action_deltas, _identity_pose()[None, :]],
        axis=0,
    )

    n = len(t_body_b)
    target_indices = np.minimum(np.arange(n) + 20, n - 1)
    t_b_goal = homogeneous_inv(t_body_b) @ t_body_b[target_indices]
    action_goals = _poses_from_transform(t_b_goal)

    return {
        "poses_body": _poses_from_transform(t_body_b),
        "action_deltas": action_deltas,
        "action_goals": action_goals,
        "target_indices": target_indices,
    }


@pytest.fixture(scope="session")
def sample_root() -> Path:
    if not RAW_SAMPLE_ROOT.exists():
        pytest.skip(f"Sample dataset missing: {RAW_SAMPLE_ROOT}")
    return RAW_SAMPLE_ROOT


@pytest.mark.parametrize(
    ("viewpoint", "expected_optical_axis"),
    [
        ("125cm_0deg", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ("125cm_30deg", np.array([0.8660254, 0.0, -0.5], dtype=np.float32)),
        ("125cm_45deg", np.array([0.70710677, 0.0, -0.70710677], dtype=np.float32)),
        ("60cm_15deg", np.array([0.9659258, 0.0, -0.25881904], dtype=np.float32)),
    ],
)
def test_camera_body_rotation(viewpoint: str, expected_optical_axis: np.ndarray):
    t_b_c, t_c_b = _expected_camera_body_transforms(viewpoint)

    optical_axis = t_b_c[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right_axis = t_b_c[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert np.allclose(optical_axis, expected_optical_axis, atol=ATOL)
    assert np.allclose(right_axis, np.array([0.0, -1.0, 0.0], dtype=np.float32), atol=ATOL)
    assert np.allclose(t_c_b, np.linalg.inv(t_b_c), atol=ATOL)


@pytest.mark.parametrize("viewpoint", VIEWPOINTS)
def test_trajectory_matches_manual_geometry(sample_root: Path, viewpoint: str):
    traj = next(iter(VLN_CE_Trajectories(str(sample_root), get_task_idx=lambda _: 0, viewpoint=viewpoint)))
    traj._process_traj()
    expected = _manual_expected(sample_root, viewpoint)

    assert len(traj.images) == traj.length == len(expected["poses_body"])
    assert np.allclose(traj.T_b_c, _expected_camera_body_transforms(viewpoint)[0], atol=ATOL)
    assert np.allclose(traj.camera_intrinsics, _expected_intrinsics(traj.image_width, traj.image_height), atol=ATOL)
    assert np.allclose(traj.poses_body, expected["poses_body"], atol=ATOL)
    assert np.allclose(traj.action_deltas, expected["action_deltas"], atol=ATOL)
    assert np.allclose(traj.action_goals, expected["action_goals"], atol=ATOL)
    assert np.allclose(traj.metadata["video.front.body_from_camera"], traj.T_b_c, atol=ATOL)
    assert np.allclose(traj.metadata["video.front.K"], traj.camera_intrinsics, atol=ATOL)
    assert np.allclose(traj.poses_body[0], _identity_pose(), atol=ATOL)
    assert np.allclose(traj.action_deltas[-1], _identity_pose(), atol=ATOL)
    assert np.allclose(traj.action_goals[-1], _identity_pose(), atol=ATOL)


@pytest.mark.parametrize("viewpoint", VIEWPOINTS)
def test_end_to_end_conversion_outputs(sample_root: Path, tmp_path: Path, viewpoint: str):
    output_dir = tmp_path / viewpoint
    cmd = [
        sys.executable,
        str(REPO_ROOT / "vln_ce.py"),
        "--raw_dir",
        str(sample_root),
        "--output_dir",
        str(output_dir),
        "--num_processes",
        "1",
        "--viewpoint",
        viewpoint,
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"Conversion failed for {viewpoint}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    dataset_root = output_dir / "VLN-CE-r2r-test"
    data_file = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    video_file = dataset_root / "videos" / "chunk-000" / "video.front" / "episode_000000.mp4"
    meta_file = dataset_root / "meta" / "info.json"
    extras_file = dataset_root / "meta" / "episodes_extras.jsonl"

    assert dataset_root.exists()
    assert data_file.exists()
    assert video_file.exists()
    assert meta_file.exists()
    assert extras_file.exists()

    with open(meta_file, "r") as f:
        info = json.load(f)
    assert info["total_episodes"] == 1

    converted_df = pd.read_parquet(data_file)
    expected = _manual_expected(sample_root, viewpoint)

    assert len(converted_df) == len(expected["poses_body"])
    assert np.allclose(
        np.stack(converted_df["observation.state"].to_numpy()),
        expected["poses_body"],
        atol=ATOL,
    )
    assert np.allclose(
        np.stack(converted_df["action"].to_numpy())[:, :7],
        expected["action_deltas"],
        atol=ATOL,
    )
    assert np.allclose(
        np.stack(converted_df["action"].to_numpy())[:, 7:],
        expected["action_goals"],
        atol=ATOL,
    )

    with open(extras_file, "r") as f:
        extras = json.loads(next(f))
    scene_dir = next(p for p in sample_root.iterdir() if p.is_dir())
    sample_image = next((scene_dir / "videos" / "chunk-000" / f"observation.images.rgb.{viewpoint}").glob("*.jpg"))
    from PIL import Image
    with Image.open(sample_image) as image:
        width, height = image.size
    assert np.allclose(np.array(extras["video.front.K"], dtype=np.float32), _expected_intrinsics(width, height), atol=ATOL)
    assert np.allclose(
        np.array(extras["video.front.body_from_camera"], dtype=np.float32),
        _expected_camera_body_transforms(viewpoint)[0],
        atol=ATOL,
    )
