"""
Visualize a single episode from an RGB-Pose LeRobot dataset.

Produces a side-by-side video:
  - Left: 3D trajectory of the body pose (observation.state)
  - Right: Original RGB front-view video

Usage:
    uv run examples/rgb_pose_example/visualize_episode.py \
        --dataset_dir ./tmp/sekai-test \
        --episode_index 0 \
        --output visualize_ep0.mp4
"""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation


def load_episode(dataset_dir: Path, episode_index: int):
    """Load parquet data and video path for one episode."""
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunk_size = info["chunks_size"]
    chunk_idx = episode_index // chunk_size

    parquet_path = dataset_dir / f"data/chunk-{chunk_idx:03d}/episode_{episode_index:06d}.parquet"
    video_path = dataset_dir / f"videos/chunk-{chunk_idx:03d}/video.front_view/episode_{episode_index:06d}.mp4"

    df = pd.read_parquet(parquet_path)
    states = np.stack(df["observation.state"].values)  # (N, 7)
    fps = info["fps"]

    return states, video_path, fps


def quaternion_to_forward(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion (xyzw) to a forward direction vector in the body frame.

    Body frame: +X forward. So we rotate [1, 0, 0] by the quaternion.
    """
    rot = Rotation.from_quat(quat_xyzw)  # scipy expects xyzw
    forward = rot.apply(np.array([1.0, 0.0, 0.0]))
    return forward


def render_3d_frame(
    fig,
    ax,
    positions: np.ndarray,
    quats: np.ndarray,
    current_idx: int,
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Render one frame of the 3D trajectory plot and return as BGR image."""
    ax.cla()

    # Full trajectory (faded)
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="steelblue",
        alpha=0.3,
        linewidth=1,
    )

    # Traversed trajectory
    end = current_idx + 1
    ax.plot(
        positions[:end, 0],
        positions[:end, 1],
        positions[:end, 2],
        color="steelblue",
        linewidth=2,
    )

    # Current position
    pos = positions[current_idx]
    ax.scatter(*pos, color="red", s=60, zorder=5)

    # Orientation arrow
    fwd = quaternion_to_forward(quats[current_idx])
    arrow_len = np.linalg.norm(positions[-1] - positions[0]) * 0.05
    arrow_len = max(arrow_len, 0.3)
    ax.quiver(
        pos[0], pos[1], pos[2],
        fwd[0], fwd[1], fwd[2],
        length=arrow_len,
        color="red",
        arrow_length_ratio=0.3,
        linewidth=2,
    )

    # Labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Body Pose Trajectory  [frame {current_idx}]", fontsize=10)

    # Equal aspect ratio
    all_ranges = np.ptp(positions, axis=0)
    max_range = all_ranges.max() / 2.0
    mid = (positions.max(axis=0) + positions.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]  # RGBA -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (canvas_w, canvas_h))
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize an episode trajectory")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the LeRobot dataset")
    parser.add_argument("--episode_index", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--output", type=str, default="visualize_episode.mp4", help="Output video path")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    states, video_path, fps = load_episode(dataset_dir, args.episode_index)
    positions = states[:, :3]   # (N, 3) translation
    quats = states[:, 3:]       # (N, 4) quaternion xyzw

    # Open source video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = len(states)

    # Canvas sizes — make left panel same height as video
    canvas_h = vid_h
    canvas_w = vid_h  # square for 3D plot

    # Set up matplotlib figure
    dpi = 100
    fig = plt.figure(figsize=(canvas_w / dpi, canvas_h / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.tight_layout()

    # Output video
    out_w = canvas_w + vid_w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, canvas_h))

    print(f"Rendering {n_frames} frames  |  episode {args.episode_index}  |  fps={fps}")
    for i in range(n_frames):
        ret, rgb_frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.resize(rgb_frame, (vid_w, canvas_h))

        plot_frame = render_3d_frame(fig, ax, positions, quats, i, canvas_w, canvas_h)
        combined = np.hstack([plot_frame, rgb_frame])
        writer.write(combined)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  frame {i + 1}/{n_frames}")

    cap.release()
    writer.release()
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
