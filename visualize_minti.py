"""Visualize a converted Minti LeRobot dataset as an animated GIF.

For a given episode, produces a side-by-side animation:
  - Left:  3D plot of body and camera coordinate frames moving along the
           trajectory (in the first-frame body coordinate system).
  - Right: the corresponding video.ego_view frame.

Usage:
    uv run visualize_minti.py \
        --dataset_root ./output/minti-yrj_scene_0002 \
        --episode_index 0 \
        --output ./vis_scene_0002_ep0.gif
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def _state_to_T(state: np.ndarray) -> np.ndarray:
    """[x, y, z, yaw_deg] -> 4x4 homogeneous transform (yaw about +z)."""
    x, y, z, yaw = state
    R = Rotation.from_euler("Z", yaw, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def _draw_frame(
    ax,
    T: np.ndarray,
    scale: float,
    colors=("r", "g", "b"),
    linewidth: float = 2.5,
    label_prefix: str = "",
    label_offset: float = 0.0,
):
    """Draw xyz axes for a 4x4 frame with customisable colours."""
    origin = T[:3, 3]
    x_axis = T[:3, 0] * scale
    y_axis = T[:3, 1] * scale
    z_axis = T[:3, 2] * scale
    ax.quiver(*origin, *x_axis, color=colors[0], linewidth=linewidth, arrow_length_ratio=0.2)
    ax.quiver(*origin, *y_axis, color=colors[1], linewidth=linewidth, arrow_length_ratio=0.2)
    ax.quiver(*origin, *z_axis, color=colors[2], linewidth=linewidth, arrow_length_ratio=0.2)
    if label_prefix:
        ax.text(origin[0], origin[1], origin[2] + label_offset, label_prefix,
                fontsize=10, fontweight="bold")


def _read_video_frames(video_path: Path, n_frames: int) -> list[np.ndarray]:
    reader = imageio.get_reader(str(video_path))
    frames = []
    for i, f in enumerate(reader):
        frames.append(np.asarray(f))
        if len(frames) >= n_frames:
            break
    reader.close()
    return frames


def visualize_episode(
    dataset_root: Path,
    episode_index: int,
    output_path: Path,
    stride: int = 1,
    fps: int = 10,
    frame_scale: float = 0.5,
):
    meta = LeRobotDatasetMetadata(dataset_root.name, root=dataset_root)

    # ---- read parquet for poses ----
    parquet_path = meta.root / meta.get_data_file_path(episode_index)
    df = pd.read_parquet(parquet_path)
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    N = len(states)

    # ---- read T_b_c from episodes_extras.jsonl ----
    extras_path = meta.root / "meta" / "episodes_extras.jsonl"
    T_b_c = np.eye(4)
    with open(extras_path, "r") as f:
        for line in f:
            item = json.loads(line)
            if item.get("episode_index") == episode_index:
                T_b_c = np.array(item["T_b_c"], dtype=np.float64)
                break

    # ---- read video frames ----
    video_key = "video.ego_view"
    video_path = meta.root / meta.get_video_file_path(episode_index, video_key)
    video_frames = _read_video_frames(video_path, N)
    if len(video_frames) < N:
        N = len(video_frames)
        states = states[:N]

    # ---- precompute per-frame body and camera transforms ----
    T_w_b_all = np.stack([_state_to_T(s) for s in states])
    T_w_c_all = np.einsum("nij,jk->nik", T_w_b_all, T_b_c)

    body_traj = T_w_b_all[:, :3, 3]
    cam_traj = T_w_c_all[:, :3, 3]
    # Local view window (meters) around the current body position
    view_half = max(2.0, frame_scale * 4.0)

    # ---- render frames ----
    indices = list(range(0, N, stride))
    out_frames = []

    # Bigger figure with left panel taking more space.
    fig: Figure = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.05)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_img = fig.add_subplot(gs[0, 1])

    body_colors = ("#e53935", "#43a047", "#1e88e5")   # bright RGB
    cam_colors = ("#ff7043", "#9ccc65", "#26c6da")    # softer / cyan-ish
    cam_scale = frame_scale * 0.6                      # smaller so it doesn't overlap body

    for k in indices:
        ax3d.clear()
        ax_img.clear()

        # Plot trajectory up to this frame
        ax3d.plot(body_traj[: k + 1, 0], body_traj[: k + 1, 1], body_traj[: k + 1, 2],
                  color="0.35", linewidth=1.2, label="body path")
        ax3d.plot(cam_traj[: k + 1, 0], cam_traj[: k + 1, 1], cam_traj[: k + 1, 2],
                  color="orange", linewidth=1.0, linestyle="--", label="camera path")

        # Draw body and camera frames at current pose with distinct colours/sizes
        _draw_frame(ax3d, T_w_b_all[k], scale=frame_scale,
                    colors=body_colors, linewidth=3.0,
                    label_prefix="body", label_offset=frame_scale * 0.15)
        _draw_frame(ax3d, T_w_c_all[k], scale=cam_scale,
                    colors=cam_colors, linewidth=2.0,
                    label_prefix="cam", label_offset=-cam_scale * 0.6)

        # Thin line connecting body origin to camera origin for clarity
        b0 = T_w_b_all[k, :3, 3]
        c0 = T_w_c_all[k, :3, 3]
        ax3d.plot([b0[0], c0[0]], [b0[1], c0[1]], [b0[2], c0[2]],
                  color="black", linewidth=0.8, alpha=0.5)

        # Center the view on current body position
        cx, cy, cz = T_w_b_all[k, :3, 3]
        ax3d.set_xlim(cx - view_half, cx + view_half)
        ax3d.set_ylim(cy - view_half, cy + view_half)
        ax3d.set_zlim(cz - view_half, cz + view_half)
        ax3d.set_xlabel("x (m, +forward)")
        ax3d.set_ylabel("y (m, +left)")
        ax3d.set_zlabel("z (m, +up)")
        ax3d.set_title(
            f"Frame {k}/{N - 1}    "
            f"body (bold RGB)  vs  camera (soft RGB, OpenCV: +x right / +y down / +z forward)",
            fontsize=10,
        )
        try:
            ax3d.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass
        ax3d.legend(loc="upper left", fontsize=9)

        ax_img.imshow(video_frames[k])
        ax_img.axis("off")
        ax_img.set_title("video.ego_view")

        fig.canvas.draw()
        # Use buffer_rgba for modern matplotlib compatibility
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        out_frames.append(buf)

    plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), out_frames, fps=fps)
    print(f"Saved: {output_path}  ({len(out_frames)} frames)")


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to converted dataset, e.g. ./output/minti-yrj_scene_0002")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--output", type=str, default="vis.gif")
    parser.add_argument("--stride", type=int, default=2,
                        help="Skip frames to keep gif size manageable")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame_scale", type=float, default=0.5,
                        help="Size of axis arrows in meters (also controls view window size)")
    args = parser.parse_args()

    visualize_episode(
        dataset_root=Path(args.dataset_root),
        episode_index=args.episode_index,
        output_path=Path(args.output),
        stride=args.stride,
        fps=args.fps,
        frame_scale=args.frame_scale,
    )


if __name__ == "__main__":
    main()
