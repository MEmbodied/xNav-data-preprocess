import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


matplotlib.use("Agg")
from matplotlib import pyplot as plt


AXIS_COLORS = {
    "x": "#d62728",
    "y": "#2ca02c",
    "z": "#ffd700",
}


def parse_args():
    parser = ArgumentParser(description="Visualize a converted VLN-CE episode as a synchronized GIF.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the converted LeRobot dataset root.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument("--output-gif", type=str, required=True, help="Output GIF path.")
    parser.add_argument("--axis-length", type=float, default=0.35, help="Axis length for body and camera frames.")
    parser.add_argument("--fps", type=float, default=None, help="Override GIF fps. Defaults to dataset fps.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quicker previews.")
    return parser.parse_args()


def pose7d_to_matrix(pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix().astype(np.float32)
    transform[:3, 3] = pose[:3]
    return transform


def load_episode_extras(extras_path: Path, episode_index: int) -> dict:
    with open(extras_path, "r") as f:
        for line in f:
            item = json.loads(line)
            if item["episode_index"] == episode_index:
                return item
    raise ValueError(f"Episode {episode_index} not found in {extras_path}")


def read_video_frames(video_path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return frames


def compute_plot_limits(body_positions: np.ndarray, camera_positions: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    all_points = np.concatenate([body_positions, camera_positions], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = np.max(maxs - mins)
    radius = max(0.5, span / 2.0 + 0.25)
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (center[2] - radius, center[2] + radius),
    )


def draw_frame_axes(ax, transform: np.ndarray, axis_length: float, colors: dict[str, str], label_prefix: str, linestyle: str, linewidth: float):
    origin = transform[:3, 3]
    rotation = transform[:3, :3]
    axes = {
        "x": rotation[:, 0],
        "y": rotation[:, 1],
        "z": rotation[:, 2],
    }
    for axis_name, direction in axes.items():
        end = origin + axis_length * direction
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=colors[axis_name],
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"{label_prefix} {axis_name.upper()}",
        )


def configure_axes(ax, x_lim, y_lim, z_lim, title: str):
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_xlabel("X forward")
    ax.set_ylabel("Y left")
    ax.set_zlabel("Z up")
    ax.view_init(elev=28, azim=-62)
    ax.set_title(title)


def render_gif(
    states: np.ndarray,
    frames_rgb: list[np.ndarray],
    body_from_camera: np.ndarray,
    output_gif: Path,
    fps: float,
    axis_length: float,
):
    num_frames = min(len(states), len(frames_rgb))
    states = states[:num_frames]
    frames_rgb = frames_rgb[:num_frames]

    body_transforms = np.stack([pose7d_to_matrix(pose) for pose in states], axis=0)
    camera_transforms = np.stack([t_body @ body_from_camera for t_body in body_transforms], axis=0)
    body_positions = body_transforms[:, :3, 3]
    camera_positions = camera_transforms[:, :3, 3]

    x_lim, y_lim, z_lim = compute_plot_limits(body_positions, camera_positions)
    gif_frames = []

    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0])
        ax_body = fig.add_subplot(gs[0, 0], projection="3d")
        ax_camera = fig.add_subplot(gs[1, 0], projection="3d")
        ax_img = fig.add_subplot(gs[:, 1])

        ax_body.plot(
            body_positions[:, 0],
            body_positions[:, 1],
            body_positions[:, 2],
            color="black",
            linewidth=1.5,
            alpha=0.35,
            label="body path",
        )
        ax_body.plot(
            body_positions[: frame_idx + 1, 0],
            body_positions[: frame_idx + 1, 1],
            body_positions[: frame_idx + 1, 2],
            color="black",
            linewidth=2.5,
        )
        draw_frame_axes(
            ax_body,
            body_transforms[frame_idx],
            axis_length=axis_length,
            colors=AXIS_COLORS,
            label_prefix="body",
            linestyle="-",
            linewidth=2.4,
        )
        ax_body.scatter(
            [body_positions[frame_idx, 0]],
            [body_positions[frame_idx, 1]],
            [body_positions[frame_idx, 2]],
            color="black",
            s=40,
        )
        configure_axes(ax_body, x_lim, y_lim, z_lim, "Body Trajectory + Body Frame")
        ax_body.legend(loc="upper left", fontsize=8)

        ax_camera.plot(
            camera_positions[:, 0],
            camera_positions[:, 1],
            camera_positions[:, 2],
            color="#444444",
            linewidth=1.5,
            alpha=0.35,
            label="camera path",
        )
        ax_camera.plot(
            camera_positions[: frame_idx + 1, 0],
            camera_positions[: frame_idx + 1, 1],
            camera_positions[: frame_idx + 1, 2],
            color="#444444",
            linewidth=2.5,
        )
        draw_frame_axes(
            ax_camera,
            camera_transforms[frame_idx],
            axis_length=axis_length * 0.9,
            colors=AXIS_COLORS,
            label_prefix="camera",
            linestyle="-",
            linewidth=2.4,
        )
        ax_camera.scatter(
            [camera_positions[frame_idx, 0]],
            [camera_positions[frame_idx, 1]],
            [camera_positions[frame_idx, 2]],
            color="#444444",
            s=36,
        )
        configure_axes(ax_camera, x_lim, y_lim, z_lim, "Camera Trajectory + Camera Frame")
        ax_camera.legend(loc="upper left", fontsize=8)

        ax_img.imshow(frames_rgb[frame_idx])
        ax_img.axis("off")
        ax_img.set_title(f"video.front frame {frame_idx}")

        fig.suptitle(
            "Left: body and camera shown separately | Right: synchronized RGB",
            fontsize=12,
        )
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
        gif_frames.append(image)
        plt.close(fig)

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_gif, gif_frames, duration=1.0 / fps, loop=0)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_gif = Path(args.output_gif)

    repo_id = dataset_root.name
    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    data_path = dataset_root / meta.get_data_file_path(args.episode_index)
    video_path = dataset_root / meta.get_video_file_path(args.episode_index, "video.front")
    extras_path = dataset_root / "meta" / "episodes_extras.jsonl"

    df = pd.read_parquet(data_path)
    extras = load_episode_extras(extras_path, args.episode_index)
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    body_from_camera = np.array(extras["video.front.body_from_camera"], dtype=np.float32)
    frames_rgb = read_video_frames(video_path, max_frames=args.max_frames)
    fps = args.fps or meta.fps

    render_gif(
        states=states,
        frames_rgb=frames_rgb,
        body_from_camera=body_from_camera,
        output_gif=output_gif,
        fps=float(fps),
        axis_length=args.axis_length,
    )
    print(f"Saved GIF to {output_gif}")


if __name__ == "__main__":
    main()
