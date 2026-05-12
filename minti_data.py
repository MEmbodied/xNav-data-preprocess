"""Port Minti scene data to LeRobotDataset v2.1 format.

Usage:
    uv run minti_data.py --raw_dir /data-25T/DataEngine/yrj_scene_0002 \
        --output_dir ./output --num_processes 8
"""
import logging
import time
from argparse import ArgumentParser
from pathlib import Path

logging.basicConfig(level=logging.INFO)

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from utils import Trajectories
from utils.lerobot.lerobot_creater import LeRobotCreator
from utils.minti.trajectory import MintiTrajectories


parser = ArgumentParser(description="Port Minti dataset to LeRobotDataset format")
parser.add_argument("--raw_dir", type=str, required=True, help="Path to a single Minti scene directory")
parser.add_argument("--output_dir", type=str, default=".", help="Output root directory")
parser.add_argument("--codec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"], help="Video codec")
parser.add_argument("--num_processes", type=int, default=8, help="Number of worker processes")
args = parser.parse_args()


def port(
    raw_dir: str,
    repo_id: str,
    root: str,
    traj_cls: type[Trajectories],
    num_processes: int,
    codec: str = "h264",
):
    logging.info(f"Porting raw dataset from {raw_dir} to LeRobotDataset repo {repo_id}")

    features = traj_cls.FEATURES

    creator = LeRobotCreator(
        root=str(root),
        robot_type=traj_cls.ROBOT_TYPE,
        fps=traj_cls.FPS,
        features=features,
        num_workers=max(1, num_processes),
        num_video_encoders=int(max(1, num_processes) * 1.75),
        codec=codec,
        has_extras=True,
    )

    def get_task_idx(task: str) -> int:
        return creator.add_task(task)

    trajectories = traj_cls(raw_dir, get_task_idx=get_task_idx)

    start_time = time.time()
    num_episodes = len(trajectories)
    logging.info(f"Number of episodes {num_episodes}")

    for episode_index, episode in enumerate(trajectories):
        creator.submit_episode(episode)
        elapsed_time = time.time() - start_time
        if (episode_index + 1) % 10 == 0:
            logging.info(
                "\033[92m"
                + f"Submitted {episode_index + 1} / {num_episodes} episodes "
                + f"(elapsed {elapsed_time:.3f} s)"
                + "\033[0m"
            )

    logging.info("All episodes submitted. Waiting for completion...")
    creator.wait()
    logging.info("LeRobotCreator finished.")


def validate_dataset(repo_id: str, root: str):
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    if meta.total_episodes == 0:
        raise ValueError("Number of episodes is 0.")
    for ep_idx in range(meta.total_episodes):
        data_path = meta.root / meta.get_data_file_path(ep_idx)
        if not data_path.exists():
            raise ValueError(f"Parquet file is missing in: {data_path}")
        for vid_key in meta.video_keys:
            vid_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
            if not vid_path.exists():
                raise ValueError(f"Video file is missing in: {vid_path}")
    logging.info("Validation passed.")


def main():
    raw_dir = Path(args.raw_dir)
    folder_name = raw_dir.name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = f"minti-{folder_name}"
    root = output_dir / dataset_name

    port(
        raw_dir=str(raw_dir),
        repo_id=dataset_name,
        root=root,
        traj_cls=MintiTrajectories,
        num_processes=args.num_processes,
        codec=args.codec,
    )

    validate_dataset(dataset_name, root=root)


if __name__ == "__main__":
    main()
