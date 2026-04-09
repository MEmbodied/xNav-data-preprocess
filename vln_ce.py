import logging
from argparse import ArgumentParser
from pathlib import Path

logging.basicConfig(level=logging.INFO)

import time
import json


from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from utils.lerobot.lerobot_creater import LeRobotCreator
from utils import Trajectories
from utils.vln_ce.trajectory import VLN_CE_Trajectories

parser = ArgumentParser(description="Port VLN-CE dataset to LeRobotDataset format")
parser.add_argument("--raw_dir", type=str, default="InternData-n1-demo", help="Path to the raw VLN-CE dataset directory")
parser.add_argument("--output_dir", type=str, default=".", help="Path to the output LeRobotDataset directory")
parser.add_argument("--codec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"], help="Video codec to use for encoding")
parser.add_argument("--num_processes", type=int, default=8, help="Number of processes for image writing")
parser.add_argument(
    "--viewpoint",
    type=str,
    default="125cm_0deg",
    choices=["125cm_0deg", "125cm_30deg", "125cm_45deg", "60cm_15deg", "60cm_30deg"],
    help="Viewpoint to use for poses, RGB frames, and relative-goal annotations",
)
args = parser.parse_args()


def port(
    raw_dir: str,
    repo_id: str,
    root: str,
    traj_cls: type[Trajectories],
    num_processes: int,
    codec: str = "h264",
    *args, **kwargs
):
    """Port raw dataset to LeRobotDataset format."""
    logging.info(f"Porting raw dataset from {raw_dir} to LeRobotDataset repo {repo_id}")
    
    # VLN_CE_Trajectories.FEATURES defines the schema
    features = traj_cls.FEATURES

    logging.info(f"Creating LeRobotCreator at {root}")
    creator = LeRobotCreator(
        root=str(root),
        robot_type=traj_cls.ROBOT_TYPE,
        fps=traj_cls.FPS,
        features=features,
        num_workers=max(1, num_processes),
        num_video_encoders=int(max(1, num_processes)  * 1.75), # more encoders than workers
        codec=codec,
        has_extras=True,
    )
    
    def get_task_idx(task: str) -> int:
        return creator.add_task(task)

    trajectories = traj_cls(raw_dir, get_task_idx=get_task_idx, *args, **kwargs)

    start_time = time.time()
    num_episodes = len(trajectories)
    logging.info(f"Number of episodes {num_episodes}")

    for episode_index, episode in enumerate(trajectories):
        creator.submit_episode(episode)
        
        elapsed_time = time.time() - start_time
        if (episode_index + 1) % 10 == 0:
            logging.info('\033[92m' + f"Submitted {episode_index + 1} / {num_episodes} episodes (elapsed {elapsed_time:.3f} s)" + '\033[0m')

    logging.info("All episodes submitted. Waiting for completion...")
    creator.wait()
    logging.info("LeRobotCreator finished.")
    

def validate_dataset(repo_id, root: str):
    """Sanity check that ensure meta data can be loaded and all files are present."""
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


def main():
    raw_dir = Path(args.raw_dir)
    folder_name = raw_dir.name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a distinct folder name for CE dataset
    dataset_name = f"VLN-CE-{folder_name}"
    root = output_dir / dataset_name
    
    port(
        raw_dir=raw_dir,
        repo_id=dataset_name,
        root=root,
        traj_cls=VLN_CE_Trajectories,
        num_processes=args.num_processes,
        codec=args.codec,
        viewpoint=args.viewpoint,
    )
    
    validate_dataset(dataset_name, root=root)

if __name__ == "__main__":
    main()
