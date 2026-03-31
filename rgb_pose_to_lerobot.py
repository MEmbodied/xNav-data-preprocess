import logging
from argparse import ArgumentParser
from pathlib import Path

from utils.rgb_pose_dataset import export_rgb_pose_source, load_source_cls


def parse_args():
    parser = ArgumentParser(description="Export RGB plus camera pose trajectories to LeRobot format.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to the raw source directory.")
    parser.add_argument("--source_cls", type=str, required=True, help="Import path such as package.module:ClassName.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory used to store the exported dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="LeRobot dataset directory name. Defaults to raw_dir name.")
    parser.add_argument(
        "--codec",
        type=str,
        default="h264",
        choices=["h264", "hevc", "libsvtav1"],
        help="Video codec to use for encoding.",
    )
    parser.add_argument(
        "--pix_fmt",
        type=str,
        default="auto",
        choices=["auto", "yuv420p", "yuv444p"],
        help="Pixel format used for video encoding. auto keeps odd resolutions encodable.",
    )
    parser.add_argument("--num_processes", type=int, default=8, help="Number of worker processes.")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name or raw_dir.name
    root = output_dir / dataset_name

    source_cls = load_source_cls(args.source_cls)
    export_rgb_pose_source(
        raw_dir=raw_dir,
        repo_id=dataset_name,
        root=root,
        source_cls=source_cls,
        num_processes=args.num_processes,
        codec=args.codec,
        pix_fmt=args.pix_fmt, 
    )


if __name__ == "__main__":
    main()
