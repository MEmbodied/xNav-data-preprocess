#!/usr/bin/env python

import glob
import importlib
import logging
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import av
import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
from PIL import Image

def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        # logging.warning(
        #     "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        # )
        return "pyav"

def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")

def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    video_path = str(video_path)

    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True

    reader = torchvision.io.VideoReader(video_path, "video")
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    reader.seek(first_ts, keyframes_only=keyframes_only)

    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"Timestamp deviation > {tolerance_s=}. "
        f"\nqueried: {query_ts}"
        f"\nloaded: {loaded_ts}"
        f"\nvideo: {video_path}"
    )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_frames = closest_frames.type(torch.float32) / 255

    return closest_frames

def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    device: str = "cpu",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    if importlib.util.find_spec("torchcodec"):
        from torchcodec.decoders import VideoDecoder
    else:
        raise ImportError("torchcodec is required but not available.")

    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    loaded_frames = []
    loaded_ts = []
    metadata = decoder.metadata
    average_fps = metadata.average_fps

    frame_indices = [round(ts * average_fps) for ts in timestamps]
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all()

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_frames = closest_frames.type(torch.float32) / 255
    return closest_frames

def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}.")

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)
    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    if vcodec == "libsvtav1" and pix_fmt == "yuv444p":
        pix_fmt = "yuv420p"

    template = "frame_" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    with Image.open(input_list[0]) as dummy_image:
        width, height = dummy_image.size

    video_options = {}
    if g is not None: video_options["g"] = str(g)
    if crf is not None: video_options["crf"] = str(crf)
    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)

    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        for input_data in input_list:
            with Image.open(input_data) as input_image:
                input_frame = av.VideoFrame.from_image(input_image.convert("RGB"))
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")

@dataclass
class VideoFrame:
    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)
    def __call__(self):
        return self.pa_type

# Suppress "Overwriting feature type" log from datasets
_datasets_log = logging.getLogger("datasets")
_prev_level = _datasets_log.level
_datasets_log.setLevel(logging.ERROR)
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        register_feature(VideoFrame, "VideoFrame")
finally:
    # Restore original logging level
    _datasets_log.setLevel(_prev_level)

def get_video_info(video_path: Path | str) -> dict:
    logging.getLogger("libav").setLevel(av.logging.ERROR)
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
            video_info["video.height"] = video_stream.height
            video_info["video.width"] = video_stream.width
            video_info["video.codec"] = video_stream.codec.canonical_name
            video_info["video.pix_fmt"] = video_stream.pix_fmt
            video_info["video.is_depth_map"] = False
            video_info["video.fps"] = int(video_stream.base_rate)
            video_info["video.channels"] = 3 # simplified
        except IndexError:
            pass
    av.logging.restore_default_callback()
    return video_info
