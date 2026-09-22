"""Loading pixels: photos via Pillow, video frames sampled with ffmpeg."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from megadetector.visualization.visualization_utils import load_image

from .scan import MediaFile


def sample_video(path: Path, width: int, height: int, fps: float, out_width: int) -> list[tuple[float, Image.Image]]:
    """Returns (timestamp_s, frame) pairs sampled at `fps` frames per second."""
    out_w = min(out_width, width)
    out_h = int(round(height * out_w / width / 2)) * 2
    frame_bytes = out_w * out_h * 3

    def run(hwaccel: bool) -> subprocess.CompletedProcess:
        cmd = ["ffmpeg", "-nostdin", "-v", "error"]
        if hwaccel:
            cmd += ["-hwaccel", "videotoolbox"]
        cmd += ["-i", str(path), "-an", "-vf", f"fps={fps},scale={out_w}:{out_h}",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        return subprocess.run(cmd, capture_output=True)

    proc = run(hwaccel=sys.platform == "darwin")
    if proc.returncode != 0 or not proc.stdout:
        proc = run(hwaccel=False)
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()[:300]}")
    data = proc.stdout
    frames = []
    for i in range(len(data) // frame_bytes):
        arr = np.frombuffer(data, np.uint8, frame_bytes, i * frame_bytes).reshape(out_h, out_w, 3)
        frames.append((round(i / fps, 2), Image.fromarray(arr)))
    return frames


def load_frames(media: MediaFile, meta: dict, fps: float, frame_width: int) -> list[tuple[float | None, Image.Image]]:
    if media.kind == "image":
        return [(None, load_image(str(media.original)))]  # applies EXIF rotation
    return sample_video(media.original, meta["width"], meta["height"], fps, frame_width)
