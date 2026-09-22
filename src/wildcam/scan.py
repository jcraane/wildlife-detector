"""Finding media files and reading their metadata. Everything here is read-only."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from .config import Config

# e.g. IMG_20260810_171738.jpg, 2026-08-10 17.17.38.mp4, VID20260810171738.mp4
FILENAME_TIME_RE = re.compile(
    r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[-_ T.]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})")


@dataclass(frozen=True)
class MediaFile:
    path: Path       # path as found under the input folder (may be a symlink)
    original: Path   # resolved real path of the original file
    rel: str         # path relative to the input folder, used for display
    kind: str        # "image" or "video"

    @property
    def key(self) -> str:
        """Stable cache key: independent of which folder (sample or full) it was found in."""
        return hashlib.sha1(str(self.original).encode()).hexdigest()[:16]

    def fingerprint(self) -> dict:
        st = self.original.stat()
        return {"size": st.st_size, "mtime": int(st.st_mtime)}


def scan(cfg: Config) -> list[MediaFile]:
    root = cfg.input_dir
    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):  # .DS_Store, ._AppleDouble files
                continue
            ext = os.path.splitext(name)[1].lower()
            kind = "image" if ext in cfg.image_exts else "video" if ext in cfg.video_exts else None
            if kind is None:
                continue
            path = Path(dirpath) / name
            files.append(MediaFile(path, path.resolve(), str(path.relative_to(root)), kind))
    return files


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration:format_tags=creation_time",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _exif_time(path: Path) -> datetime | None:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            value = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal, DateTime
        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S") if value else None
    except Exception:
        return None


def _filename_time(name: str) -> datetime | None:
    m = FILENAME_TIME_RE.search(name)
    if not m:
        return None
    try:
        return datetime(*map(int, m.groups()))
    except ValueError:
        return None


def read_metadata(media: MediaFile) -> dict:
    """Capture time (camera local time), dimensions and duration."""
    meta: dict = {"capture_time": None, "capture_time_source": None,
                  "width": None, "height": None, "duration_s": None}
    when, source = None, None
    if media.kind == "image":
        with Image.open(media.original) as im:
            meta["width"], meta["height"] = im.size
        when, source = _exif_time(media.original), "exif"
    else:
        info = ffprobe(media.original)
        stream = (info.get("streams") or [{}])[0]
        meta["width"], meta["height"] = stream.get("width"), stream.get("height")
        fmt = info.get("format", {})
        if fmt.get("duration"):
            meta["duration_s"] = round(float(fmt["duration"]), 2)
        created = fmt.get("tags", {}).get("creation_time")
        if created:
            # Trail cameras have no time zone setting and write local time with a
            # "Z" suffix (verified against the EXIF time of the paired photo), so
            # the value is kept as naive local time.
            when = datetime.fromisoformat(created.replace("Z", "")).replace(tzinfo=None, microsecond=0)
            source = "video_metadata"
    if when is None:
        when, source = _filename_time(media.original.name), "filename"
    if when is None:
        when = datetime.fromtimestamp(media.original.stat().st_mtime).replace(microsecond=0)
        source = "file_mtime"
    meta["capture_time"] = when.isoformat(sep=" ")
    meta["capture_time_source"] = source
    return meta
