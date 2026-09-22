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


HASH_CHUNK = 64 * 1024


@dataclass(frozen=True)
class MediaFile:
    path: Path       # path as found under the input folder (may be a symlink)
    original: Path   # resolved real path of the original file
    rel: str         # path relative to the input folder, used for display
    kind: str        # "image" or "video"
    key: str         # content key: the same file keeps its cache and labels after a move or rename

    def fingerprint(self) -> dict:
        return {"size": self.original.stat().st_size}


def content_key(path: Path) -> str:
    """Hash of the size plus the first and last 64 KB: cheap, and unique for camera files."""
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(HASH_CHUNK))
        if size > 2 * HASH_CHUNK:
            f.seek(-HASH_CHUNK, os.SEEK_END)
            h.update(f.read(HASH_CHUNK))
    return h.hexdigest()[:16]


def _keys(cfg: Config, originals: list[Path]) -> list[str]:
    """Content keys, remembered per path + size + mtime so unchanged files are not re-read."""
    index_path = cfg.cache_dir / "keys.json"
    try:
        index = json.loads(index_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        index = {}
    keys, changed = [], False
    for original in originals:
        st = original.stat()
        stamp = [st.st_size, int(st.st_mtime)]
        entry = index.get(str(original))
        if entry is None or entry[:2] != stamp:
            entry = [*stamp, content_key(original)]
            index[str(original)] = entry
            changed = True
        keys.append(entry[2])
    if changed:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index))
    return keys


def scan(cfg: Config) -> list[MediaFile]:
    root = cfg.input_dir
    found = []
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
            found.append((path, path.resolve(), str(path.relative_to(root)), kind))
    keys = _keys(cfg, [original for _, original, _, _ in found])
    return [MediaFile(*f, key) for f, key in zip(found, keys)]


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
