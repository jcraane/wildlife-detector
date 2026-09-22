"""Running MegaDetector over every photo and sampled video frame, with a per-file cache."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .config import Config
from .frames import load_frames
from .scan import MediaFile, read_metadata

CACHE_VERSION = 1
CATEGORY_NAMES = {"1": "animal", "2": "person", "3": "vehicle"}


def cache_path(cfg: Config, media: MediaFile) -> Path:
    return cfg.detections_dir / f"{media.key}.json"


def best_frame_path(cfg: Config, media: MediaFile) -> Path:
    return cfg.detections_dir / "frames" / f"{media.key}.jpg"


def read_cache(cfg: Config, media: MediaFile) -> dict | None:
    try:
        return json.loads(cache_path(cfg, media).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _cache_valid(cfg: Config, media: MediaFile, rec: dict | None) -> bool:
    return (rec is not None
            and rec.get("version") == CACHE_VERSION
            and rec.get("fingerprint") == media.fingerprint()
            and rec.get("frame_width") == cfg.frame_width
            and rec.get("min_cached_conf", 1.0) <= cfg.min_cached_conf)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _load(cfg: Config, media: MediaFile):
    """Runs in a worker thread: metadata + pixels, so the GPU never waits on ffmpeg."""
    try:
        meta = read_metadata(media)
        return meta, load_frames(media, meta, cfg.fps, cfg.frame_width), None
    except Exception as e:  # corrupt or truncated file
        return None, [], f"{type(e).__name__}: {e}"


def run_detection(cfg: Config, files: list[MediaFile], redetect: bool = False) -> None:
    todo = [m for m in files if redetect or not _cache_valid(cfg, m, read_cache(cfg, m))]
    cached = len(files) - len(todo)
    n_img = sum(m.kind == "image" for m in todo)
    print(f"{len(files)} files: {cached} already cached, {len(todo)} to process "
          f"({n_img} photos, {len(todo) - n_img} videos)")
    if not todo:
        return

    from megadetector.detection import run_detector
    detector = run_detector.load_detector(cfg.model)
    print(f"Model {cfg.model} on device: {getattr(detector, 'device', 'unknown')}")

    n_frames = n_hits = 0
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=cfg.workers)
    pending: deque = deque()
    queue = iter(todo)

    def refill():
        while len(pending) < cfg.workers * 2:
            media = next(queue, None)
            if media is None:
                return
            pending.append((media, pool.submit(_load, cfg, media)))

    try:
        refill()
        with tqdm(total=len(todo), unit="file", dynamic_ncols=True) as bar:
            while pending:
                media, future = pending.popleft()
                refill()
                meta, frames, error = future.result()
                record = {
                    "version": CACHE_VERSION,
                    "original": str(media.original),
                    "kind": media.kind,
                    "fingerprint": media.fingerprint(),
                    "model": cfg.model,
                    "fps": cfg.fps,
                    "frame_width": cfg.frame_width,
                    "min_cached_conf": cfg.min_cached_conf,
                    "meta": meta,
                    "frames": [],
                    "best_frame": None,
                    "error": error,
                    "processed_at": datetime.now().isoformat(timespec="seconds"),
                }
                if frames:
                    results = detector.generate_detections_one_batch(
                        [img for _, img in frames], image_id=[str(i) for i in range(len(frames))],
                        detection_threshold=cfg.min_cached_conf)
                    best = (-1.0, None)
                    hit = False
                    for (t, img), res in zip(frames, results):
                        dets = [{"category": CATEGORY_NAMES.get(d["category"], d["category"]),
                                 "conf": round(float(d["conf"]), 4),
                                 "bbox": [round(float(v), 4) for v in d["bbox"]]}
                                for d in res.get("detections") or []]
                        record["frames"].append({"t": t, "detections": dets,
                                                 "failure": res.get("failure")})
                        hit = hit or any(d["conf"] >= cfg.threshold_for(d["category"]) for d in dets)
                        top = max((d["conf"] for d in dets), default=-1.0)
                        if top > best[0]:
                            best = (top, img)
                    if media.kind == "video" and best[1] is not None:
                        out = best_frame_path(cfg, media)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        best[1].save(out, quality=90)
                        record["best_frame"] = out.name
                    n_frames += len(frames)
                    n_hits += hit
                _write_json(cache_path(cfg, media), record)
                elapsed = time.monotonic() - started
                bar.set_postfix(frames_per_s=f"{n_frames / elapsed:.1f}", hits=n_hits, refresh=False)
                bar.update()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
