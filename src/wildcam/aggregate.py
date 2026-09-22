"""Turning cached detections into per-file results at a given threshold. No model needed."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .config import Config
from .detect import read_cache
from .labels import NO_ANIMAL
from .scan import MediaFile

CATEGORIES = ("animal", "person", "vehicle")
SORTED_FOLDERS = CATEGORIES + ("empty", "error", "not_processed")

CSV_COLUMNS = [
    "path", "original_path", "type", "category", "categories", "max_conf",
    "animal_conf", "person_conf", "vehicle_conf", "best_frame_time_s",
    "hit_frames", "sampled_frames", "capture_time", "capture_time_source",
    "duration_s", "width", "height", "species", "species_score", "species_source", "species_all", "error",
]


def summarize(media: MediaFile, rec: dict | None, cfg: Config, species: dict) -> dict:
    row = {c: None for c in CSV_COLUMNS}
    row.update(path=media.rel, original_path=str(media.original), type=media.kind)
    if rec is None:
        row["category"] = "not_processed"
        return row
    row.update(rec.get("meta") or {})
    row["error"] = rec.get("error")
    frames = rec.get("frames") or []
    row["sampled_frames"] = len(frames)
    if row["error"] or not frames:
        row["category"] = "error"
        return row

    def qualifies(d: dict) -> bool:
        return d["conf"] >= cfg.threshold_for(d["category"])

    # Best frame: the strongest qualifying detection, or the strongest overall
    # for empty files (so max_conf still shows how close they came).
    best_conf, best_frame, best_qualifies = -1.0, frames[0], False
    per_cat = dict.fromkeys(CATEGORIES, 0.0)
    hit_frames = 0
    for frame in frames:
        dets = frame["detections"]
        if any(qualifies(d) for d in dets):
            hit_frames += 1
        for d in dets:
            per_cat[d["category"]] = max(per_cat.get(d["category"], 0.0), d["conf"])
            ok = qualifies(d)
            if (ok, d["conf"]) > (best_qualifies, best_conf):
                best_conf, best_frame, best_qualifies = d["conf"], frame, ok

    hit_cats = sorted((c for c in per_cat if per_cat[c] >= cfg.threshold_for(c)), key=lambda c: -per_cat[c])
    sp = species.get(media.key)
    if sp and "animal" in hit_cats and sp["t"] == best_frame["t"]:
        row["species"], row["species_score"] = sp["label"], sp["score"]
        row["species_source"] = sp.get("source")
        row["species_all"] = ";".join(dict.fromkeys(b["label"] for b in sp["boxes"] if b["label"] != NO_ANIMAL))
        if sp["label"] == NO_ANIMAL:
            hit_cats.remove("animal")
    row.update(
        category=hit_cats[0] if hit_cats else "empty",
        categories=";".join(hit_cats),
        max_conf=round(max(best_conf, 0.0), 4),
        animal_conf=per_cat["animal"], person_conf=per_cat["person"], vehicle_conf=per_cat["vehicle"],
        best_frame_time_s=best_frame["t"] if media.kind == "video" else None,
        hit_frames=hit_frames,
    )
    # Kept for the report, not written to the CSV.
    row["_key"] = media.key
    row["_best_frame_file"] = rec.get("best_frame")
    row["_boxes"] = [d for d in best_frame["detections"] if qualifies(d)]
    return row


def load_species(cfg: Config) -> dict:
    path = cfg.species_dir / "by_key.json"
    return json.loads(path.read_text()) if path.exists() else {}


def aggregate(cfg: Config, files: list[MediaFile], with_species: bool = True) -> list[dict]:
    species = load_species(cfg) if with_species else {}
    rows = [summarize(m, read_cache(cfg, m), cfg, species) for m in files]
    rows.sort(key=lambda r: (r["category"] in ("empty", "error", "not_processed"), -(r["max_conf"] or 0)))
    return rows


def write_tables(cfg: Config, rows: list[dict]) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    payload = {"threshold": cfg.threshold, "category_thresholds": cfg.category_thresholds, "model": cfg.model, "fps": cfg.fps, "files": public}
    (cfg.output_dir / "results.json").write_text(json.dumps(payload, indent=1))


def _link_name(row: dict) -> str:
    stamp = (row["capture_time"] or "").replace("-", "").replace(":", "").replace(" ", "_")
    flat = row["path"].replace(os.sep, "_")
    return f"{stamp}_{flat}" if stamp else flat


def write_sorted_view(cfg: Config, rows: list[dict]) -> None:
    """Rebuilds output/sorted/<category>/ with symlinks to the originals.

    Only symlinks inside our own output folder are removed; the originals are
    never touched. A file with several categories is linked in each of them.
    """
    root = cfg.output_dir / "sorted"
    if root.exists():
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            for name in filenames + dirnames:
                p = Path(dirpath) / name
                if p.is_symlink():
                    p.unlink()
                elif p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
    for folder in SORTED_FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)
    for row in rows:
        folders = row["categories"].split(";") if row["categories"] else [row["category"]]
        for folder in folders:
            os.symlink(row["original_path"], root / folder / _link_name(row))
