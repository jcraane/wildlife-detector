"""Optional species step: SpeciesNet on the best frame of every animal hit.

SpeciesNet lives in its own uv project (./species) because its onnx/protobuf
requirements conflict with the `megadetector` package. It runs its own
detector + classifier + geofenced ensemble on each image, and resumes from
its predictions file, so re-runs only classify new hits.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tqdm import tqdm

from .config import PROJECT_DIR, Config


def _full_res_frame(cfg: Config, row: dict) -> Path:
    """Re-extracts the best video frame at full resolution for the classifier."""
    out = cfg.species_dir / "frames" / f"{row['_key']}.jpg"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(row["best_frame_time_s"] or 0),
             "-i", row["original_path"], "-frames:v", "1", "-q:v", "2", str(out)],
            check=True)
    return out


def run_species(cfg: Config, rows: list[dict]) -> None:
    animals = [r for r in rows if "animal" in (r["categories"] or "").split(";")]
    print(f"SpeciesNet on {len(animals)} animal hits (country: {cfg.country or 'none'})")
    if not animals:
        return
    cfg.species_dir.mkdir(parents=True, exist_ok=True)
    key_by_path = {}
    instances = []
    for r in tqdm(animals, desc="Preparing frames", unit="file"):
        path = str(_full_res_frame(cfg, r)) if r["type"] == "video" else r["original_path"]
        key_by_path[path] = r["_key"]
        inst = {"filepath": path}
        if cfg.country:
            inst["country"] = cfg.country
        instances.append(inst)
    instances_json = cfg.species_dir / "instances.json"
    instances_json.write_text(json.dumps({"instances": instances}, indent=1))
    predictions_json = cfg.species_dir / "predictions.json"
    subprocess.run(
        ["uv", "run", "--project", str(PROJECT_DIR / "species"), "python", "-m",
         "speciesnet.scripts.run_model", "--instances_json", str(instances_json),
         "--predictions_json", str(predictions_json), "--bypass_prompts"],
        check=True)

    by_key_path = cfg.species_dir / "by_key.json"
    by_key = json.loads(by_key_path.read_text()) if by_key_path.exists() else {}
    for pred in json.loads(predictions_json.read_text()).get("predictions", []):
        key = key_by_path.get(pred.get("filepath"))
        if key is None or "prediction" not in pred:
            continue
        taxonomy = pred["prediction"].split(";")
        by_key[key] = {
            "label": taxonomy[-1] or next((t for t in reversed(taxonomy[1:]) if t), "unknown"),
            "score": round(float(pred.get("prediction_score", 0)), 4),
            "taxonomy": ";".join(taxonomy[1:-1]),
            "source": pred.get("prediction_source"),
        }
    by_key_path.write_text(json.dumps(by_key, indent=1))
