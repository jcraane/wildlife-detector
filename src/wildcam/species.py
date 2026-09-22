"""Optional species step: zero-shot BioCLIP-2 on the animal boxes of every animal hit.

Each MegaDetector animal box on a hit's best frame is cropped from the
full-resolution image and compared against the labels in config.toml. Image
embeddings are cached per crop, so editing the label list only re-scores.

With hand-made labels from the report (labels.csv), a logistic regression is
trained on the embeddings of the labelled crops and replaces zero-shot for the
labelled classes. A manual label always wins for its own file.
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm
from megadetector.visualization.visualization_utils import load_image

from .config import Config
from .labels import load_labels

CROP_PAD = 0.1   # extra context around each box, as a fraction of its size
BATCH_SIZE = 32


def _device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _crop_id(row: dict, bbox: list[float]) -> str:
    return f"{row['_key']}|{row['best_frame_time_s']}|{','.join(f'{v:.4f}' for v in bbox)}"


def _best_frame(row: dict) -> Image.Image:
    """Full-resolution best frame: the photo itself, or the video frame at best_frame_time_s."""
    if row["type"] == "image":
        return load_image(row["original_path"])
    png = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(row["best_frame_time_s"] or 0),
         "-i", row["original_path"], "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(png)).convert("RGB")


def _crops(row: dict, boxes: list[dict]) -> list[Image.Image]:
    img = _best_frame(row)
    w_img, h_img = img.size
    out = []
    for b in boxes:
        x, y, w, h = b["bbox"]
        out.append(img.crop((int(max(0, (x - w * CROP_PAD) * w_img)), int(max(0, (y - h * CROP_PAD) * h_img)),
                             int(min(w_img, (x + w * (1 + CROP_PAD)) * w_img)),
                             int(min(h_img, (y + h * (1 + CROP_PAD)) * h_img)))))
    return out


def _encode(v: np.ndarray) -> str:
    return base64.b64encode(v.astype(np.float16).tobytes()).decode()


def _decode(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32)


def _animal_boxes(row: dict) -> list[dict]:
    return sorted((b for b in row["_boxes"] if b["category"] == "animal"), key=lambda b: -b["conf"])


def run_species(cfg: Config, rows: list[dict]) -> None:
    if not cfg.species_labels:
        raise SystemExit("No [species.labels] in config.toml")
    animals = [r for r in rows if "animal" in (r["categories"] or "").split(";")]
    print(f"Species ({cfg.species_model}) on {len(animals)} animal hits, {len(cfg.species_labels)} labels")
    if not animals:
        return
    emb_path = cfg.species_dir / "embeddings.json"
    embeddings: dict[str, str] = json.loads(emb_path.read_text()) if emb_path.exists() else {}
    jobs = [(r, missing) for r in animals
            if (missing := [b for b in _animal_boxes(r) if _crop_id(r, b["bbox"]) not in embeddings])]

    import open_clip
    import torch
    device = _device()
    model, _, preprocess = open_clip.create_model_and_transforms(cfg.species_model)
    tokenizer = open_clip.get_tokenizer(cfg.species_model)
    model = model.to(device).eval()

    if jobs:
        print(f"Embedding crops for {len(jobs)} files on {device} ({len(animals) - len(jobs)} cached)")
        pool = ThreadPoolExecutor(max_workers=cfg.workers)
        batch, ids = [], []

        def flush():
            with torch.no_grad():
                feats = model.encode_image(torch.stack(batch).to(device))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            for crop_id, f in zip(ids, feats.cpu().numpy()):
                embeddings[crop_id] = _encode(f)
            batch.clear()
            ids.clear()

        try:
            loaded = pool.map(lambda job: (job, _crops(*job)), jobs)
            for (r, boxes), crops in tqdm(loaded, total=len(jobs), unit="file", dynamic_ncols=True):
                for b, crop in zip(boxes, crops):
                    batch.append(preprocess(crop))
                    ids.append(_crop_id(r, b["bbox"]))
                if len(batch) >= BATCH_SIZE:
                    flush()
            if batch:
                flush()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            cfg.species_dir.mkdir(parents=True, exist_ok=True)
            emb_path.write_text(json.dumps(embeddings))

    names = list(cfg.species_labels)
    with torch.no_grad():
        text = []
        for name in names:
            t = model.encode_text(tokenizer([f"a photo of {p}." for p in cfg.species_labels[name]]).to(device))
            t = (t / t.norm(dim=-1, keepdim=True)).mean(0)
            text.append(t / t.norm())
        text = torch.stack(text).cpu().numpy()
    scale = float(model.logit_scale.exp().item())
    classifier = _train_on_labels(cfg, animals, embeddings)
    manual = load_labels(cfg)

    # Merged into earlier results, so a run on a sample folder keeps the full set's labels.
    by_key_path = cfg.species_dir / "by_key.json"
    by_key = json.loads(by_key_path.read_text()) if by_key_path.exists() else {}
    for r in animals:
        scored = []
        for b in _animal_boxes(r):
            emb = _decode(embeddings[_crop_id(r, b["bbox"])])
            logits = scale * (text @ emb)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            i = int(p.argmax())
            label, score, source = names[i], float(p[i]), "zero-shot"
            if classifier is not None:
                probs = classifier.predict_proba(emb[None])[0]
                j = int(probs.argmax())
                # Keep a confident zero-shot label for a class you have no examples of yet.
                if not (label not in classifier.classes_ and score >= cfg.zero_shot_min):
                    label, score, source = str(classifier.classes_[j]), float(probs[j]), "examples"
            scored.append({"bbox": b["bbox"], "label": label, "score": round(score, 4), "source": source})
        if not scored:  # best frame is a person/vehicle frame without an animal box
            continue
        if r["_key"] in manual:
            scored[0].update(label=manual[r["_key"]], score=1.0, source="manual")
        by_key[r["_key"]] = {"t": r["best_frame_time_s"], "label": scored[0]["label"],
                             "score": scored[0]["score"], "source": scored[0]["source"], "boxes": scored}
    by_key_path.write_text(json.dumps(by_key, indent=1))
    counts = Counter(by_key[r["_key"]]["label"] for r in animals if r["_key"] in by_key)
    print("Labels: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))


def _train_on_labels(cfg: Config, animals: list[dict], embeddings: dict[str, str]):
    """Logistic regression on the crop embeddings of hand-labelled files, or None."""
    labels = load_labels(cfg)
    if not labels:
        return None
    X, y = [], []
    for r in animals:
        boxes = _animal_boxes(r)
        if r["_key"] in labels and boxes:
            X.append(_decode(embeddings[_crop_id(r, boxes[0]["bbox"])]))
            y.append(labels[r["_key"]])
    counts = Counter(y)
    print(f"{len(y)} labelled examples from {cfg.labels_file.name}: "
          + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    if len(labels) > len(y):
        print(f"  {len(labels) - len(y)} labels skipped (file is not a current animal hit)")
    if len(counts) < 2:
        print("  Need examples for at least two labels to train; using zero-shot only.")
        return None

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X = np.stack(X)
    classifier = LogisticRegression(max_iter=5000, class_weight="balanced", C=10.0)
    folds = min(5, min(counts.values()))
    if folds >= 2:
        acc = cross_val_score(classifier, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0))
        print(f"  Cross-validated accuracy on your examples: {acc.mean():.0%} ({folds} folds)")
    else:
        print("  Add a second example to every label to get an accuracy estimate.")
    return classifier.fit(X, y)
