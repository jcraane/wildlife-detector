"""Hand-made species labels exported from the report ("Export labels" button)."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .config import Config

NO_ANIMAL = "no animal"


def load_labels(cfg: Config) -> dict[str, str]:
    """Maps cache key -> label. Rows with an empty label are ignored."""
    path = cfg.labels_file
    if not path.exists():
        return {}
    labels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            if not label:
                continue
            key = (row.get("key") or "").strip()
            if not key and row.get("path"):
                key = hashlib.sha1(str(Path(row["path"]).expanduser().resolve()).encode()).hexdigest()[:16]
            if key:
                labels[key] = label
    return labels
