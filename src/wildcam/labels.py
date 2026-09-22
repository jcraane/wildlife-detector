"""Hand-made species labels exported from the report ("Export labels" button)."""

from __future__ import annotations

import csv
from pathlib import Path

from .config import Config
from .scan import content_key

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
            if not key and row.get("path") and Path(row["path"]).expanduser().is_file():
                key = content_key(Path(row["path"]).expanduser().resolve())
            if key:
                labels[key] = label
    return labels
