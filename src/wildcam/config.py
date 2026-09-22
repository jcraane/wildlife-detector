"""Configuration: defaults, overridden by config.toml, overridden by CLI flags."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    input_dir: Path = Path("~/Downloads/camera")
    output_dir: Path = PROJECT_DIR / "output"
    cache_dir: Path = PROJECT_DIR / "cache"
    model: str = "MDV1000-REDWOOD"
    fps: float = 1.0
    frame_width: int = 1920
    min_cached_conf: float = 0.01
    workers: int = 4
    threshold: float = 0.2
    category_thresholds: dict[str, float] = field(default_factory=dict)
    species_enabled: bool = False
    country: str = "NLD"
    image_exts: frozenset[str] = field(
        default_factory=lambda: frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"}))
    video_exts: frozenset[str] = field(
        default_factory=lambda: frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts"}))

    @property
    def detections_dir(self) -> Path:
        """Cache folder for one model + frame rate combination."""
        fps = f"{self.fps:g}".replace(".", "_")
        return self.cache_dir / f"{self.model.lower()}_fps{fps}"

    def threshold_for(self, category: str) -> float:
        return self.category_thresholds.get(category, self.threshold)

    @property
    def species_dir(self) -> Path:
        return self.cache_dir / "speciesnet"


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_DIR / p


def load_config(path: Path | None = None) -> Config:
    cfg = Config()
    path = path or PROJECT_DIR / "config.toml"
    if path.exists():
        data = tomllib.loads(path.read_text())
        detect = data.get("detect", {})
        aggregate = data.get("aggregate", {})
        species = data.get("species", {})
        cfg.input_dir = Path(data.get("input_dir", cfg.input_dir))
        cfg.output_dir = Path(data.get("output_dir", cfg.output_dir))
        cfg.cache_dir = Path(data.get("cache_dir", cfg.cache_dir))
        cfg.model = detect.get("model", cfg.model)
        cfg.fps = float(detect.get("fps", cfg.fps))
        cfg.frame_width = int(detect.get("frame_width", cfg.frame_width))
        cfg.min_cached_conf = float(detect.get("min_cached_conf", cfg.min_cached_conf))
        cfg.workers = int(detect.get("workers", cfg.workers))
        cfg.threshold = float(aggregate.get("threshold", cfg.threshold))
        cfg.category_thresholds = {k: float(v) for k, v in aggregate.get("category_thresholds", {}).items()}
        cfg.species_enabled = bool(species.get("enabled", cfg.species_enabled))
        cfg.country = species.get("country", cfg.country)
    cfg.input_dir = _resolve(cfg.input_dir)
    cfg.output_dir = _resolve(cfg.output_dir)
    cfg.cache_dir = _resolve(cfg.cache_dir)
    return cfg
