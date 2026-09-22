"""Command line entry point: `uv run wildcam <command>`."""

from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter
from pathlib import Path

from .config import PROJECT_DIR, load_config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="config file (default: ./config.toml)")
    p.add_argument("--input", type=Path, help="footage folder (overrides input_dir)")
    p.add_argument("--output", type=Path, help="output folder (overrides output_dir)")
    p.add_argument("--threshold", type=float, help="hit confidence threshold (overrides [aggregate] threshold)")
    p.add_argument("--fps", type=float, help="video frames sampled per second (overrides [detect] fps)")


def _config(args):
    cfg = load_config(args.config)
    if args.input:
        cfg.input_dir = args.input.expanduser().resolve()
    if args.output:
        cfg.output_dir = args.output.expanduser().resolve()
    if args.threshold is not None:
        cfg.threshold = args.threshold
    if args.fps is not None:
        cfg.fps = args.fps
    if getattr(args, "species", None) is not None:
        cfg.species_enabled = args.species
    if not cfg.input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {cfg.input_dir}")
    lowest = min([cfg.threshold, *cfg.category_thresholds.values()])
    if lowest < cfg.min_cached_conf:
        raise SystemExit(f"Threshold {lowest} is below min_cached_conf {cfg.min_cached_conf}; "
                         "lower min_cached_conf in config.toml and run detect with --redetect.")
    return cfg


def cmd_scan(cfg, args) -> None:
    from .scan import ffprobe, scan
    files = scan(cfg)
    by_ext = Counter(f.original.suffix.lower() for f in files)
    size = sum(f.original.stat().st_size for f in files)
    print(f"{cfg.input_dir}: {len(files)} media files, {size / 1e9:.2f} GB")
    for ext, n in by_ext.most_common():
        print(f"  {ext}: {n}")
    videos = [f for f in files if f.kind == "video"]
    if videos:
        probe = random.Random(0).sample(videos, min(25, len(videos)))
        durations = [float(ffprobe(v.original)["format"]["duration"]) for v in probe]
        print(f"  video duration (sample of {len(probe)}): median {statistics.median(durations):.1f} s")


def cmd_sample(cfg, args) -> None:
    """Symlinks a random mix of photos and videos into a sample folder (inside this project)."""
    from .scan import scan
    files = scan(cfg)
    rng = random.Random(args.seed)
    images = [f for f in files if f.kind == "image"]
    videos = [f for f in files if f.kind == "video"]
    n_vid = min(len(videos), args.n // 2)
    picked = rng.sample(videos, n_vid) + rng.sample(images, min(len(images), args.n - n_vid))
    dest = args.dest if args.dest.is_absolute() else PROJECT_DIR / args.dest
    for f in sorted(picked, key=lambda f: f.rel):
        link = dest / f.rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(f.original)
    print(f"Linked {len(picked)} files ({n_vid} videos) into {dest}")


def cmd_detect(cfg, args) -> None:
    from .detect import run_detection
    from .scan import scan
    run_detection(cfg, scan(cfg), redetect=args.redetect)


def cmd_species(cfg, args) -> None:
    from .aggregate import aggregate
    from .scan import scan
    from .species import run_species
    run_species(cfg, aggregate(cfg, scan(cfg)))


def cmd_aggregate(cfg, args) -> None:
    from .aggregate import aggregate, write_sorted_view, write_tables
    from .report import write_report
    from .scan import scan
    rows = aggregate(cfg, scan(cfg))
    write_tables(cfg, rows)
    write_sorted_view(cfg, rows)
    report = write_report(cfg, rows)
    counts = Counter(r["category"] for r in rows)
    overrides = "".join(f", {k} {v}" for k, v in cfg.category_thresholds.items())
    print(f"Threshold {cfg.threshold}{overrides}: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    print(f"  {cfg.output_dir / 'results.csv'}\n  {cfg.output_dir / 'sorted'}/\n  {report}")


def cmd_run(cfg, args) -> None:
    cmd_detect(cfg, args)
    if cfg.species_enabled:
        cmd_species(cfg, args)
    cmd_aggregate(cfg, args)


def main() -> None:
    parser = argparse.ArgumentParser(prog="wildcam", description="Find animals, people and vehicles in trail camera footage.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="inventory of the footage folder, no processing")
    _add_common(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("sample", help="symlink a random sample of files into ./sample")
    _add_common(p)
    p.add_argument("-n", type=int, default=20, help="number of files (half videos, half photos)")
    p.add_argument("--dest", type=Path, default=Path("sample"))
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("detect", help="run MegaDetector (cached, resumable)")
    _add_common(p)
    p.add_argument("--redetect", action="store_true", help="ignore the cache and process every file again")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("species", help="run SpeciesNet on animal hits")
    _add_common(p)
    p.set_defaults(func=cmd_species)

    p = sub.add_parser("aggregate", aliases=["report"], help="re-apply the threshold: CSV/JSON, sorted/ view, report.html")
    _add_common(p)
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("run", help="detect, species (if enabled), aggregate")
    _add_common(p)
    p.add_argument("--redetect", action="store_true", help="ignore the cache and process every file again")
    p.add_argument("--species", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable SpeciesNet (overrides [species] enabled)")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(_config(args), args)
