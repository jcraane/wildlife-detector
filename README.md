# wildcam

Finds the trail camera photos and clips that contain an animal, person or
vehicle, and skips the empty triggers (wind, grass, light changes).

- Detector: [MegaDetector](https://github.com/agentmorris/MegaDetector) v1000-redwood,
  via the `megadetector` package. Upstream recommends it as the most accurate model.
- Optional species labels: Google's [SpeciesNet](https://github.com/google/cameratrapai).
- Runs on the Apple GPU (MPS) when available, otherwise CUDA or CPU.
- The footage folder is only ever read. The sorted view uses symlinks.

## Quick start

```sh
uv run wildcam run          # detect (cached) + aggregate + report
open output/report.html
```

Outputs in `output/`:

| File | What |
|---|---|
| `results.csv` / `results.json` | one row per file: path, type, category, confidences, best-frame time, capture time, species |
| `sorted/{animal,person,vehicle,empty,error}/` | symlinks to the originals, named `<capture time>_<folder>_<file>` |
| `report.html` | self-contained report: best frame per hit with boxes, sortable, filterable by category and confidence; click a card to open the original |

## Commands

| Command | Does |
|---|---|
| `uv run wildcam scan` | inventory of the footage folder, no processing |
| `uv run wildcam sample -n 20` | symlinks a random mix of files into `./sample` for a test run |
| `uv run wildcam detect` | runs MegaDetector on new or changed files only |
| `uv run wildcam aggregate` | re-applies thresholds and rewrites CSV, JSON, `sorted/` and report (seconds, no model) |
| `uv run wildcam species` | SpeciesNet on the current animal hits |
| `uv run wildcam run` | `detect`, then `species` if enabled, then `aggregate` |

Common flags: `--input DIR`, `--output DIR`, `--threshold X`, `--fps X`.
Test run on the sample: `uv run wildcam run --input sample --output output-sample`.

## Re-running and resuming

Each file's detections are cached in `cache/<model>_fps<fps>/`. The cache key
is the original file's real path, and its size and modification time are
checked on every run.
- Interrupting a run (Ctrl-C) is safe. The next `run` or `detect` skips
  everything that's already cached.
- Adding new footage to the folder and running `run` processes only the new files.
- `--redetect` forces all files to be processed again.

## Changing the threshold

Thresholds live in `config.toml`:

```toml
[aggregate]
threshold = 0.2            # animal (and any category without an override)

[aggregate.category_thresholds]
person = 0.7
vehicle = 0.8
```

After editing, run `uv run wildcam aggregate`. It takes seconds because the
detections are cached down to confidence 0.01 (`min_cached_conf`). For a
one-off try, use `uv run wildcam aggregate --threshold 0.3`.

The report's "Min confidence" slider hides hits below a value, which lets you
see what a higher threshold would drop before you change the config.

Changing `fps` in `[detect]` uses a separate cache folder, so it needs one detection run.

## Species classification (optional)

SpeciesNet runs in its own uv environment in `species/`. Its dependencies
(onnx/protobuf) conflict with the `megadetector` package, so the two can't
share one environment. `uv` sets it up automatically the first time it's used.

Turn it on with either:

```sh
uv run wildcam run --species        # one-off
```

or `enabled = true` under `[species]` in `config.toml`. Set `country` there
(ISO 3166-1 alpha-3, e.g. `NLD`) so predictions are limited to species that
occur in that country. An empty value turns this off.

SpeciesNet classifies the best frame of every animal hit. For videos, that
frame is re-extracted at full resolution. Predictions are cached in
`cache/speciesnet/` and later runs only classify new hits. The species and its
score appear in the CSV and the report.

## Notes on this camera (SILTCON trail camera)

- Each trigger writes one photo and one ~10 s 4K video with the same capture time.
- Photos have no EXIF date. Their file modification time matches the video's
  `creation_time` and is used as the capture time (`capture_time_source = file_mtime`).
- The video `creation_time` is camera local time written with a `Z` suffix,
  so it is treated as local time.
- Zero-byte files (failed writes on the card) show up as `error`.
