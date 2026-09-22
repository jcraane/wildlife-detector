# wildcam

Finds the trail camera photos and clips that contain an animal, person or
vehicle, and skips the empty triggers (wind, grass, light changes).

- Detector: [MegaDetector](https://github.com/agentmorris/MegaDetector) v1000-redwood,
  via the `megadetector` package. Upstream recommends it as the most accurate model.
- Optional species labels: [BioCLIP-2](https://huggingface.co/imageomics/bioclip-2) with a list of local species.
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
| `uv run wildcam species` | BioCLIP-2 species labels for the current animal hits |
| `uv run wildcam run` | `detect`, then `species` if enabled, then `aggregate` |

Common flags: `--input DIR`, `--output DIR`, `--threshold X`, `--fps X`.
Test run on the sample: `uv run wildcam run --input sample --output output-sample`.

## Re-running and resuming

Each file's detections are cached in `cache/<model>_fps<fps>/`. The cache key
is based on the file's content: its size plus a hash of the first and last
64 KB. A file keeps its cached results and your labels even after it's moved
or renamed.
- Interrupting a run (Ctrl-C) is safe. The next `run` or `detect` skips
  everything that's already cached.
- `--redetect` forces all files to be processed again.

## Adding new footage

The footage folder is `input_dir`, and it's scanned recursively. Set your own
path in `config.local.toml`, which isn't committed. Its values override
`config.toml`:

```toml
input_dir = "~/path/to/WildlifeCamera"
```

Copy each SD card into its own dated folder:

```
WildlifeCamera/
  2026-09-22/100MEDIA/DSCF0001.JPG ...
  2026-10-05/100MEDIA/DSCF0001.JPG ...   <- new card, own folder
```

Then run `uv run wildcam run`. Only the new files are processed, and the
species step labels them using your existing corrections.

Don't copy a new card's `100MEDIA` into an existing one. Trail cameras
restart their numbering, so Finder would replace older files with the same
names. Moving or renaming existing footage is fine: the content-based cache
key still recognises it.

With Google Drive for desktop, first-time processing reads every file. Mark
the folder "Available offline" in Drive if files are online-only, to avoid
downloading them one at a time.

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

Animal hits can be labelled with [BioCLIP-2](https://huggingface.co/imageomics/bioclip-2)
(MIT license), a species model that picks the best match from the list of
labels you give it. Each MegaDetector animal box on a hit's best frame is
cropped from the full-resolution image and scored against the labels in
`config.toml`:

```toml
[species]
enabled = true

[species.labels]
"domestic rabbit" = "Oryctolagus cuniculus domesticus, domestic rabbit"
"red squirrel" = "Sciurus vulgaris, red squirrel"
# ...
```

Run it with `uv run wildcam run --species`, or on its own with
`uv run wildcam species`, then `uv run wildcam aggregate`. The first run
downloads the model (~1.7 GB). Crop embeddings are cached in
`cache/species_bioclip-2/`, so after editing the labels a re-run only
re-scores and takes seconds. Results appear in the CSV (`species`,
`species_score`, and `species_all` for every animal box on the best frame)
and as a species filter in the report.

Why not SpeciesNet: it was tried first. On this camera it labelled the pet
rabbits as cat or dog and the rodents as American species, because it can
answer with any of its ~2,000 classes. BioCLIP-2 with a short local list was
far more accurate on the same sample.

### Improving the labels with your own corrections

Zero-shot labels are rough: black-and-white pet rabbits, for example, often
come out as "domestic cat". Correcting a few cards teaches the pipeline what
the animals at your camera look like:

1. In `output/report.html`, sort by **Species score, lowest first**. Type the
   right label in a card's label box, or press **✓** to accept the prediction.
   Any label works, including ones that aren't in `config.toml`, such as
   `no animal` or `rabbit on lens`. Edits are kept in your browser until
   you export them.
2. Click **Export labels**. Your browser saves `labels.csv`.
3. Move it into the project and re-run:

   ```sh
   mv ~/Downloads/labels.csv .
   uv run wildcam species && uv run wildcam aggregate
   ```

With examples for two or more labels, a logistic regression is trained on
the cached crop embeddings of your labelled files. It then labels all the
other hits, which takes seconds, and prints a cross-validated accuracy once
every label has at least two examples. Your own label always wins for its
file. A zero-shot label you have no examples of yet is kept when it scores at
least `zero_shot_min` (0.9), so a rare visitor still shows up. A file
labelled `no animal`, or predicted as it, moves to `empty`.

About 10–20 examples for each common case and a few for each rare animal is
a good start. Repeat steps 1–3 as needed: the report loads your existing
labels, and each export contains all of them. `labels.csv` is in
`.gitignore`, because it contains paths to your footage.

Limits:
- Without examples, only the listed labels can be returned. Add an animal to the list if it
  visits and isn't there.
- Zero-shot has no working "no animal" label, so MegaDetector false
  positives get the closest species until you label a few as `no animal`.
- Mouse and rat species aren't reliably told apart on night footage; read
  those labels as "rodent".

## Notes on this camera (SILTCON trail camera)

- Each trigger writes one photo and one ~10 s 4K video with the same capture time.
- Photos have no EXIF date. Their file modification time matches the video's
  `creation_time` and is used as the capture time (`capture_time_source = file_mtime`).
- The video `creation_time` is camera local time written with a `Z` suffix,
  so it is treated as local time.
- Zero-byte files (failed writes on the card) show up as `error`.
