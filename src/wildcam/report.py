"""Self-contained HTML report: one card per hit, best frame with boxes, filterable."""

from __future__ import annotations

import base64
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image
from megadetector.visualization.visualization_utils import load_image

from .aggregate import CATEGORIES
from .config import Config

THUMB_WIDTH = 400
THUMB_QUALITY = 70
PAGE_SIZE = 120  # cards rendered per scroll step


def _thumbnail(cfg: Config, row: dict) -> str | None:
    """Base64 JPEG of the best frame (boxes are drawn as SVG on top, not burned in)."""
    thumb = cfg.detections_dir / f"thumbs{THUMB_WIDTH}" / f"{row['_key']}.jpg"
    if not thumb.exists():
        if row["type"] == "video":
            if not row["_best_frame_file"]:
                return None
            img = Image.open(cfg.detections_dir / "frames" / row["_best_frame_file"])
        else:
            img = load_image(row["original_path"])
        img = img.convert("RGB")
        img.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
        thumb.parent.mkdir(parents=True, exist_ok=True)
        img.save(thumb, quality=THUMB_QUALITY)
    return base64.b64encode(thumb.read_bytes()).decode()


def write_report(cfg: Config, rows: list[dict]) -> Path:
    hits = [r for r in rows if r["category"] in CATEGORIES]
    counts = Counter(r["category"] for r in rows)
    items = []
    for r in hits:
        items.append({
            "img": _thumbnail(cfg, r),
            "boxes": r["_boxes"],
            "path": r["path"],
            "href": Path(r["original_path"]).as_uri(),
            "type": r["type"],
            "cats": r["categories"].split(";"),
            "conf": r["max_conf"],
            "t": r["best_frame_time_s"],
            "hitFrames": r["hit_frames"],
            "frames": r["sampled_frames"],
            "when": r["capture_time"],
            "species": r["species"],
            "speciesScore": r["species_score"],
        })
    summary = {
        "total": len(rows), "threshold": cfg.threshold,
        "thresholdText": ", ".join([f"{cfg.threshold}"] + [f"{k} {v}" for k, v in cfg.category_thresholds.items()]),
        "model": cfg.model, "fps": cfg.fps,
        "counts": {k: counts.get(k, 0) for k in (*CATEGORIES, "empty", "error", "not_processed")},
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "input": str(cfg.input_dir),
    }
    page = TEMPLATE.replace("__PAGE_SIZE__", str(PAGE_SIZE)).replace("__DATA__", json.dumps({"summary": summary, "items": items})
                            .replace("</", "<\\/"))
    out = cfg.output_dir / "report.html"
    out.write_text(page)
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wildlife Camera Hits</title>
<style>
:root {
  --bg: #f6f5f1; --surface: #ffffff; --text: #1d1f1c; --muted: #6b6f68; --line: #e2e1db;
  --animal: #2f8f4e; --person: #c2410c; --vehicle: #2563eb; --accent: #1d1f1c;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141614; --surface: #1d201d; --text: #e9ebe6; --muted: #9aa097; --line: #2d312c;
    --animal: #5cc97f; --person: #fb8a4c; --vehicle: #6ea0ff; --accent: #e9ebe6;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
header { padding: 24px 16px 8px; max-width: 1400px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
.meta { color: var(--muted); font-size: 13px; }
.stats { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 4px; }
.stat { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 8px 12px; min-width: 92px; }
.stat b { display: block; font-size: 20px; font-variant-numeric: tabular-nums; }
.stat span { color: var(--muted); font-size: 12px; }
.controls { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line); }
.controls .inner { max-width: 1400px; margin: 0 auto; padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { border: 1px solid var(--line); background: var(--surface); color: var(--text); border-radius: 999px;
  padding: 5px 12px; cursor: pointer; font: inherit; }
.chip[aria-pressed="true"] { background: var(--accent); color: var(--bg); border-color: var(--accent); }
label.ctl { display: flex; align-items: center; gap: 8px; color: var(--muted); }
input[type=range] { width: 160px; }
input[type=search], select { font: inherit; padding: 5px 8px; border-radius: 8px; border: 1px solid var(--line);
  background: var(--surface); color: var(--text); }
main { max-width: 1400px; margin: 0 auto; padding: 16px; display: grid; gap: 14px;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.frame { position: relative; display: block; background: #000; aspect-ratio: 16 / 9; }
.frame img { width: 100%; height: 100%; object-fit: contain; display: block; }
.frame svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.frame .noimg { color: #aaa; display: grid; place-items: center; height: 100%; }
.body { padding: 10px 12px 12px; }
.row { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
.tags { display: flex; gap: 4px; flex-wrap: wrap; }
.tag { font-size: 12px; font-weight: 600; padding: 1px 8px; border-radius: 999px; color: #fff; }
.tag.animal { background: var(--animal); } .tag.person { background: var(--person); } .tag.vehicle { background: var(--vehicle); }
.conf { font-variant-numeric: tabular-nums; font-weight: 600; }
.species { margin-top: 6px; font-weight: 600; }
.species small { color: var(--muted); font-weight: 400; }
.details { color: var(--muted); font-size: 12px; margin-top: 6px; }
.details a { color: inherit; word-break: break-all; }
.empty-state { grid-column: 1 / -1; color: var(--muted); text-align: center; padding: 48px 0; }
</style>
</head>
<body>
<header>
  <h1>Wildlife camera hits</h1>
  <div class="meta" id="meta"></div>
  <div class="stats" id="stats"></div>
</header>
<div class="controls"><div class="inner">
  <div class="chips" id="chips"></div>
  <label class="ctl">Min confidence <input type="range" id="minConf" min="0" max="1" step="0.01"><span id="minConfVal"></span></label>
  <label class="ctl">Sort <select id="sort"><option value="conf">Confidence</option><option value="time">Capture time</option></select></label>
  <input type="search" id="q" placeholder="Filter by file or species">
</div></div>
<main id="grid"></main>
<div id="more" style="height:1px"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const {summary, items} = JSON.parse(document.getElementById('data').textContent);
const colors = {animal: 'var(--animal)', person: 'var(--person)', vehicle: 'var(--vehicle)'};
const state = {cat: 'all', minConf: summary.threshold, sort: 'conf', q: ''};

document.getElementById('meta').textContent =
  `${summary.input} · model ${summary.model} · ${summary.fps} frame/s · threshold ${summary.thresholdText} · generated ${summary.generated}`;
const c = summary.counts;
document.getElementById('stats').innerHTML = [
  ['Files', summary.total], ['Animal', c.animal], ['Person', c.person], ['Vehicle', c.vehicle],
  ['Empty', c.empty], ...(c.error ? [['Errors', c.error]] : []), ...(c.not_processed ? [['Not processed', c.not_processed]] : []),
].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');

const chips = document.getElementById('chips');
for (const cat of ['all', 'animal', 'person', 'vehicle']) {
  const n = cat === 'all' ? items.length : items.filter(i => i.cats.includes(cat)).length;
  const b = document.createElement('button');
  b.className = 'chip'; b.dataset.cat = cat; b.textContent = `${cat} (${n})`;
  b.onclick = () => { state.cat = cat; render(); };
  chips.appendChild(b);
}
const slider = document.getElementById('minConf');
slider.min = summary.threshold; slider.value = summary.threshold;
slider.oninput = () => { state.minConf = +slider.value; render(); };
document.getElementById('sort').onchange = e => { state.sort = e.target.value; render(); };
document.getElementById('q').oninput = e => { state.q = e.target.value.toLowerCase(); render(); };

const esc = s => String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));

function boxes(item) {
  return item.boxes.filter(b => b.conf >= state.minConf).map(b => {
    const [x, y, w, h] = b.bbox.map(v => v * 100);
    const col = colors[b.category] || '#fff';
    return `<rect x="${x}%" y="${y}%" width="${w}%" height="${h}%" fill="none" stroke="${col}" stroke-width="2.5"/>` +
      `<text x="${x}%" y="${y}%" dy="-4" fill="${col}" font-size="13" font-weight="700" paint-order="stroke" stroke="#000" stroke-width="3">${b.category} ${b.conf.toFixed(2)}</text>`;
  }).join('');
}

function card(item) {
  const img = item.img
    ? `<img loading="lazy" src="data:image/jpeg;base64,${item.img}" alt="">`
    : `<div class="noimg">no frame</div>`;
  const when = item.type === 'video'
    ? `video · best frame at ${item.t}s · ${item.hitFrames}/${item.frames} frames hit`
    : 'photo';
  const species = item.species
    ? `<div class="species">${esc(item.species)} <small>${(+item.speciesScore).toFixed(2)}</small></div>` : '';
  return `<article class="card">
    <a class="frame" href="${item.href}" target="_blank" title="Open original">${img}<svg>${boxes(item)}</svg></a>
    <div class="body">
      <div class="row"><div class="tags">${item.cats.map(c => `<span class="tag ${c}">${c}</span>`).join('')}</div>
        <span class="conf">${item.conf.toFixed(2)}</span></div>
      ${species}
      <div class="details">${esc(item.when)} · ${when}<br><a href="${item.href}" target="_blank">${esc(item.path)}</a></div>
    </div></article>`;
}

function render() {
  for (const b of chips.children) b.setAttribute('aria-pressed', b.dataset.cat === state.cat);
  document.getElementById('minConfVal').textContent = state.minConf.toFixed(2);
  let list = items.filter(i =>
    (state.cat === 'all' || i.cats.includes(state.cat)) && i.conf >= state.minConf &&
    (!state.q || i.path.toLowerCase().includes(state.q) || (i.species || '').toLowerCase().includes(state.q)));
  list.sort(state.sort === 'conf' ? (a, b) => b.conf - a.conf : (a, b) => (a.when || '').localeCompare(b.when || ''));
  current = list; shown = 0;
  grid.innerHTML = list.length ? '' : '<div class="empty-state">No hits match these filters.</div>';
  showMore();
}

// Cards are added in pages as you scroll, so thousands of hits stay responsive.
const PAGE_SIZE = __PAGE_SIZE__;
const grid = document.getElementById('grid');
let current = [], shown = 0;
function showMore() {
  if (shown >= current.length) return;
  grid.insertAdjacentHTML('beforeend', current.slice(shown, shown + PAGE_SIZE).map(card).join(''));
  shown += PAGE_SIZE;
}
new IntersectionObserver(entries => { if (entries[0].isIntersecting) showMore(); }, {rootMargin: '1500px'})
  .observe(document.getElementById('more'));
render();
</script>
</body>
</html>
"""
