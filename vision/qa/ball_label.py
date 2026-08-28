"""Hand-labelling page for the ball: out/qa/ball_label.html (static, no server).

    .venv/bin/python -m vision.qa.ball_label [--every 5] [--frames data/frames]

Every 5th of data/frames/f_00001.jpg..f_00600.jpg (game10 at 1 fps, frame i =
game10 frame (i-1)*50). One click = ball centre, prefilled in grey from the
game10 tracks when they exist. Export: ball_labels.json
{"frames": [{"file": "f_00006.jpg", "game10_frame": 250, "ball": [cx, cy, r] | null, "status": "ball"|"none"|"open"}]}.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .common import OUT, QA_DIR, ROOT, TrackIndex, read_json, read_tracks

FRAMES_DIR = ROOT / "data" / "frames"
SOURCE_FPS = 50  # game10 is 1080p50, frames were extracted at 1 fps
DEFAULT_R = 17
TRACK_CANDIDATES = (
    (OUT / "game10" / "tracks.jsonl", OUT / "game10" / "tracks_meta.json"),
    (OUT / "tracks.jsonl", OUT / "tracks_meta.json"),
)


def game10_tracks() -> tuple[Path, list[dict]] | None:
    for tracks, meta_path in TRACK_CANDIDATES:
        if not tracks.exists():
            continue
        meta = read_json(meta_path) or {}
        clip = str(meta.get("clip") or "")
        if tracks.parent.name == "game10" or clip.endswith("game10.mp4"):
            return tracks, read_tracks(tracks)
    return None


def build_frames(frames_dir: Path, every: int) -> list[dict]:
    files = sorted(p for p in frames_dir.glob("f_*.jpg") if re.fullmatch(r"f_\d{5}\.jpg", p.name))
    out = []
    for p in files:
        i = int(p.stem[2:])
        if (i - 1) % every == 0:
            out.append({"file": p.name, "index": i, "game10_frame": (i - 1) * SOURCE_FPS, "prefill": None})
    return out


def prefill(frames: list[dict], tracks: list[dict]) -> int:
    index = TrackIndex(tracks)
    n = 0
    for f in frames:
        line = index.nearest(f["game10_frame"], max_gap=index.stride)
        b = (line or {}).get("ball")
        if not b:
            continue
        x1, y1, x2, y2 = b["bbox"]
        c = b.get("center") or [(x1 + x2) / 2, (y1 + y2) / 2]
        f["prefill"] = [round(c[0], 1), round(c[1], 1), round(max(x2 - x1, y2 - y1) / 2, 1)]
        n += 1
    return n


def write_page(out: Path, frames: list[dict], frames_rel: str, tracks_label: str | None) -> Path:
    manifest = json.dumps({"frames": frames, "frames_dir": frames_rel, "default_r": DEFAULT_R, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "tracks": tracks_label})
    page = PAGE.replace("{{MANIFEST}}", manifest).replace("{{N}}", str(len(frames)))
    path = out / "ball_label.html"
    tmp = path.with_suffix(".tmp.html")
    tmp.write_text(page)
    tmp.replace(path)
    return path


PAGE = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ball markieren</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 16px 20px 90px; font: 15px/1.45 -apple-system, system-ui, sans-serif; background: #161616; color: #ececec; user-select: none; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .intro { color: #ddd; margin-bottom: 4px; }
  .keys { color: #9a9a9a; font-size: 13px; margin-bottom: 12px; }
  .keys b { color: #ffd23f; font-weight: 600; }
  .stage { position: relative; width: 1400px; max-width: 100%; }
  canvas#img { display: block; width: 100%; height: auto; border-radius: 6px; cursor: crosshair; background: #000; }
  canvas#mag { position: absolute; width: 200px; height: 200px; border: 2px solid #ffd23f; border-radius: 6px; pointer-events: none; display: none; background: #000; }
  .status { display: flex; gap: 18px; align-items: center; margin: 10px 0; flex-wrap: wrap; }
  .status b { font-weight: 600; }
  .state { padding: 4px 10px; border-radius: 6px; background: #242424; }
  .state.ball { background: #3a3a1a; color: #ffd23f; }
  .state.none { background: #3a1a1a; color: #ff8080; }
  .state.pre { background: #2a2a2a; color: #aaa; }
  .dots { display: flex; flex-wrap: wrap; gap: 3px; margin: 8px 0; max-width: 1400px; }
  .dot { width: 9px; height: 9px; border-radius: 2px; background: #333; cursor: pointer; }
  .dot.ball { background: #ffd23f; } .dot.none { background: #d04040; } .dot.cur { outline: 2px solid #fff; }
  .bar { position: fixed; left: 0; right: 0; bottom: 0; background: #202020; border-top: 1px solid #383838; padding: 12px 20px; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
  .bar button { font: inherit; padding: 8px 16px; border-radius: 6px; border: 1px solid #555; background: #2c2c2c; color: #fff; cursor: pointer; }
  .bar button.primary { background: #ffd23f; color: #111; border-color: #ffd23f; font-weight: 600; }
  #summary { color: #bbb; }
</style>
</head>
<body>
<h1>Ball markieren: game10, {{N}} Bilder</h1>
<div class="intro">Klick auf die Ballmitte. Kein Ball im Bild: n.</div>
<div class="keys"><b>Klick</b> setzt den Ball, graue Vorgabe: Klick darauf uebernimmt sie. <b>Ziehen</b> verschiebt. <b>+</b> / <b>-</b> Radius. <b>n</b> kein Ball. <b>Backspace</b> loeschen. <b>Leertaste</b> oder <b>Pfeil rechts</b> weiter, <b>Pfeil links</b> zurueck. Lupe folgt dem Cursor. Stand bleibt im Browser gespeichert, du kannst jederzeit pausieren.</div>
<div class="stage">
  <canvas id="img" width="1920" height="1080"></canvas>
  <canvas id="mag" width="200" height="200"></canvas>
</div>
<div class="status">
  <span>Bild <b id="pos"></b> von {{N}}</span>
  <span id="file"></span>
  <span id="frame"></span>
  <span class="state" id="state"></span>
</div>
<div class="dots" id="dots"></div>
<div class="bar">
  <button class="primary" id="download">Download ball_labels.json</button>
  <button id="prev">zurueck</button>
  <button id="next">weiter</button>
  <button id="reset">Alles loeschen</button>
  <span id="summary"></span>
</div>
<script>
const M = {{MANIFEST}};
const KEY = "followcam-ball-labels:game10:" + M.frames.length;
const W = 1920, H = 1080, MAG = 200, ZOOM = 2;
const cv = document.getElementById("img"), ctx = cv.getContext("2d");
const mag = document.getElementById("mag"), mctx = mag.getContext("2d");
let cur = 0, labels = {}, img = new Image(), imgReady = false, drag = null, hover = null;
try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s) { labels = s.labels || {}; cur = Math.min(s.cur || 0, M.frames.length - 1); } } catch (e) {}
function save() { try { localStorage.setItem(KEY, JSON.stringify({ labels, cur, saved: new Date().toISOString() })); } catch (e) {} }
function f() { return M.frames[cur]; }
function lab() { return labels[f().file]; }  // undefined = open, null = kein Ball, [cx,cy,r] = Ball
function scale() { return cv.getBoundingClientRect().width / W; }
function toNative(ev) { const r = cv.getBoundingClientRect(); return [(ev.clientX - r.left) * W / r.width, (ev.clientY - r.top) * H / r.height]; }
function load() {
  imgReady = false; img = new Image();
  img.onload = () => { imgReady = true; draw(); };
  img.onerror = () => { ctx.fillStyle = "#300"; ctx.fillRect(0, 0, W, H); ctx.fillStyle = "#fff"; ctx.font = "40px sans-serif"; ctx.fillText("Bild nicht gefunden: " + M.frames_dir + f().file, 40, 100); };
  img.src = M.frames_dir + f().file;
  status();
}
function circle(c, color, dash) {
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 3; if (dash) ctx.setLineDash([8, 6]);
  ctx.beginPath(); ctx.arc(c[0], c[1], c[2], 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(c[0] - 6, c[1]); ctx.lineTo(c[0] + 6, c[1]); ctx.moveTo(c[0], c[1] - 6); ctx.lineTo(c[0], c[1] + 6); ctx.stroke(); ctx.restore();
}
function draw() {
  if (!imgReady) return;
  ctx.drawImage(img, 0, 0, W, H);
  const l = lab();
  if (l === undefined && f().prefill) circle(f().prefill, "#bbbbbb", true);
  if (Array.isArray(l)) circle(l, "#ffd23f", false);
  if (l === null) { ctx.fillStyle = "rgba(200,40,40,0.85)"; ctx.fillRect(0, 0, 420, 60); ctx.fillStyle = "#fff"; ctx.font = "bold 34px sans-serif"; ctx.fillText("kein Ball sichtbar", 16, 44); }
  drawMag();
}
function drawMag() {
  if (!hover || !imgReady) { mag.style.display = "none"; return; }
  const [x, y] = hover, half = MAG / ZOOM / 2;
  mctx.fillStyle = "#000"; mctx.fillRect(0, 0, MAG, MAG);
  mctx.drawImage(img, x - half, y - half, MAG / ZOOM, MAG / ZOOM, 0, 0, MAG, MAG);
  const l = lab(), c = Array.isArray(l) ? l : (l === undefined ? f().prefill : null);
  if (c) { mctx.strokeStyle = Array.isArray(l) ? "#ffd23f" : "#bbb"; mctx.lineWidth = 2; mctx.beginPath(); mctx.arc((c[0] - x + half) * ZOOM, (c[1] - y + half) * ZOOM, c[2] * ZOOM, 0, Math.PI * 2); mctx.stroke(); }
  mctx.strokeStyle = "#ffd23f"; mctx.lineWidth = 1; mctx.beginPath(); mctx.moveTo(MAG / 2 - 10, MAG / 2); mctx.lineTo(MAG / 2 + 10, MAG / 2); mctx.moveTo(MAG / 2, MAG / 2 - 10); mctx.lineTo(MAG / 2, MAG / 2 + 10); mctx.stroke();
  const s = scale(), r = cv.getBoundingClientRect();
  let left = x * s + 24, top = y * s - MAG - 24;
  if (left + MAG > r.width) left = x * s - MAG - 24;
  if (top < 0) top = y * s + 24;
  mag.style.left = left + "px"; mag.style.top = top + "px"; mag.style.display = "block";
}
function status() {
  const l = lab(), st = document.getElementById("state");
  document.getElementById("pos").textContent = cur + 1;
  document.getElementById("file").textContent = f().file;
  document.getElementById("frame").textContent = "game10 Frame " + f().game10_frame + " (" + Math.floor(f().game10_frame / 50 / 60) + ":" + String(Math.floor(f().game10_frame / 50) % 60).padStart(2, "0") + ")";
  if (Array.isArray(l)) { st.textContent = "Ball bei " + Math.round(l[0]) + ", " + Math.round(l[1]) + ", Radius " + Math.round(l[2]); st.className = "state ball"; }
  else if (l === null) { st.textContent = "kein Ball"; st.className = "state none"; }
  else if (f().prefill) { st.textContent = "offen, graue Vorgabe aus den Tracks"; st.className = "state pre"; }
  else { st.textContent = "offen"; st.className = "state"; }
  let nb = 0, nn = 0;
  for (const fr of M.frames) { const v = labels[fr.file]; if (Array.isArray(v)) nb++; else if (v === null) nn++; }
  document.getElementById("summary").textContent = (nb + nn) + " von " + M.frames.length + " beschriftet (" + nb + " Ball, " + nn + " kein Ball)";
  const dots = document.getElementById("dots");
  if (!dots.childElementCount) M.frames.forEach((fr, i) => { const d = document.createElement("span"); d.className = "dot"; d.title = fr.file; d.onclick = () => go(i); dots.appendChild(d); });
  M.frames.forEach((fr, i) => { const v = labels[fr.file]; const d = dots.children[i]; d.className = "dot" + (Array.isArray(v) ? " ball" : v === null ? " none" : "") + (i === cur ? " cur" : ""); });
}
function set(v) { if (v === undefined) delete labels[f().file]; else labels[f().file] = v; save(); draw(); status(); }
function go(i) { cur = Math.max(0, Math.min(M.frames.length - 1, i)); save(); load(); }
function next() { go(cur + 1); }
function prev() { go(cur - 1); }
function radius() { const l = lab(); return Array.isArray(l) ? l[2] : (f().prefill ? f().prefill[2] : M.default_r); }
cv.addEventListener("mousedown", ev => {
  const p = toNative(ev), l = lab();
  if (Array.isArray(l) && Math.hypot(p[0] - l[0], p[1] - l[1]) <= l[2] + 8) { drag = { dx: l[0] - p[0], dy: l[1] - p[1], moved: false }; return; }
  const pre = f().prefill;
  if (l === undefined && pre && Math.hypot(p[0] - pre[0], p[1] - pre[1]) <= pre[2] + 8) { set([pre[0], pre[1], pre[2]]); drag = { dx: 0, dy: 0, moved: false }; return; }
  set([p[0], p[1], radius()]); drag = { dx: 0, dy: 0, moved: false };
});
cv.addEventListener("mousemove", ev => {
  hover = toNative(ev);
  if (drag) { const l = lab(); if (Array.isArray(l)) { l[0] = hover[0] + drag.dx; l[1] = hover[1] + drag.dy; drag.moved = true; save(); } }
  draw();
});
window.addEventListener("mouseup", () => { if (drag) { drag = null; status(); } });
cv.addEventListener("mouseleave", () => { hover = null; drawMag(); });
document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT") return;
  const k = ev.key;
  if (k === " " || k === "ArrowRight") { ev.preventDefault(); next(); }
  else if (k === "ArrowLeft") { ev.preventDefault(); prev(); }
  else if (k === "n" || k === "N") set(null);
  else if (k === "Backspace" || k === "Delete") { ev.preventDefault(); set(undefined); }
  else if (k === "+" || k === "=") { const l = lab(); if (Array.isArray(l)) { l[2] = Math.min(80, l[2] + 1); set(l); } }
  else if (k === "-" || k === "_") { const l = lab(); if (Array.isArray(l)) { l[2] = Math.max(4, l[2] - 1); set(l); } }
});
document.getElementById("next").onclick = next;
document.getElementById("prev").onclick = prev;
document.getElementById("reset").onclick = () => { if (confirm("Alle Markierungen loeschen?")) { labels = {}; save(); draw(); status(); } };
document.getElementById("download").onclick = () => {
  const frames = M.frames.map(fr => { const v = labels[fr.file]; return { file: fr.file, game10_frame: fr.game10_frame,
    ball: Array.isArray(v) ? [Math.round(v[0] * 10) / 10, Math.round(v[1] * 10) / 10, Math.round(v[2] * 10) / 10] : null,
    status: Array.isArray(v) ? "ball" : v === null ? "none" : "open" }; });
  const doc = { clip: "data/clips/game10.mp4", source_fps: 50, image_size: [W, H], generated: M.generated, saved: new Date().toISOString(), frames };
  const blob = new Blob([JSON.stringify(doc, null, 1)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "ball_labels.json";
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};
load();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=Path, default=FRAMES_DIR)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", type=Path, default=QA_DIR)
    args = ap.parse_args(argv)
    frames = build_frames(args.frames, args.every)
    if not frames:
        print(f"no f_*.jpg in {args.frames}")
        return 1
    found = game10_tracks()
    n_pre, label = 0, None
    if found:
        n_pre = prefill(frames, found[1])
        label = str(found[0].relative_to(ROOT))
    args.out.mkdir(parents=True, exist_ok=True)
    rel = Path(*([".."] * len(args.out.resolve().relative_to(ROOT).parts))) / args.frames.resolve().relative_to(ROOT)
    page = write_page(args.out, frames, str(rel) + "/", label)
    print(f"{len(frames)} frames, {n_pre} prefilled from {label or 'no game10 tracks'} -> {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
