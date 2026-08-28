"""Generate out/court_click.html: a static, server-less landmark clicker for the browser.

    .venv/bin/python vision/court/click_page.py

Exports the keyframe stills at native resolution to out/court_click/, writes the
page with a court sketch (from vision.court.geometry) that marks the landmark
being asked for, a 2x magnifier, keyboard control, localStorage state and a
"Download court_points.json" button. vision/court/from_points.py turns that file
into court_calib_<clip>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision.court.geometry import FIBA, polylines  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"

STILLS = [("dev60", 1500), ("dev60", 1000), ("game10", 8500), ("game10", 12000), ("game10", 26000)]  # dev60 = game10[3000:], so 4500 would repeat 1500

# Most useful first for a camera that sees the far basket (side 0 = far baseline).
ORDER = [
    "paint_0_base_bottom", "paint_0_base_top", "paint_0_front_bottom", "paint_0_front_top",
    "three_0_bottom", "three_0_top", "corner_bl", "corner_tl", "halfway_bottom", "halfway_top",
    "paint_1_front_bottom", "paint_1_front_top", "paint_1_base_bottom", "paint_1_base_top",
    "three_1_bottom", "three_1_top", "corner_br", "corner_tr",
]
LABELS = {
    "paint_0_base_bottom": "Zone am fernen Korb: Ecke an der Grundlinie, links",
    "paint_0_base_top": "Zone am fernen Korb: Ecke an der Grundlinie, rechts",
    "paint_0_front_bottom": "Zone am fernen Korb: Freiwurflinie, linke Ecke",
    "paint_0_front_top": "Zone am fernen Korb: Freiwurflinie, rechte Ecke",
    "three_0_bottom": "Dreierlinie trifft ferne Grundlinie, links",
    "three_0_top": "Dreierlinie trifft ferne Grundlinie, rechts",
    "corner_bl": "Ferne Grundlinie: Ecke links (Seitenlinie)",
    "corner_tl": "Ferne Grundlinie: Ecke rechts (Seitenlinie)",
    "halfway_bottom": "Mittellinie: Ende links (Seitenlinie)",
    "halfway_top": "Mittellinie: Ende rechts (Seitenlinie)",
    "paint_1_front_bottom": "Zone am nahen Korb: Freiwurflinie, linke Ecke",
    "paint_1_front_top": "Zone am nahen Korb: Freiwurflinie, rechte Ecke",
    "paint_1_base_bottom": "Zone am nahen Korb: Ecke an der Grundlinie, links",
    "paint_1_base_top": "Zone am nahen Korb: Ecke an der Grundlinie, rechts",
    "three_1_bottom": "Dreierlinie trifft nahe Grundlinie, links",
    "three_1_top": "Dreierlinie trifft nahe Grundlinie, rechts",
    "corner_br": "Nahe Grundlinie: Ecke links (Seitenlinie)",
    "corner_tr": "Nahe Grundlinie: Ecke rechts (Seitenlinie)",
}


def export_stills() -> list[dict]:
    d = OUT / "court_click"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    caps = {}
    for clip, frame in STILLS:
        path = d / f"{clip}_{frame}.jpg"
        if not path.exists():
            cap = caps.setdefault(clip, cv2.VideoCapture(str(ROOT / "data" / "clips" / f"{clip}.mp4")))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if not ok:
                print(f"Frame {frame} aus {clip} nicht lesbar, übersprungen")
                continue
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
        img = cv2.imread(str(path))
        out.append({"clip": clip, "frame": frame, "src": f"court_click/{path.name}", "w": img.shape[1], "h": img.shape[0]})
    return out


def sketch_svg() -> str:
    """Court seen from the camera: far baseline (court x = 0) at the top, court y to the right."""
    s = 9.0  # svg units per metre
    m = 8
    W = FIBA.width_m * s + 2 * m
    H = FIBA.length_m * s + 2 * m

    def P(x, y):
        return f"{m + y * s:.1f},{m + x * s:.1f}"

    parts = [f'<svg id="sketch" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}">']
    parts.append(f'<rect x="{m}" y="{m}" width="{FIBA.width_m * s:.1f}" height="{FIBA.length_m * s:.1f}" fill="#2b2f38"/>')
    for poly in polylines(FIBA):
        parts.append('<polyline fill="none" stroke="#cfd4dd" stroke-width="1" points="' + " ".join(P(x, y) for x, y in poly) + '"/>')
    for lm in FIBA.landmarks:
        cx, cy = P(lm.x, lm.y).split(",")
        parts.append(f'<circle class="lm" data-id="{lm.id}" cx="{cx}" cy="{cy}" r="3.2"/>')
    parts.append(f'<text x="{W / 2:.0f}" y="{m - 2}" text-anchor="middle" font-size="7" fill="#8b93a7">ferner Korb (Kamera schaut hierhin)</text>')
    parts.append(f'<text x="{W / 2:.0f}" y="{H - 1:.0f}" text-anchor="middle" font-size="7" fill="#8b93a7">naher Korb</text>')
    parts.append("</svg>")
    return "".join(parts)


HTML = """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Court Klicken</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--text:#e6e8ee;--muted:#8b93a7;--ok:#3fb950;--warn:#f0b429;--cur:#3B82F6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{display:grid;grid-template-columns:1fr 380px;gap:14px;padding:14px;min-height:100vh}
.stage{position:relative;background:#000;border:1px solid var(--line);border-radius:8px;overflow:hidden;align-self:start}
#img{display:block;width:100%;height:auto;cursor:crosshair;user-select:none;-webkit-user-drag:none}
#marks{position:absolute;inset:0;pointer-events:none}
.mark{position:absolute;width:12px;height:12px;margin:-6px 0 0 -6px;border:2px solid var(--ok);border-radius:50%}
.mark.cur{border-color:var(--cur)}
.mark span{position:absolute;left:12px;top:-4px;font-size:11px;color:var(--ok);white-space:nowrap;text-shadow:0 0 3px #000}
#mag{position:absolute;right:10px;bottom:10px;width:260px;height:260px;border:1px solid #666;background:#000;pointer-events:none;display:none}
.side{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:10px;align-self:start;position:sticky;top:14px}
h1{font-size:16px;margin:0}.muted{color:var(--muted);font-size:12px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{background:#232838;color:var(--text);border:1px solid #3a4256;border-radius:6px;padding:6px 10px;font:inherit;cursor:pointer}
button.primary{background:var(--cur);border-color:var(--cur);color:#fff}
#list{list-style:none;margin:0;padding:0;max-height:46vh;overflow:auto;border:1px solid var(--line);border-radius:6px}
#list li{padding:5px 8px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;gap:8px;align-items:center;font-size:13px}
#list li.cur{background:#1f2a44}#list li.done{color:var(--ok)}#list li .k{width:16px;text-align:center;color:var(--muted)}
.lm{fill:#444a57}.lm.done{fill:var(--ok)}.lm.cur{fill:var(--cur);r:5}
#msg{min-height:38px;font-size:13px}
kbd{background:#232838;border:1px solid #3a4256;border-radius:4px;padding:0 5px;font-size:12px}
</style></head><body>
<div class="wrap">
 <div class="stage"><img id="img" draggable="false"><div id="marks"></div><canvas id="mag" width="260" height="260"></canvas></div>
 <div class="side">
  <h1>Court klicken</h1>
  <div class="row"><button id="prev">Vorheriges Bild</button><button id="next">Naechstes Bild</button><span id="which" class="muted"></span></div>
  <div id="cursketch">__SKETCH__</div>
  <div><b id="curlabel"></b><div class="muted">Klick setzt den blauen Punkt. <kbd>n</kbd> ueberspringen, <kbd>u</kbd> zurueck, <kbd>&larr;</kbd> <kbd>&rarr;</kbd> Bild wechseln. Punkte, die nicht im Bild sind, einfach ueberspringen. 6 bis 8 gute Punkte pro Bild reichen.</div></div>
  <ul id="list"></ul>
  <div id="msg" class="muted"></div>
  <div class="row"><button id="dl" class="primary">Download court_points.json</button><button id="clear">Bild leeren</button></div>
  <div class="muted">Stand wird im Browser gespeichert. Nach dem Download an das Terminal: <code>.venv/bin/python vision/court/from_points.py ~/Downloads/court_points.json</code></div>
 </div>
</div>
<script>
const IMAGES = __IMAGES__;
const ORDER = __ORDER__;
const LABELS = __LABELS__;
const KEY = "court_click_v1";
let state = {}; try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
let idx = 0, cur = 0, history = [];
const img = document.getElementById("img"), marks = document.getElementById("marks"), mag = document.getElementById("mag");
const list = document.getElementById("list"), msg = document.getElementById("msg");
const natImg = new Image();
function store(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch(e){} }
function pts(){ const im = IMAGES[idx]; state[im.clip] = state[im.clip] || {}; state[im.clip][im.frame] = state[im.clip][im.frame] || {}; return state[im.clip][im.frame]; }
function scale(){ return img.clientWidth / IMAGES[idx].w; }
function nextUnset(from){ const p = pts(); for (let k = 0; k < ORDER.length; k++){ const i = (from + k) % ORDER.length; if (!p[ORDER[i]]) return i; } return from % ORDER.length; }
function render(){
  const im = IMAGES[idx], p = pts();
  document.getElementById("which").textContent = `${im.clip} Frame ${im.frame} (${idx + 1}/${IMAGES.length})`;
  const done = ORDER.filter(id => p[id]).length;
  document.getElementById("curlabel").textContent = LABELS[ORDER[cur]];
  list.innerHTML = ORDER.map((id, i) => `<li class="${i === cur ? "cur" : ""} ${p[id] ? "done" : ""}" data-i="${i}"><span class="k">${p[id] ? "x" : ""}</span>${LABELS[id]}</li>`).join("");
  list.querySelectorAll("li").forEach(li => li.onclick = () => { cur = +li.dataset.i; render(); });
  const c = list.querySelector("li.cur"); if (c) c.scrollIntoView({block: "nearest"});
  document.querySelectorAll("#sketch .lm").forEach(el => { el.classList.toggle("done", !!p[el.dataset.id]); el.classList.toggle("cur", el.dataset.id === ORDER[cur]); });
  const s = scale();
  marks.innerHTML = ORDER.filter(id => p[id]).map(id => `<div class="mark ${id === ORDER[cur] ? "cur" : ""}" style="left:${p[id][0] * s}px;top:${p[id][1] * s}px"><span>${id}</span></div>`).join("");
  const total = Object.values(state).reduce((a, c) => a + Object.values(c).reduce((b, f) => b + Object.keys(f).length, 0), 0);
  msg.textContent = `${done} von ${ORDER.length} Punkten in diesem Bild, ${total} gesamt. ` + (done >= 6 ? "Genug fuer dieses Bild." : "Mindestens 6 Punkte, davon nicht alle auf der Grundlinie.");
}
function load(i){ idx = (i + IMAGES.length) % IMAGES.length; img.src = IMAGES[idx].src; natImg.src = IMAGES[idx].src; cur = nextUnset(0); history = []; img.onload = render; render(); }
img.addEventListener("click", ev => {
  const r = img.getBoundingClientRect(), s = scale();
  const x = Math.round((ev.clientX - r.left) / s * 10) / 10, y = Math.round((ev.clientY - r.top) / s * 10) / 10;
  const id = ORDER[cur]; pts()[id] = [x, y]; history.push(id); store(); cur = nextUnset(cur + 1); render();
});
img.addEventListener("mousemove", ev => {
  const r = img.getBoundingClientRect(), s = scale();
  const x = (ev.clientX - r.left) / s, y = (ev.clientY - r.top) / s;
  const ctx = mag.getContext("2d"), z = 2, half = 65;
  ctx.imageSmoothingEnabled = false; ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 260, 260);
  try { ctx.drawImage(natImg, x - half, y - half, 2 * half, 2 * half, 0, 0, 260, 260); } catch (e) {}
  ctx.strokeStyle = "#ff3b3b"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(130, 110); ctx.lineTo(130, 150); ctx.moveTo(110, 130); ctx.lineTo(150, 130); ctx.stroke();
  mag.style.display = "block";
  mag.style.right = (ev.clientX - r.left < r.width * 0.6) ? "10px" : ""; mag.style.left = (ev.clientX - r.left < r.width * 0.6) ? "" : "10px";
});
img.addEventListener("mouseleave", () => mag.style.display = "none");
document.addEventListener("keydown", ev => {
  if (ev.key === "n" || ev.key === "j") { cur = (cur + 1) % ORDER.length; render(); }
  else if (ev.key === "k") { cur = (cur - 1 + ORDER.length) % ORDER.length; render(); }
  else if (ev.key === "u") { const id = history.pop(); if (id) { delete pts()[id]; cur = ORDER.indexOf(id); store(); render(); } }
  else if (ev.key === "ArrowRight") load(idx + 1);
  else if (ev.key === "ArrowLeft") load(idx - 1);
});
document.getElementById("next").onclick = () => load(idx + 1);
document.getElementById("prev").onclick = () => load(idx - 1);
document.getElementById("clear").onclick = () => { const im = IMAGES[idx]; if (state[im.clip]) delete state[im.clip][im.frame]; store(); cur = 0; render(); };
document.getElementById("dl").onclick = () => {
  const out = {clips: {}};
  for (const clip in state) for (const fr in state[clip]) if (Object.keys(state[clip][fr]).length) { out.clips[clip] = out.clips[clip] || {}; out.clips[clip][fr] = state[clip][fr]; }
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "court_points.json"; document.body.appendChild(a); a.click(); a.remove();
};
window.addEventListener("resize", render);
load(0);
</script></body></html>
"""


def main() -> int:
    images = export_stills()
    page = (HTML.replace("__SKETCH__", sketch_svg())
            .replace("__IMAGES__", json.dumps(images))
            .replace("__ORDER__", json.dumps(ORDER))
            .replace("__LABELS__", json.dumps(LABELS, ensure_ascii=False)))
    out = OUT / "court_click.html"
    out.write_text(page)
    print(f"geschrieben: {out} ({len(images)} Bilder)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
