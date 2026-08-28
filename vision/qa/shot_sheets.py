"""Per-shot frame strips for human verification (auto-with-veto).

    .venv/bin/python -m vision.qa.shot_sheets [--events out/events.json] [--tracks out/tracks.jsonl]

For every shot in events.json: 8 frames from 1.5 s before to 1.0 s after the
event, ball box (yellow) + hoop box (green) + shooter box (team color) from
tracks.jsonl, tiled into out/qa/shot_<n>_<made|miss>.jpg. Plus a static
out/qa/index.html where Sami ticks correct / wrong / missed shooter per shot
and downloads the verdicts as json. No server, no dependencies in the page.
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .clips import render_clip
from .common import (
    ROOT,
    BALL_COLOR,
    EVENTS,
    HOOP_COLOR,
    META,
    QA_DIR,
    TEAM_COLORS,
    TEAM_NAMES,
    TRACKS,
    FrameGrabber,
    TrackIndex,
    draw_box,
    fit_height,
    fmt_t,
    put_text,
    qa_lock,
    read_json,
    read_tracks,
    resolve_clip,
    save_jpg,
    tile,
    with_header,
)

BEFORE_S, AFTER_S, N_FRAMES, TILE_H, COLS = 1.5, 1.0, 8, 480, 4


def render_shot(n: int, shot: dict, grab: FrameGrabber, index: TrackIndex) -> tuple[np.ndarray, dict]:
    t0 = float(shot["t"])
    f0 = int(shot.get("frame") if shot.get("frame") is not None else round(t0 * grab.fps))
    offsets = np.linspace(-BEFORE_S, AFTER_S, N_FRAMES)
    event_k = int(np.argmin(np.abs(offsets)))
    tiles = []
    seen_ball = seen_shooter = 0
    for k, off in enumerate(offsets):
        fi = f0 + int(round(off * grab.fps))
        img = grab.get(fi)
        if img is None:
            img = np.zeros((grab.height or 1080, grab.width or 1920, 3), np.uint8)
        line = index.nearest(fi, max_gap=index.stride * 2)
        if line:
            for hp in line.get("hoops") or []:
                draw_box(img, hp["bbox"], HOOP_COLOR, "hoop", 3)
            for p in line.get("players") or []:
                if shot.get("player_id") is not None and p["id"] == shot["player_id"]:
                    draw_box(img, p["bbox"], TEAM_COLORS.get(p.get("team"), TEAM_COLORS[-1]), f"shooter #{p['id']}", 4)
                    seen_shooter += 1
            b = line.get("ball")
            if b:
                draw_box(img, b["bbox"], BALL_COLOR, f"ball {b.get('conf', 0):.2f}", 3)
                seen_ball += 1
        if not (line and line.get("hoops")) and shot.get("hoop_bbox"):
            draw_box(img, shot["hoop_bbox"], HOOP_COLOR, "hoop (event)", 1)
        if shot.get("shooter_foot") and k == event_k:
            x, y = (int(v) for v in shot["shooter_foot"])
            cv2.circle(img, (x, y), 18, (255, 255, 255), 3)
        small = fit_height(img, TILE_H)
        tag = f"{off:+.2f} s   f{fi}   {fmt_t(t0 + off)}"
        if line is None:
            tag += "   (no track line)"
        put_text(small, tag, (10, 26), 0.7, (255, 255, 255), 2)
        if k == event_k:
            cv2.rectangle(small, (0, 0), (small.shape[1] - 1, small.shape[0] - 1), (255, 255, 255), 4)
            put_text(small, "EVENT", (small.shape[1] - 120, small.shape[0] - 14), 0.7, (255, 255, 255), 2)
        tiles.append(small)
    body = tile(tiles, COLS)
    team = shot.get("team", -1)
    verdict = "MADE" if shot.get("made") else "MISS"
    pid = shot.get("player_id")
    shooter = f"shooter #{pid}" if pid is not None else "shooter unknown"
    if pid is not None and shot.get("shooter_confirmed") is False:
        shooter += " (unconfirmed)"
    head = [
        f"shot {n}   t = {fmt_t(t0)} ({t0:.2f} s)   {TEAM_NAMES.get(team, team)}   {verdict}   {shooter}",
        f"frames {f0 - int(BEFORE_S * grab.fps)} .. {f0 + int(AFTER_S * grab.fps)} of {grab.n} @ {grab.fps:g} fps   "
        f"ball box in {seen_ball}/{N_FRAMES} tiles, shooter box in {seen_shooter}/{N_FRAMES}   "
        f"yellow = ball, green = hoop, white circle = shooter foot at event",
    ]
    meta = {
        "n": n,
        "t": t0,
        "t_label": fmt_t(t0),
        "frame": f0,
        "team": team,
        "made": bool(shot.get("made")),
        "player_id": pid,
        "shooter_confirmed": shot.get("shooter_confirmed"),
        "player_key": shot.get("player_key"),
        "made_hint": shot.get("made_hint"),
        "ball_tiles": seen_ball,
        "shooter_tiles": seen_shooter,
    }
    return with_header(body, head), meta


def _local(iso: str | None) -> str:
    """ISO-8601 UTC stamp from the browser -> local wall clock."""
    if not iso:
        return "?"
    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
        return dt.strftime("%H:%M")
    except ValueError:
        return iso


def verdicts_file(out: Path, clip: str) -> Path:
    return out / f"verdicts_{Path(clip).stem}.json"


def write_index(out: Path, sheets: list[dict], clip: str, events_path: Path, tracks_n: int) -> Path:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    made = sum(1 for s in sheets if s["made"])
    stored = read_json(verdicts_file(out, clip)) or {}
    stored_line = (
        f"Stored verdicts from {verdicts_file(out, clip).name} (reviewed {_local(stored.get('reviewed'))}) are pre-selected; "
        f"shots are matched by time, so they survive a rebuild."
        if stored.get("shots")
        else "No stored verdicts yet. Verdicts are kept in this browser until you download them."
    )
    rows = []
    for s in sheets:
        n = s["n"]
        title = (
            f"shot {n}  {s['t_label']}  {TEAM_NAMES.get(s['team'], s['team'])}  "
            f"{'MADE' if s['made'] else 'MISS'}  "
            + (f"shooter #{s['player_id']}" if s["player_id"] is not None else "shooter unknown")
            + (f" ({s['player_key']})" if s.get("player_key") else "")
        )
        if s.get("video_half"):
            video = f"""
  <video controls muted loop autoplay playsinline preload="metadata" src="{s['video_half']}" data-normal="{s['video']}" data-half="{s['video_half']}"></video>
  <div class="speed">
    <button type="button" data-speed="half" class="on">half speed</button>
    <button type="button" data-speed="normal">normal speed</button>
    <span class="cap">{html.escape(s.get('video_caption', ''))}</span>
  </div>"""
        else:
            video = f"""
  <div class="cap">{html.escape(s.get('video_caption', 'no video'))}</div>"""
        rows.append(
            f"""
<section class="shot" id="shot-{n}" data-n="{n}">
  <h2>{html.escape(title)}</h2>{video}
  <div class="cap">frame strip, -1.5 s to +1.0 s, frame-exact reference</div>
  <a href="{s['file']}" target="_blank"><img src="{s['file']}" alt="{html.escape(title)}" loading="lazy"></a>
  <div class="row">
    <label><input type="radio" name="v{n}" value="correct"> correct</label>
    <label><input type="radio" name="v{n}" value="wrong"> wrong</label>
    <label><input type="radio" name="v{n}" value="missed_shooter"> missed shooter</label>
    <input type="text" class="note" name="n{n}" placeholder="note (optional)">
  </div>
</section>"""
        )
    manifest = json.dumps(
        {"clip": clip, "events": str(events_path), "generated": stamp, "tracks_frames": tracks_n, "shots": sheets}
    )
    stored_js = json.dumps({"reviewed": stored.get("reviewed"), "uncalled": stored.get("uncalled_shots", 0), "shots": stored.get("shots", [])})
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shot QA</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; padding: 20px 24px 120px; font: 15px/1.45 -apple-system, system-ui, sans-serif; background: #161616; color: #ececec; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .meta {{ color: #9a9a9a; font-size: 13px; margin-bottom: 18px; }}
  .howto {{ background: #222; border: 1px solid #333; border-radius: 8px; padding: 10px 14px; margin-bottom: 22px; }}
  .howto b {{ color: #ffd23f; }}
  .shot {{ border-top: 1px solid #2c2c2c; padding: 16px 0 10px; }}
  .shot h2 {{ font-size: 16px; margin: 0 0 8px; font-weight: 600; }}
  .shot img {{ display: block; width: 100%; max-width: 1700px; height: auto; border-radius: 6px; }}
  .shot video {{ display: block; width: 100%; max-width: 960px; height: auto; border-radius: 6px; background: #000; }}
  .speed {{ display: flex; gap: 8px; align-items: center; margin: 8px 0 14px; flex-wrap: wrap; }}
  .speed button {{ font: inherit; font-size: 13px; padding: 5px 12px; border-radius: 6px; border: 1px solid #444; background: #242424; color: #ddd; cursor: pointer; }}
  .speed button.on {{ background: #3a3a3a; border-color: #777; color: #fff; }}
  .cap {{ color: #9a9a9a; font-size: 13px; margin: 4px 0 6px; }}
  .row {{ display: flex; gap: 22px; align-items: center; margin-top: 10px; flex-wrap: wrap; }}
  .row label {{ cursor: pointer; user-select: none; padding: 6px 10px; border-radius: 6px; background: #242424; }}
  .row label:has(input:checked) {{ background: #3a3a3a; outline: 1px solid #666; }}
  .row input[type=radio] {{ margin-right: 6px; }}
  .note {{ flex: 1; min-width: 200px; max-width: 480px; background: #1e1e1e; color: #ddd; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px 8px; }}
  .extra {{ border-top: 1px solid #2c2c2c; padding: 18px 0; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
  .extra input[type=number] {{ width: 70px; background: #1e1e1e; color: #ddd; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px 8px; }}
  .bar {{ position: fixed; left: 0; right: 0; bottom: 0; background: #202020; border-top: 1px solid #383838; padding: 12px 24px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
  .bar button {{ font: inherit; padding: 8px 16px; border-radius: 6px; border: 1px solid #555; background: #2c2c2c; color: #fff; cursor: pointer; }}
  .bar button.primary {{ background: #ffd23f; color: #111; border-color: #ffd23f; font-weight: 600; }}
  #summary {{ color: #bbb; }}
  .empty {{ padding: 40px 0; color: #999; }}
</style>
</head>
<body>
<h1>Shot QA: {html.escape(Path(clip).name)}</h1>
<div class="meta">{len(sheets)} shots called by the system ({made} made, {len(sheets) - made} missed), {tracks_n} tracked frames, sheets generated {stamp}. {html.escape(stored_line)}</div>
<div class="howto">Per sheet: <b>correct</b> = shot and made/miss are right. <b>wrong</b> = no shot here or made/miss is flipped. <b>missed shooter</b> = the shot is right but the shooter id is wrong or unknown. Click a sheet to open it full size. Below the list: how many real shots the system did not call at all.</div>
{''.join(rows) if rows else '<div class="empty">No shots in events.json yet.</div>'}
<div class="extra">
  <label>shots the system did not call (seen in overlay.mp4 but not listed above): <input type="number" id="uncalled" min="0" step="1" value="0"></label>
</div>
<div class="bar">
  <button class="primary" id="download">Download verdicts json</button>
  <button id="reset">Reset</button>
  <span id="summary"></span>
</div>
<script>
const MANIFEST = {manifest};
const STORED = {stored_js};
const KEY = "followcam-qa-verdicts:" + MANIFEST.clip;
const MATCH_S = 0.5;
function state() {{
  const shots = MANIFEST.shots.map(s => {{
    const r = document.querySelector('input[name="v' + s.n + '"]:checked');
    const note = document.querySelector('input[name="n' + s.n + '"]');
    return {{ n: s.n, t: s.t, frame: s.frame, team: s.team, made: s.made, player_id: s.player_id,
      verdict: r ? r.value : null, note: note && note.value ? note.value : "" }};
  }});
  return {{ shots, uncalled: Number(document.getElementById("uncalled").value || 0), saved: new Date().toISOString() }};
}}
function counts(shots) {{
  const c = {{ correct: 0, wrong: 0, missed_shooter: 0, open: 0 }};
  for (const s of shots) {{ if (s.verdict) c[s.verdict]++; else c.open++; }}
  return c;
}}
function summary() {{
  const st = state();
  const c = counts(st.shots);
  const rated = c.correct + c.wrong + c.missed_shooter;
  const prec = rated ? Math.round(100 * (c.correct + c.missed_shooter) / rated) : 0;
  document.getElementById("summary").textContent =
    c.correct + " correct, " + c.wrong + " wrong, " + c.missed_shooter + " missed shooter, " + c.open + " open" +
    (rated ? "   shot precision " + prec + "%" : "") +
    (st.uncalled ? "   " + st.uncalled + " uncalled" : "");
  try {{ localStorage.setItem(KEY, JSON.stringify(st)); }} catch (e) {{}}
}}
function apply(src) {{
  if (!src || !src.shots) return false;
  let hit = 0;
  for (const s of MANIFEST.shots) {{
    let best = null;
    for (const v of src.shots) {{
      const d = Math.abs((v.t || 0) - s.t);
      if (d <= MATCH_S && (!best || d < Math.abs(best.t - s.t))) best = v;
    }}
    if (!best) continue;
    if (best.verdict) {{ const r = document.querySelector('input[name="v' + s.n + '"][value="' + best.verdict + '"]'); if (r) {{ r.checked = true; hit++; }} }}
    const note = document.querySelector('input[name="n' + s.n + '"]'); if (note && best.note) note.value = best.note;
  }}
  document.getElementById("uncalled").value = src.uncalled || 0;
  return hit > 0;
}}
function restore() {{
  let local = null;
  try {{ local = JSON.parse(localStorage.getItem(KEY) || "null"); }} catch (e) {{}}
  const fileNewer = STORED.reviewed && (!local || !local.saved || STORED.reviewed > local.saved);
  if (fileNewer) {{ if (!apply(STORED)) apply(local); }}
  else {{ if (!apply(local)) apply(STORED); }}
}}
document.addEventListener("change", summary);
document.addEventListener("input", summary);
document.getElementById("reset").onclick = () => {{
  if (!confirm("Clear all verdicts?")) return;
  document.querySelectorAll("input[type=radio]").forEach(r => r.checked = false);
  document.querySelectorAll("input.note").forEach(n => n.value = "");
  document.getElementById("uncalled").value = 0;
  try {{ localStorage.removeItem(KEY); }} catch (e) {{}}
  summary();
}};
document.getElementById("download").onclick = () => {{
  const st = state();
  const doc = {{ clip: MANIFEST.clip, events: MANIFEST.events, sheets_generated: MANIFEST.generated,
    reviewed: st.saved, counts: counts(st.shots), uncalled_shots: st.uncalled, shots: st.shots }};
  const blob = new Blob([JSON.stringify(doc, null, 1)], {{ type: "application/json" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "verdicts_" + MANIFEST.clip.split("/").pop().replace(/\.[^.]+$/, "") + ".json";
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}};
document.querySelectorAll(".speed button").forEach(btn => {{
  btn.onclick = () => {{
    const wrap = btn.closest(".shot");
    const v = wrap.querySelector("video");
    const speed = btn.dataset.speed;
    const next = v.dataset[speed];
    if (!next || v.getAttribute("src") === next) return;
    const ratio = v.duration ? v.currentTime / v.duration : 0;
    v.setAttribute("src", next);
    v.load();
    v.addEventListener("loadedmetadata", () => {{ v.currentTime = ratio * v.duration; v.play().catch(() => {{}}); }}, {{ once: true }});
    wrap.querySelectorAll(".speed button").forEach(b => b.classList.toggle("on", b === btn));
  }};
}});
if ("IntersectionObserver" in window) {{
  const io = new IntersectionObserver(entries => {{
    for (const e of entries) {{
      const v = e.target;
      if (e.isIntersecting) v.play().catch(() => {{}}); else v.pause();
    }}
  }}, {{ threshold: 0.2 }});
  document.querySelectorAll(".shot video").forEach(v => io.observe(v));
}}
restore();
summary();
</script>
</body>
</html>
"""
    path = out / "index.html"
    tmp = path.with_suffix(".tmp.html")
    tmp.write_text(page)
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", type=Path, default=EVENTS)
    ap.add_argument("--tracks", type=Path, default=TRACKS)
    ap.add_argument("--clip", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=QA_DIR)
    args = ap.parse_args(argv)

    events = read_json(args.events)
    meta = read_json(META) or {}
    frames = read_tracks(args.tracks)
    index = TrackIndex(frames)
    shots = (events or {}).get("shots", [])
    if events is None:
        print(f"{args.events} missing or unreadable, writing an empty index")
    clip = resolve_clip(str(args.clip) if args.clip else None, (events or {}).get("clip"), meta.get("clip"))
    grab = FrameGrabber(clip)
    args.out.mkdir(parents=True, exist_ok=True)
    for old in list(args.out.glob("shot_*.jpg")) + list(args.out.glob("shot_*.mp4")):
        old.unlink()
    sheets = []
    for n, shot in enumerate(shots, 1):
        img, m = render_shot(n, shot, grab, index)
        m["file"] = f"shot_{n}_{'made' if m['made'] else 'miss'}.jpg"
        save_jpg(args.out / m["file"], img)
        try:
            m.update(render_clip(n, m["t"], clip, args.out))
        except RuntimeError as exc:
            print(f"  shot {n}: no video ({exc})")
            m.update({"video": None, "video_half": None, "video_source": None, "video_caption": f"video failed: {exc}"})
        sheets.append(m)
        print(f"  {m['file']}  t={m['t_label']}  team {m['team']}  shooter {m['player_id']}  ball tiles {m['ball_tiles']}/{N_FRAMES}  video {m['video_source']}")
    grab.close()
    clip_label = str(clip.relative_to(ROOT)) if clip.is_relative_to(ROOT) else str(clip)
    (args.out / "sheets.json").write_text(json.dumps({"clip": clip_label, "shots": sheets}, indent=1))
    page = write_index(args.out, sheets, clip_label, args.events, len(frames))
    print(f"{len(sheets)} sheets -> {args.out}, index {page}")
    return 0


if __name__ == "__main__":
    with qa_lock():
        raise SystemExit(main())
