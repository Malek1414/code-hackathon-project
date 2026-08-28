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
import re
import time
from pathlib import Path

import cv2
import numpy as np

from .clips import OVERLAY, render_clip
from .numbers_sheet import IDENTITIES, build_number_cards
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
        from datetime import datetime

        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone().strftime("%H:%M")
    except ValueError:
        return iso


def verdicts_file(out: Path, clip: str) -> Path:
    return out / f"verdicts_{Path(clip).stem}.json"


TEAM_LETTER = {0: "A", 1: "B"}


def key_number(key: str | None) -> int | None:
    """'A24' -> 24, 'B?805' -> None, 'A12~5' -> 12."""
    if not key:
        return None
    m = re.match(r"^[ABX](\d{1,2})(?:~|$)", key)
    return int(m.group(1)) if m else None


def shot_signature(n: int, shot: dict, tracks: Path) -> str:
    """What a sheet depends on: the shot's own fields plus the tracks and overlay versions.
    events.json is rewritten often without changing a shot, so its mtime is deliberately not part of it."""
    fields = {k: shot.get(k) for k in ("t", "frame", "player_id", "team", "made", "shooter_foot", "hoop_bbox", "shooter_confirmed", "player_key")}
    versions = [int(pth.stat().st_mtime) if pth.exists() else 0 for pth in (tracks, OVERLAY)]
    return json.dumps([n, fields, versions], sort_keys=True)


KNOWN = QA_DIR / "known_shots.json"  # snapshot of an earlier events version (its "shots" with "t"); shots not in it are "neu, bitte pruefen"
KNOWN_MATCH_S = 1.5


def mark_new(sheets: list[dict], known_path: Path = KNOWN) -> int:
    known = [float(m["t"]) for m in ((read_json(known_path) or {}).get("shots") or []) if m.get("t") is not None]
    n_new = 0
    for m in sheets:
        m["is_new"] = bool(known) and not any(abs(m["t"] - t) <= KNOWN_MATCH_S for t in known)
        n_new += m["is_new"]
    return n_new


def _shot_block(s: dict) -> str:
    n = s["n"]
    letter = TEAM_LETTER.get(s["team"])
    team_txt = f"Team {letter}" if letter else "Team unbekannt"
    call = "Treffer" if s["made"] else "Fehlwurf"
    shooter = f"Werfer Track {s['player_id']}" if s["player_id"] is not None else "Werfer unbekannt"
    if s.get("player_key"):
        shooter += f" ({s['player_key']})"
    title = f"Wurf {n}, {s['t_label']}, {team_txt}, {call}, {shooter}"
    flipped = "war ein Fehlwurf" if s["made"] else "war ein Treffer"
    number = s.get("number_prefill")
    num_val = f' value="{number}"' if number is not None else ""
    if s.get("video_half"):
        cap = s.get("video_caption_de", "")
        video = f"""
  <video controls muted loop autoplay playsinline preload="metadata" src="{s['video_half']}" data-normal="{s['video']}" data-half="{s['video_half']}"></video>
  <div class="speed">
    <button type="button" data-speed="half" class="on">halbe Geschwindigkeit</button>
    <button type="button" data-speed="normal">normal</button>
    <span class="cap">{html.escape(cap)}</span>
  </div>"""
    else:
        video = f"""
  <div class="cap">{html.escape(s.get('video_caption_de', 'kein Video'))}</div>"""
    badge = '<span class="badge">neu, bitte pruefen</span>' if s.get("is_new") else ""
    return f"""
<section class="shot{' new' if s.get('is_new') else ''}" id="shot-{n}" data-n="{n}">
  <h2>{html.escape(title)} {badge}</h2>{video}
  <div class="cap">Bildstreifen, 1,5 s vor bis 1,0 s nach dem Ereignis. Gelb = Ball, gruen = Korb, farbige Box = markierter Werfer, weisser Kreis = Standpunkt des Werfers.</div>
  <a href="{s['file']}" target="_blank"><img src="{s['file']}" alt="{html.escape(title)}" loading="lazy"></a>
  <div class="q">
    <div class="qt">1. War das ein Wurf, und stimmt {call}?</div>
    <div class="row">
      <label><input type="radio" name="s{n}" value="ok"> stimmt</label>
      <label><input type="radio" name="s{n}" value="flipped"> {flipped}</label>
      <label><input type="radio" name="s{n}" value="no_shot"> war kein Wurf</label>
    </div>
  </div>
  <div class="q">
    <div class="qt">2. Ist der markierte Werfer die richtige Person?</div>
    <div class="row">
      <label><input type="radio" name="h{n}" value="yes"> ja</label>
      <label><input type="radio" name="h{n}" value="no"> nein</label>
    </div>
  </div>
  <div class="q">
    <div class="qt">3. Welche Rueckennummer hat der Werfer?</div>
    <div class="row">
      <input type="number" class="num" name="num{n}" min="0" max="99" step="1" placeholder="Nr."{num_val}>
      <span class="toggle">
        <label><input type="radio" name="t{n}" value="0"{' checked' if s['team'] == 0 else ''}> Team A</label>
        <label><input type="radio" name="t{n}" value="1"{' checked' if s['team'] == 1 else ''}> Team B</label>
      </span>
      <input type="text" class="note" name="n{n}" placeholder="Notiz (optional)">
    </div>
  </div>
</section>"""


def _number_card(i: int, c: dict) -> str:
    detected = f"erkannt: {c['detected']}" if c.get("detected") is not None else "erkannt: keine"
    team = f"Team {c['team_letter']}" if c["team_letter"] in ("A", "B") else "Team unbekannt"
    tracks = ", ".join(str(t) for t in c["track_ids"][:6]) + (" ..." if len(c["track_ids"]) > 6 else "")
    span = f"{fmt_t(c['first_t'])} bis {fmt_t(c['last_t'])}" if c.get("first_t") is not None else ""
    img = f'<img src="{c["img"]}" alt="{html.escape(c["key"] or "")}" loading="lazy">' if c.get("img") else '<div class="noimg">keine Box gefunden</div>'
    return f"""
<div class="card" data-i="{i}">
  {img}
  <div class="cardmeta"><b>{html.escape(detected)}</b>, {team}, {html.escape(c['key'] or '')}<br><span class="dim">Tracks {tracks}, {c.get('frames') or 0} Frames, {span}</span></div>
  <div class="row">
    <input type="number" class="num" name="nn{i}" min="0" max="99" step="1" placeholder="richtige Nr.">
    <label><input type="checkbox" name="un{i}"> nicht lesbar</label>
  </div>
</div>"""


def _ball_section(out: Path) -> str:
    chk = read_json(out / "ball_check.json") or {}
    rec = read_json(out / "ball_recall.json") or {}
    parts = []
    if chk.get("frames"):
        parts.append(
            f"Ball-Box in {chk['ball_frames']} von {chk['frames']} Frames ({100 * chk['ball_share']:.0f} %), "
            f"{chk['rejects']} verworfene Kandidaten" + (" (noch keine Rejects-Datei von TRACK)" if not chk.get("rejects_file") else "")
            + f", Stand {chk.get('generated', '?')[11:16]}."
        )
    if rec.get("frames_total"):
        parts.append(f"Stichprobe ball_recall.jpg: {rec['ball_frames_total']} von {rec['frames_total']} Frames mit Ball-Box.")
    name = chk.get("video") if chk.get("video") and (out / chk["video"]).exists() else "ball_check.mp4"
    video = (
        f'<video controls muted playsinline preload="metadata" src="{name}"></video>'
        if (out / name).exists()
        else '<div class="empty">ball_check.mp4 wird noch gebaut.</div>'
    )
    return f"""
<div class="ball">
  <div class="cap">Worauf achten: jede gelbe Box, die auf einem Wandobjekt (Leuchten, Schilder, Zuschauer) sitzt, ist ein Fehler. Zaehle sie. Rote x sind Kandidaten, die das System selbst verworfen hat (S statisch, R Radius, H Kopf, G Gate).</div>
  {video}
  <div class="links"><a href="{name}" target="_blank">{name}</a><a href="ball_recall.jpg" target="_blank">ball_recall.jpg (40 Zufallsframes)</a></div>
  <div class="cap">{html.escape(' '.join(parts) if parts else 'Noch keine Ball-Auswertung.')}</div>
  <div class="row">
    <label>gelbe Boxen an der Wand gezaehlt: <input type="number" class="num" name="wall_hits" min="0" step="1" placeholder="Anzahl"></label>
    <input type="text" class="note" name="ball_note" placeholder="Notiz zum Ball (optional)">
  </div>
</div>"""


def write_index(out: Path, sheets: list[dict], numbers: list[dict], clip: str, events_path: Path, tracks_n: int, known_path: Path = KNOWN) -> Path:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    made = sum(1 for s in sheets if s["made"])
    stored = read_json(verdicts_file(out, clip)) or {}
    stored_line = (
        f"Gespeicherte Antworten aus {verdicts_file(out, clip).name} (Stand {_local(stored.get('reviewed'))}) sind vorbelegt."
        if stored.get("shots") or stored.get("numbers")
        else "Noch keine gespeicherten Antworten."
    )
    n_new = mark_new(sheets, known_path)
    ordered = sorted(sheets, key=lambda m: (not m.get("is_new"), m["t"]))
    if not sheets:
        shots_html = '<div class="empty">Noch keine Wuerfe in events.json.</div>'
    elif n_new:
        new_part = "".join(_shot_block(m) for m in ordered if m.get("is_new"))
        old_part = "".join(_shot_block(m) for m in ordered if not m.get("is_new"))
        shots_html = (
            f'<h3 class="sub">Neu, bitte pruefen ({n_new})</h3>{new_part}'
            f'<h3 class="sub">Bereits bewertet ({len(sheets) - n_new})</h3>{old_part}'
        )
    else:
        shots_html = "".join(_shot_block(m) for m in ordered)
    ident = read_json(IDENTITIES) or {}
    if numbers:
        cards_html = "".join(_number_card(i, c) for i, c in enumerate(numbers))
    elif ident.get("clip") and Path(ident["clip"]).name != Path(clip).name:
        cards_html = f'<div class="empty">identities.json gehoert noch zu {html.escape(Path(ident["clip"]).name)}, der Nummern-Check fuer {html.escape(Path(clip).name)} kommt, sobald NUMBERS nachzieht.</div>'
    else:
        cards_html = '<div class="empty">Noch keine identities.json.</div>'
    chk = read_json(out / "ball_check.json") or {}
    ball_video = {k: chk.get(k) for k in ("video", "generated", "ball_frames", "frames", "rejects", "rejects_file")}
    manifest = json.dumps({"clip": clip, "events": str(events_path), "generated": stamp, "tracks_frames": tracks_n, "shots": sheets, "numbers": numbers, "ball_video": ball_video})
    stored_js = json.dumps({"reviewed": stored.get("reviewed"), "uncalled": stored.get("uncalled_shots", 0), "shots": stored.get("shots", []), "numbers": stored.get("numbers", [])})
    page = PAGE.replace("{{BALL}}", _ball_section(out)).replace("{{TITLE}}", html.escape(Path(clip).name)).replace(
        "{{META}}",
        f"{len(sheets)} Wuerfe vom System erkannt ({made} Treffer, {len(sheets) - made} Fehlwuerfe), {tracks_n} verfolgte Frames, "
        f"{len(numbers)} Spieler im Nummern-Check, Stand {stamp}. {html.escape(stored_line)}"
        + (f" {n_new} Wuerfe sind neu gegenueber der letzten Bewertungsrunde und stehen oben." if n_new else ""),
    ).replace("{{SHOTS}}", shots_html).replace("{{CARDS}}", cards_html).replace("{{MANIFEST}}", manifest).replace("{{STORED}}", stored_js)
    path = out / "index.html"
    tmp = path.with_suffix(".tmp.html")
    tmp.write_text(page)
    tmp.replace(path)
    return path


PAGE = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wurf-Check</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 20px 24px 120px; font: 15px/1.45 -apple-system, system-ui, sans-serif; background: #161616; color: #ececec; }
  h1 { font-size: 20px; margin: 0 0 6px; }
  h2 { font-size: 16px; margin: 0 0 8px; font-weight: 600; }
  h3 { font-size: 18px; margin: 30px 0 6px; }
  .intro { font-size: 15px; color: #ddd; margin-bottom: 4px; }
  .meta { color: #9a9a9a; font-size: 13px; margin-bottom: 18px; }
  .shot { border-top: 1px solid #2c2c2c; padding: 16px 0 10px; }
  .shot video { display: block; width: 100%; max-width: 960px; height: auto; border-radius: 6px; background: #000; }
  .speed { display: flex; gap: 8px; align-items: center; margin: 8px 0 14px; flex-wrap: wrap; }
  .speed button { font: inherit; font-size: 13px; padding: 5px 12px; border-radius: 6px; border: 1px solid #444; background: #242424; color: #ddd; cursor: pointer; }
  .speed button.on { background: #3a3a3a; border-color: #777; color: #fff; }
  .cap { color: #9a9a9a; font-size: 13px; margin: 4px 0 6px; }
  .shot img { display: block; width: 100%; max-width: 1700px; height: auto; border-radius: 6px; }
  .q { margin-top: 12px; }
  .qt { font-weight: 600; margin-bottom: 4px; }
  .row { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
  .row label { cursor: pointer; user-select: none; padding: 6px 10px; border-radius: 6px; background: #242424; }
  .row label:has(input:checked) { background: #3a3a3a; outline: 1px solid #666; }
  .row input[type=radio], .row input[type=checkbox] { margin-right: 6px; }
  .toggle { display: inline-flex; gap: 4px; }
  .num { width: 90px; background: #1e1e1e; color: #ddd; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px 8px; font: inherit; }
  .note { flex: 1; min-width: 200px; max-width: 480px; background: #1e1e1e; color: #ddd; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px 8px; font: inherit; }
  .extra { border-top: 1px solid #2c2c2c; padding: 18px 0; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
  .extra input[type=number] { width: 70px; background: #1e1e1e; color: #ddd; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px 8px; font: inherit; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 14px; }
  .card { background: #1e1e1e; border: 1px solid #2c2c2c; border-radius: 8px; padding: 10px; }
  .card img { display: block; max-width: 100%; height: auto; border-radius: 4px; }
  .cardmeta { margin: 8px 0; font-size: 14px; }
  .dim { color: #9a9a9a; font-size: 13px; }
  .noimg { height: 120px; display: flex; align-items: center; justify-content: center; color: #777; border: 1px dashed #333; border-radius: 4px; }
  .bar { position: fixed; left: 0; right: 0; bottom: 0; background: #202020; border-top: 1px solid #383838; padding: 12px 24px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  .bar button { font: inherit; padding: 8px 16px; border-radius: 6px; border: 1px solid #555; background: #2c2c2c; color: #fff; cursor: pointer; }
  .bar button.primary { background: #ffd23f; color: #111; border-color: #ffd23f; font-weight: 600; }
  #summary { color: #bbb; }
  .empty { padding: 30px 0; color: #999; }
  h3.sub { font-size: 16px; color: #ffd23f; margin: 24px 0 4px; }
  .badge { display: inline-block; font-size: 12px; font-weight: 600; color: #111; background: #ffd23f; border-radius: 4px; padding: 2px 8px; margin-left: 8px; vertical-align: middle; }
  .shot.new { border-left: 3px solid #ffd23f; padding-left: 10px; }
  .badge.flip { background: #ff8a5c; }
  .ball video { display: block; width: 100%; max-width: 960px; height: auto; border-radius: 6px; background: #000; }
  .ball .links { display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0; }
  .ball a { color: #ffd23f; }
</style>
</head>
<body>
<h1>Wurf-Check: {{TITLE}}</h1>
<div class="intro">Diese Seite prueft, ob das System Wuerfe, Werfer und Rueckennummern richtig erkannt hat. Deine Antworten werden als JSON gespeichert und fuer Tests und Training genutzt.</div>
<div class="meta">{{META}}</div>
<h3>Ball-Check</h3>
{{BALL}}
<h3>Wuerfe</h3>
{{SHOTS}}
<div class="extra">
  <label>Wuerfe, die das System nicht erkannt hat (im Video gesehen, oben nicht aufgefuehrt): <input type="number" id="uncalled" min="0" step="1" value="0"></label>
</div>
<h3>Nummern pruefen</h3>
<div class="cap">Pro Spieler drei Ausschnitte aus dem Video. Trage die Nummer ein, die du auf dem Trikot siehst, oder markiere "nicht lesbar". Leer lassen heisst: die erkannte Nummer stimmt.</div>
<div class="cards">{{CARDS}}</div>
<div class="bar">
  <button class="primary" id="download">Antworten als JSON herunterladen</button>
  <button id="reset">Zuruecksetzen</button>
  <span id="summary"></span>
</div>
<script>
const MANIFEST = {{MANIFEST}};
const STORED = {{STORED}};
const KEY = "followcam-qa-verdicts:" + MANIFEST.clip;
const MATCH_S = 1.5;  // event times shift a little between track versions
const $ = (sel) => document.querySelector(sel);
function radio(name) { const r = $('input[name="' + name + '"]:checked'); return r ? r.value : null; }
function setRadio(name, value) { if (value === null || value === undefined) return; const r = $('input[name="' + name + '"][value="' + value + '"]'); if (r) r.checked = true; }
function numVal(name) { const el = $('input[name="' + name + '"]'); if (!el || el.value === "") return null; const v = Number(el.value); return Number.isFinite(v) ? v : null; }
function setVal(name, v) { const el = $('input[name="' + name + '"]'); if (el && v !== null && v !== undefined) el.value = v; }
function state() {
  const shots = MANIFEST.shots.map(s => {
    const h = radio("h" + s.n);
    const t = radio("t" + s.n);
    return { n: s.n, t: s.t, frame: s.frame, team: s.team, made: s.made, player_id: s.player_id, player_key: s.player_key || null,
      shot: radio("s" + s.n), shooter_ok: h === null ? null : h === "yes",
      number: numVal("num" + s.n), number_team: t === null ? null : Number(t),
      note: ($('input[name="n' + s.n + '"]') || {}).value || "" };
  });
  const ball = { wall_hits: numVal("wall_hits"), note: ($('input[name="ball_note"]') || {}).value || "", video: MANIFEST.ball_video || null };
  const numbers = MANIFEST.numbers.map((c, i) => ({ key: c.key, track_ids: c.track_ids, team: c.team, detected: c.detected,
    true_number: numVal("nn" + i), unreadable: !!($('input[name="un' + i + '"]') || {}).checked }));
  return { shots, numbers, ball, uncalled: Number($("#uncalled").value || 0), saved: new Date().toISOString() };
}
function counts(st) {
  const c = { shots_answered: 0, shots_open: 0, shot_ok: 0, shot_flipped: 0, shot_no_shot: 0, shooter_ok: 0, shooter_wrong: 0, numbers_answered: 0, numbers_confirmed: 0, numbers_corrected: 0, numbers_unreadable: 0 };
  for (const s of st.shots) {
    if (s.shot) { c.shots_answered++; c["shot_" + s.shot]++; } else c.shots_open++;
    if (s.shooter_ok === true) c.shooter_ok++; else if (s.shooter_ok === false) c.shooter_wrong++;
  }
  for (const n of st.numbers) {
    if (n.unreadable) { c.numbers_answered++; c.numbers_unreadable++; }
    else if (n.true_number !== null) { c.numbers_answered++; if (n.detected !== null && n.true_number === n.detected) c.numbers_confirmed++; else c.numbers_corrected++; }
  }
  return c;
}
function summary() {
  const st = state();
  const c = counts(st);
  $("#summary").textContent =
    "Wuerfe: " + c.shots_answered + " von " + st.shots.length + " beantwortet (" + c.shot_ok + " stimmt, " + c.shot_flipped + " vertauscht, " + c.shot_no_shot + " kein Wurf), " +
    "Werfer richtig " + c.shooter_ok + ", falsch " + c.shooter_wrong + ".  " +
    "Nummern: " + c.numbers_answered + " von " + st.numbers.length + " (" + c.numbers_confirmed + " bestaetigt, " + c.numbers_corrected + " korrigiert, " + c.numbers_unreadable + " nicht lesbar)" +
    (st.uncalled ? ".  " + st.uncalled + " nicht erkannte Wuerfe" : "") +
    (st.ball.wall_hits !== null ? ".  Ball an der Wand: " + st.ball.wall_hits : "");
  try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {}
}
function applyShot(s, v) {
  const flipped = typeof v.made === "boolean" && typeof s.made === "boolean" && v.made !== s.made;
  if (flipped) {  // the system call changed since this verdict: shooter, number and note carry over, the shot answer does not
    const h = document.querySelector('#shot-' + s.n + ' h2');
    if (h && !h.querySelector(".flip")) { const b = document.createElement("span"); b.className = "badge flip"; b.textContent = "Wertung geaendert, bitte neu pruefen"; h.appendChild(b); }
  }
  if (v.shot && !flipped) setRadio("s" + s.n, v.shot);
  else if (v.verdict) { // legacy correct / wrong / missed_shooter
    if (v.verdict === "correct") { setRadio("s" + s.n, "ok"); setRadio("h" + s.n, "yes"); }
    if (v.verdict === "missed_shooter") { setRadio("s" + s.n, "ok"); setRadio("h" + s.n, "no"); }
  }
  if (v.shooter_ok === true) setRadio("h" + s.n, "yes");
  if (v.shooter_ok === false) setRadio("h" + s.n, "no");
  if (v.number !== null && v.number !== undefined) setVal("num" + s.n, v.number);
  if (v.number_team !== null && v.number_team !== undefined) setRadio("t" + s.n, v.number_team);
  if (v.note) setVal("n" + s.n, v.note);
}
function apply(src) {
  if (!src) return false;
  let hit = 0;
  for (const s of MANIFEST.shots) {
    let best = null;
    for (const v of src.shots || []) {
      const d = Math.abs((v.t || 0) - s.t);
      if (d <= MATCH_S && (!best || d < Math.abs(best.t - s.t))) best = v;
    }
    if (best) { applyShot(s, best); hit++; }
  }
  MANIFEST.numbers.forEach((c, i) => {
    const v = (src.numbers || []).find(x => x.key === c.key) ||
      (src.numbers || []).find(x => (x.track_ids || []).some(t => c.track_ids.includes(t)));
    if (!v) return;
    if (v.true_number !== null && v.true_number !== undefined) setVal("nn" + i, v.true_number);
    const un = $('input[name="un' + i + '"]'); if (un) un.checked = !!v.unreadable;
    hit++;
  });
  if (src.uncalled) $("#uncalled").value = src.uncalled;
  if (src.ball) { setVal("wall_hits", src.ball.wall_hits); if (src.ball.note) setVal("ball_note", src.ball.note); hit++; }
  return hit > 0;
}
function restore() {
  let local = null;
  try { local = JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) {}
  const fileNewer = STORED.reviewed && (!local || !local.saved || STORED.reviewed > local.saved);
  if (fileNewer) { if (!apply(STORED)) apply(local); }
  else { if (!apply(local)) apply(STORED); }
}
document.addEventListener("change", summary);
document.addEventListener("input", summary);
$("#reset").onclick = () => {
  if (!confirm("Alle Antworten loeschen?")) return;
  document.querySelectorAll("input[type=radio], input[type=checkbox]").forEach(r => r.checked = false);
  document.querySelectorAll("input.note, input.num").forEach(n => n.value = "");
  $("#uncalled").value = 0;
  try { localStorage.removeItem(KEY); } catch (e) {}
  summary();
};
$("#download").onclick = () => {
  const st = state();
  const doc = { clip: MANIFEST.clip, events: MANIFEST.events, sheets_generated: MANIFEST.generated,
    reviewed: st.saved, counts: counts(st), uncalled_shots: st.uncalled, shots: st.shots, numbers: st.numbers, ball: st.ball };
  const blob = new Blob([JSON.stringify(doc, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "verdicts_" + MANIFEST.clip.split("/").pop().replace(/\.[^.]+$/, "") + ".json";
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};
document.querySelectorAll(".speed button").forEach(btn => {
  btn.onclick = () => {
    const wrap = btn.closest(".shot");
    const v = wrap.querySelector("video");
    const next = v.dataset[btn.dataset.speed];
    if (!next || v.getAttribute("src") === next) return;
    const ratio = v.duration ? v.currentTime / v.duration : 0;
    v.setAttribute("src", next);
    v.load();
    v.addEventListener("loadedmetadata", () => { v.currentTime = ratio * v.duration; v.play().catch(() => {}); }, { once: true });
    wrap.querySelectorAll(".speed button").forEach(b => b.classList.toggle("on", b === btn));
  };
});
if ("IntersectionObserver" in window) {
  const io = new IntersectionObserver(entries => {
    for (const e of entries) { const v = e.target; if (e.isIntersecting) v.play().catch(() => {}); else v.pause(); }
  }, { threshold: 0.2 });
  document.querySelectorAll(".shot video").forEach(v => io.observe(v));
}
restore();
summary();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", type=Path, default=EVENTS)
    ap.add_argument("--tracks", type=Path, default=TRACKS)
    ap.add_argument("--identities", type=Path, default=IDENTITIES)
    ap.add_argument("--clip", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=QA_DIR)
    ap.add_argument("--known", type=Path, default=None, help="events/sheets json of an earlier round; shots not in it are 'neu' (default out/qa/known_shots.json)")
    ap.add_argument("--only-new", action="store_true", help="render only the shots flagged new (quick first page)")
    ap.add_argument("--no-video", action="store_true", help="strips only, no clip cut (videos can be added by a later run)")
    ap.add_argument("--no-numbers", action="store_true", help="skip the number cards")
    args = ap.parse_args(argv)
    known_path = args.known or KNOWN

    events = read_json(args.events)
    meta = read_json(META) or {}
    frames = read_tracks(args.tracks)
    shots = (events or {}).get("shots", [])
    if events is None:
        print(f"{args.events} missing or unreadable, writing an empty index")
    clip = resolve_clip(str(args.clip) if args.clip else None, (events or {}).get("clip"), meta.get("clip"))
    if events and meta.get("clip") and Path(events.get("clip", "")).name != Path(meta["clip"]).name:
        # events.json (STATS) and tracks.jsonl (TRACK) belong to different clips while the pipeline
        # moves from one clip to the next: draw the sheets from events only, no boxes from foreign tracks
        print(f"events.json is {events.get('clip')} but tracks.jsonl is {meta['clip']}: sheets without track boxes")
        frames = []
    index = TrackIndex(frames)
    grab = FrameGrabber(clip)
    args.out.mkdir(parents=True, exist_ok=True)
    sheets = []  # files are replaced per shot; stale ones are pruned only after a complete run (an interrupted run must not empty the page)
    previous = {m.get("sig"): m for m in ((read_json(args.out / "sheets.json") or {}).get("shots") or []) if m.get("sig")}
    reused = 0
    known_t = [float(m["t"]) for m in ((read_json(known_path) or {}).get("shots") or []) if m.get("t") is not None]
    skipped = []
    for n, shot in enumerate(shots, 1):
        is_new = bool(known_t) and not any(abs(float(shot["t"]) - t) <= KNOWN_MATCH_S for t in known_t)
        if args.only_new and not is_new:
            skipped.append(n)
            continue
        sig = shot_signature(n, shot, args.tracks) + ("" if not args.no_video else "|novideo")
        prev = previous.get(sig)
        prev_files = [args.out / prev[k] for k in ("file", "video", "video_half") if prev.get(k)] if prev else []
        if prev and len(prev_files) == (1 if args.no_video else 3) and all(f.exists() for f in prev_files):
            sheets.append(prev)  # same shot, same tracks and overlay, files present: nothing to redo
            reused += 1
            continue
        img, m = render_shot(n, shot, grab, index)
        m["sig"] = sig
        m["file"] = f"shot_{n}_{'made' if m['made'] else 'miss'}.jpg"
        m["number_prefill"] = key_number(m.get("player_key"))
        save_jpg(args.out / m["file"], img)
        try:
            if args.no_video:
                raise RuntimeError("Video folgt im naechsten Durchlauf")
            m.update(render_clip(n, m["t"], clip, args.out))
        except RuntimeError as exc:
            print(f"  shot {n}: no video ({exc})")
            m.update({"video": None, "video_half": None, "video_source": None, "video_caption": f"video failed: {exc}"})
        m["video_caption_de"] = (
            "Video aus overlay.mp4 mit Boxen, 2,0 s vor bis 1,5 s nach dem Ereignis"
            if m.get("video_source") == "overlay"
            else f"Video aus dem Rohclip ohne Boxen ({m.get('video_reason', 'kein Overlay')}), 2,0 s vor bis 1,5 s nach dem Ereignis"
            if m.get("video_source") == "raw"
            else f"{m.get('video_caption')}"
        )
        sheets.append(m)
        print(f"  {m['file']}  t={m['t_label']}  team {m['team']}  shooter {m['player_id']}  ball tiles {m['ball_tiles']}/{N_FRAMES}  video {m['video_source']}")
    numbers = [] if args.no_numbers else build_number_cards(frames, grab, args.out, args.identities, clip)
    if not numbers and not args.no_numbers:
        for stale in args.out.glob("num_*.jpg"):
            stale.unlink()
    grab.close()
    if reused:
        print(f"  {reused} of {len(shots)} sheets reused (files newer than events, tracks and overlay)")
    if not args.only_new:
        keep = {m["file"] for m in sheets} | {m.get("video") for m in sheets} | {m.get("video_half") for m in sheets}
        for stale in list(args.out.glob("shot_*.jpg")) + list(args.out.glob("shot_*.mp4")):
            if stale.name not in keep:
                stale.unlink()
    if skipped:
        print(f"  {len(skipped)} known shots skipped (--only-new)")
    clip_label = str(clip.relative_to(ROOT)) if clip.is_relative_to(ROOT) else str(clip)
    (args.out / "sheets.json").write_text(json.dumps({"clip": clip_label, "shots": sheets, "numbers": numbers}, indent=1))
    page = write_index(args.out, sheets, numbers, clip_label, args.events, len(frames), known_path)
    print(f"{len(sheets)} sheets, {len(numbers)} number cards -> {args.out}, index {page}")
    return 0


if __name__ == "__main__":
    with qa_lock():
        raise SystemExit(main())
