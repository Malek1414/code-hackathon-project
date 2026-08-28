"""Build out/dashboard.html: the coach-facing report, one self-contained page.

    .venv/bin/python vision/dashboard/build.py [--events out/events.json] [--stats out/stats.json]
        [--calib out/court_calib.json] [--tracks out/tracks.jsonl] [--minimap minimap.mp4]
        [--overlay overlay.mp4] [--out out/dashboard.html]

No external requests, no build step: CSS and JS are inlined, the videos are
referenced by relative path (html and mp4 both live in out/). Python collects
the numbers (score, timeline, shot positions in metres, player table), the
inline script draws them. Every input may be missing; the section then shows a
short note instead of failing, so the page can be rebuilt at every milestone.

Owned by the FRONTEND session. Court helpers come from vision/court (COURT).
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision.court.geometry import FIBA, polylines  # noqa: E402
from vision.court.project import Calibration, load_calibration  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Team ids follow the tracks.jsonl contract: 0 = blue jerseys, 1 = black jerseys
# with red panels, -1 = unknown. Which club wears which colour is not
# known to the pipeline, so the default names stay neutral; --team-a/--team-b
# set the club names once confirmed.
TEAMS = {
    0: {"id": 0, "name": "Team A (blue)", "short": "Team A", "color": "#4C8DFF"},
    1: {"id": 1, "name": "Team B (black/red)", "short": "Team B", "color": "#E5484D"},
    -1: {"id": -1, "name": "Unassigned", "short": "Unassigned", "color": "#8A93A6"},
}
TEAM_LETTER = {0: "A", 1: "B"}


def set_team_names(a: str | None, b: str | None) -> None:
    for t, name in ((0, a), (1, b)):
        if name:
            TEAMS[t]["name"] = name
            TEAMS[t]["short"] = name
POINTS_PER_MADE = 2  # events.json carries no two/three point split yet; a `points` key per shot wins if present

METHOD = [
    ("Auto-labeling", "Grounding DINO labels players, ball, hoop and referees on frames of the game, no hand labeling."),
    ("Detector", "YOLO11 fine-tuned on this Landesliga footage, run at full resolution so the ball survives."),
    ("Tracking", "ByteTrack keeps player ids across frames, teams are read from the jersey colour."),
    ("Court model", "A homography maps every foot position to the 2D court. Shots, distances and the minimap live there."),
]


def load_json(path: Path | None) -> dict | None:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return None


# --- identities ---------------------------------------------------------------


KEY_RE = re.compile(r"^([AB?])(\?|\d+)(?:~(\d+))?$")  # A12, A?7, A5~7 (variant of A5 built from track 7)


def parse_key(key) -> tuple[int | None, str | None]:
    """'A12' -> (12, None), 'A5~7' -> (5, '7'), 'A?7' -> (None, None)."""
    m = KEY_RE.match(str(key or ""))
    if not m or m.group(2) == "?":
        return None, None
    return int(m.group(2)), m.group(3)


class Identities:
    """Track id -> jersey number, from out/identities.json (NUMBERS). Rows that
    already carry `key`/`number` (STATS) win; this only fills the gaps."""

    def __init__(self, data: dict | None, clip: str | None = None):
        self.by_track: dict[int, tuple[str, int | None]] = {}
        self.clip = Path(str((data or {}).get("clip") or "")).stem
        if clip and self.clip and self.clip != Path(str(clip)).stem:
            print(f"identities.json is for {self.clip}, events are {Path(str(clip)).stem}, jersey numbers ignored", file=sys.stderr)
            data = None
        for pl in (data or {}).get("players") or []:
            for tid in pl.get("track_ids") or []:
                self.by_track[int(tid)] = (str(pl.get("key") or ""), pl.get("number"))
        for tid, info in ((data or {}).get("tracks") or {}).items():
            if int(tid) not in self.by_track and info.get("number") is not None:
                self.by_track[int(tid)] = (f"{TEAM_LETTER.get(int(info.get('team', -1)), '?')}{info['number']}", info["number"])

    def resolve(self, track_id, key=None, number=None) -> tuple[str, int | None, str | None]:
        """Returns (label, number, identity key). Label is '#12' for a known number,
        '#12 (track 7)' for a variant key, else 'track 7'."""
        variant = None
        if key:
            n, variant = parse_key(key)
            number = number if number is not None else n
        if number is None and track_id is not None and int(track_id) in self.by_track:
            key, n = self.by_track[int(track_id)]
            pn, variant = parse_key(key)
            number = n if n is not None else pn
        if number is not None:
            return f"#{int(number)}" + (f" (track {variant})" if variant else ""), int(number), (str(key) if key else None)
        return (f"track {track_id}" if track_id is not None else "shooter unknown"), None, (str(key) if key else None)


MIN_POSSESSION_S = 3.0  # a track without a shot needs this much ball time to make the table
TABLE_PER_TEAM = 10  # rows shown per team before "show more"


def is_active(r: dict) -> bool:
    return bool(r["fga"] or (r["possession_s"] or 0) >= MIN_POSSESSION_S)


def merge_by_identity(rows: list[dict]) -> list[dict]:
    """stats.json rows are per track id until STATS aggregates by key; fold rows
    that resolve to the same identity key into one player row."""
    merged: dict[str, dict] = {}
    out = []
    for r in rows:
        k = r.get("ident")
        if not k or "?" in k:
            out.append(r)
            continue
        if k not in merged:
            merged[k] = dict(r, track_ids=[r["id"]])
            out.append(merged[k])
            continue
        m = merged[k]
        m["track_ids"].append(r["id"])
        m["fga"] += r["fga"]
        m["fgm"] += r["fgm"]
        for f in ("possession_s", "distance_m"):
            if r[f] is not None:
                m[f] = round((m[f] or 0) + r[f], 1)
    for m in merged.values():
        m["fg_pct"] = round(m["fgm"] / m["fga"], 3) if m["fga"] else None
        m["active"] = is_active(m)
    return out


# --- data ---------------------------------------------------------------------


def shot_points(s: dict) -> int:
    return int(s.get("points") or POINTS_PER_MADE) if s.get("made") else 0


def place_shots(events: dict | None, cal: Calibration | None, ids: Identities) -> list[dict]:
    """Shots with `court_m` (metres, None without calibration) and `points`."""
    if not events:
        return []
    shots = []
    for raw in events.get("shots", []):
        s = {
            "t": float(raw.get("t") or 0.0),
            "frame": raw.get("frame"),
            "player_id": raw.get("player_id"),
            "team": int(raw.get("team", -1)) if raw.get("team") is not None else -1,
            "made": bool(raw.get("made")),
            "points": shot_points(raw),
            "unconfirmed": raw.get("shooter_confirmed") is False or raw.get("made_confirmed") is False,
            "flags": [f for f, ok in (("shooter unconfirmed", raw.get("shooter_confirmed")), ("basket unconfirmed", raw.get("made_confirmed"))) if ok is False],
            "court_m": raw.get("court_m"),
        }
        s["label"], s["number"], _ = ids.resolve(raw.get("player_id"), raw.get("player_key"))
        foot = raw.get("shooter_foot")
        if s["court_m"] is None and foot and cal is not None:
            frame = int(raw["frame"]) if raw.get("frame") is not None else None
            xy = cal.project(frame, [foot])[0]
            if np.isfinite(xy).all() and cal.on_court(xy)[0]:
                s["court_m"] = [round(float(xy[0]), 2), round(float(xy[1]), 2)]
        s["estimated"] = False
        s["no_shooter"] = not foot
        if s["court_m"] is None:
            est = estimate_court_m(foot, raw.get("hoop_bbox"), s["team"])
            if est is not None:
                s["court_m"], s["estimated"] = est, True
                s["flags"].append("position estimated from image")
        shots.append(s)
    shots.sort(key=lambda s: s["t"])
    return shots


RIM_DIAMETER_M = 0.45
RIM_HEIGHT_M = 3.05


def estimate_court_m(foot, hoop_bbox, team: int) -> list[float] | None:
    """Court position guessed from the image alone, for clips without calibration.

    Scale: the hoop box is one rim wide (0.45 m). The foot's horizontal offset
    from the hoop centre runs along the court length (the camera stands at the
    sideline), the vertical offset below the rim minus the rim height runs
    across the width towards the camera. Team 0 is drawn at the left basket,
    team 1 at the right one; the true attacking direction is unknown. Good for
    "how far from the basket", not for left/right of it."""
    if not foot or not hoop_bbox:
        return None
    x1, y1, x2, y2 = [float(v) for v in hoop_bbox[:4]]
    hw = x2 - x1
    if hw <= 2:
        return None
    m_per_px = RIM_DIAMETER_M / hw
    along = abs(float(foot[0]) - (x1 + x2) / 2) * m_per_px
    across = max(0.3, (float(foot[1]) - y2) * m_per_px - RIM_HEIGHT_M)
    hx, hy = FIBA.hoops[0] if team != 1 else FIBA.hoops[-1]
    x = hx + along if team != 1 else hx - along
    y = hy - across
    x = min(max(x, 0.2), FIBA.length_m - 0.2)
    y = min(max(y, 0.2), FIBA.width_m - 0.2)
    return [round(x, 2), round(y, 2)]


def score(shots: list[dict]) -> dict[int, int]:
    out = {0: 0, 1: 0}
    for s in shots:
        if s["team"] in out:
            out[s["team"]] += s["points"]
    return out


def team_totals(shots: list[dict], stats: dict | None) -> list[dict]:
    agg = {t: {"team": t, "fga": 0, "fgm": 0} for t in (0, 1)}
    for s in shots:
        if s["team"] in agg:
            agg[s["team"]]["fga"] += 1
            agg[s["team"]]["fgm"] += 1 if s["made"] else 0
    if not shots:
        for t in (stats or {}).get("teams") or []:
            if int(t.get("team", -1)) in agg:
                agg[int(t["team"])].update(fga=int(t.get("fga") or 0), fgm=int(t.get("fgm") or 0))
    return [agg[0], agg[1]]


def player_rows(stats: dict | None, distances: dict[int, float], ids: Identities) -> list[dict]:
    rows = []
    for p in (stats or {}).get("players") or []:
        fga, fgm = int(p.get("fga") or 0), int(p.get("fgm") or 0)
        track_id = p.get("id") if isinstance(p.get("id"), int) or str(p.get("id")).isdigit() else None
        label, number, ident = ids.resolve(track_id, p.get("key"), p.get("number"))
        fg = (fgm / fga) if fga else None
        dist = p.get("distance_m")
        if dist is None:
            tids = [int(t) for t in (p.get("track_ids") or ([track_id] if track_id is not None else [])) if str(t).lstrip("-").isdigit()]
            found = [distances[t] for t in tids if t in distances]
            dist = round(sum(found), 1) if found else None
        rows.append({
            "id": p.get("id"), "label": label, "number": number, "ident": ident, "team": int(p.get("team", -1)), "fga": fga, "fgm": fgm,
            "fg_pct": None if fg is None else round(float(fg), 3),
            "possession_s": None if p.get("possession_s") is None else round(float(p["possession_s"]), 1),
            "distance_m": None if dist is None else round(float(dist), 1),
        })
        r = rows[-1]
        r["active"] = is_active(r)
    if not any(p.get("key") for p in (stats or {}).get("players") or []) or not any(p.get("number") is not None for p in (stats or {}).get("players") or []):
        rows = merge_by_identity(rows)
    rows.sort(key=lambda r: (r["team"] if r["team"] >= 0 else 9, -r["fga"], -(r["possession_s"] or 0), r["number"] is None, r["number"] or 0, str(r["id"])))
    shown: dict[int, int] = {}
    for r in rows:
        if r["active"]:
            shown[r["team"]] = shown.get(r["team"], 0) + 1
            if shown[r["team"]] > TABLE_PER_TEAM and not r["fga"]:
                r["active"] = False
    return rows


def possessions(events: dict | None) -> list[dict]:
    out = []
    for p in (events or {}).get("possessions") or []:
        try:
            out.append({"team": int(p.get("team", -1)), "start": float(p["start_t"]), "end": float(p["end_t"]),
                        "player_id": p.get("player_id")})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def cuts_s(events: dict | None) -> list[float]:
    fps = float((events or {}).get("fps") or 0) or None
    out = []
    for c in (events or {}).get("cuts") or []:
        try:
            out.append(round(float(c) / fps, 2) if fps else float(c))
        except (TypeError, ValueError):
            continue
    return out


def clip_duration(events: dict | None, tracks_meta: dict | None, tracks: Path, shots: list[dict]) -> float:
    if events and events.get("duration_s"):
        return float(events["duration_s"])
    same_clip = (not events) or Path(str(events.get("clip") or "")).name == Path(str((tracks_meta or {}).get("clip") or "")).name
    if same_clip and tracks_meta and tracks_meta.get("source_fps") and tracks_meta.get("last_frame") is not None:
        return (float(tracks_meta["last_frame"]) - float(tracks_meta.get("first_frame") or 0)) / float(tracks_meta["source_fps"])
    last_t = 0.0
    if same_clip and tracks.exists():
        with tracks.open("rb") as fh:
            try:
                fh.seek(-4096, 2)
            except OSError:
                fh.seek(0)
            tail = fh.read().decode("utf-8", "ignore").strip().splitlines()
        for line in reversed(tail):
            try:
                last_t = float(json.loads(line)["t"])
                break
            except (ValueError, KeyError):
                continue
    if shots:
        last_t = max(last_t, shots[-1]["t"] + 5)
    return max(last_t, 10.0)


def court_geometry() -> dict:
    return {
        "length": FIBA.length_m, "width": FIBA.width_m,
        "lines": [np.asarray(p, float).round(3).tolist() for p in polylines(FIBA)],
        "hoops": [list(h) for h in FIBA.hoops],
    }


def video_duration_s(path: Path) -> float | None:
    """Duration via the bundled ffmpeg, None when it cannot be read."""
    try:
        import re as _re
        import subprocess
        import imageio_ffmpeg
        out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                             capture_output=True, text=True, timeout=20).stderr
        m = _re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", out)
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
    except Exception:  # noqa: BLE001, a probe failure must not take the page down
        return None


def same_length(a: Path, b: Path, tolerance: float = 0.15) -> bool | None:
    da, db = video_duration_s(a), video_duration_s(b)
    if da is None or db is None:
        return None
    return abs(da - db) <= tolerance * max(da, db, 1.0)


def fmt_clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


# --- page ---------------------------------------------------------------------

CSS = r"""
:root{--bg:#0B0D12;--panel:#12151C;--panel2:#171B24;--line:#222836;--text:#E7EAF0;--muted:#8A93A6;--faint:#7C8598;
--floor:#151923;--mark:#8E96A8;--rim:#F0A63A;--lions:#4C8DFF;--wiesel:#E5484D;--none:#8A93A6}
*{box-sizing:border-box}html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:1320px;margin:0 auto;padding:28px 28px 64px}
h2{font-size:12px;margin:0 0 14px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
.muted{color:var(--muted)}.faint{color:var(--faint)}.small{font-size:13px}
.num{text-align:right;font-variant-numeric:tabular-nums}
section{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 20px;min-width:0}
.stack{display:grid;gap:16px}
.grid2{display:grid;grid-template-columns:1.05fr 1fr;gap:16px}@media(max-width:1000px){.grid2{grid-template-columns:1fr}}
.videos{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.videos{grid-template-columns:1fr}}

header{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:20px;padding:22px 24px;background:var(--panel);border:1px solid var(--line);border-radius:14px}
@media(max-width:800px){header{grid-template-columns:1fr;text-align:center}.team.right{text-align:center}}
.kicker{grid-column:1/-1;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-bottom:6px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.team{display:flex;flex-direction:column;gap:6px}.team.right{text-align:right;align-items:flex-end}
.team .name{font-size:20px;font-weight:650;letter-spacing:.005em;display:flex;align-items:center;gap:10px}
.team.right .name{flex-direction:row-reverse}
.swatch{width:12px;height:12px;border-radius:3px;display:inline-block;flex:none}
.team .line{color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums}
.scoreboard{display:flex;align-items:baseline;gap:18px;font-variant-numeric:tabular-nums}
.scoreboard .pts{font-size:64px;font-weight:700;line-height:1;letter-spacing:-.02em}
.scoreboard .colon{font-size:40px;color:var(--faint);font-weight:300;transform:translateY(-6px)}
.scoreboard.empty .pts{color:var(--faint)}

.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:-4px 0 14px}
.toolbar .spacer{flex:1}
button.f{background:transparent;color:var(--text);border:1px solid var(--line);border-radius:999px;padding:5px 13px;font:inherit;font-size:13px;cursor:pointer;display:inline-flex;align-items:center;gap:8px}
button.f:hover{border-color:#39415A}button.f.on{background:var(--panel2);border-color:#414A64}
.legend{display:flex;gap:18px;color:var(--muted);font-size:13px;margin-top:10px;flex-wrap:wrap}
.legend span{display:inline-flex;align-items:center;gap:7px}

svg.chart{width:100%;height:auto;display:block;overflow:visible}
.tl-grid{stroke:var(--line);stroke-width:1}.tl-axis{fill:var(--muted);font-size:11px}
.tl-line{fill:none;stroke-width:2.2;stroke-linejoin:round}
.tl-dot{stroke:var(--panel);stroke-width:1.5;cursor:pointer}
.tl-miss{stroke-width:2;opacity:.75}
.tl-play{stroke:#E7EAF0;stroke-width:1.2;stroke-dasharray:3 3;opacity:0}.tl-cut{stroke:#2C3344;stroke-width:2;stroke-dasharray:2 4}
.tl-poss{opacity:.55}
.dim{opacity:.18}
svg.court{width:100%;height:auto;display:block}
.floor{fill:var(--floor)}.mark{fill:none;stroke:var(--mark);stroke-width:1.4;opacity:.75}.rim{fill:var(--rim);stroke:none}.board{stroke:var(--rim);stroke-width:3;stroke-linecap:round}
.shot{cursor:pointer;transition:opacity .15s}.shot.made{stroke:rgba(11,13,18,.9);stroke-width:1.5}.shot.miss{fill:var(--floor);stroke-width:2.6}.shot:hover{stroke-width:3;stroke:#fff}
.shot.unconfirmed{opacity:.75}.shot-est{fill:none;stroke-width:1.6;stroke-dasharray:3 3;pointer-events:none}
.hidden{display:none!important}
.pending{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.chip{border:1px solid var(--line);border-radius:8px;padding:4px 10px;font-size:13px;display:inline-flex;gap:8px;align-items:center;cursor:pointer;background:var(--panel2)}
.chip .m{width:9px;height:9px;border-radius:50%;display:inline-block}.chip .m.miss{background:transparent!important;border:2px solid}

.tablewrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:8px 6px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th:first-child,td:first-child{padding-left:2px}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.08em}
tbody tr:hover{background:var(--panel2)}tr.inactive{display:none}table.all tr.inactive{display:table-row}
td.id{font-weight:650;font-variant-numeric:tabular-nums}
.bar{width:52px;height:5px;background:var(--line);border-radius:3px;overflow:hidden;display:inline-block;vertical-align:middle;margin-left:10px}.bar span{display:block;height:100%}
.tdot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;vertical-align:1px}

video{width:100%;height:auto;display:block;border-radius:10px;background:#000;aspect-ratio:16/9}
.placeholder{aspect-ratio:16/9;border-radius:10px;border:1px dashed var(--line);display:flex;align-items:center;justify-content:center;color:var(--faint);font-size:13px;text-align:center;padding:20px}
.vcap{display:flex;justify-content:space-between;color:var(--muted);font-size:13px;margin-top:8px;gap:12px}

.method{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}@media(max-width:900px){.method{grid-template-columns:1fr 1fr}}@media(max-width:560px){.method{grid-template-columns:1fr}}
.step{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.step .n{color:var(--faint);font-size:12px;font-variant-numeric:tabular-nums;letter-spacing:.08em}
.step .t{font-weight:650;margin:4px 0 4px}.step .d{color:var(--muted);font-size:13.5px;line-height:1.45}
footer{color:var(--faint);font-size:12px;margin-top:22px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
#tip{position:fixed;pointer-events:none;background:#1C2130;border:1px solid #2E3648;color:var(--text);padding:7px 10px;border-radius:8px;font-size:13px;line-height:1.35;opacity:0;transition:opacity .08s;z-index:10;max-width:260px;box-shadow:0 8px 24px rgba(0,0,0,.4)}
#tip b{font-weight:650}#tip .sub{color:var(--muted)}
"""

JS = r"""
(function(){
  var D=window.DASH; var TEAM=D.teams; var team=function(t){return TEAM[String(t)]||TEAM['-1']};
  var SVG='http://www.w3.org/2000/svg';
  function el(tag,attrs,parent){var e=document.createElementNS(SVG,tag);for(var k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e}
  function clock(s){s=Math.max(0,Math.round(s));return Math.floor(s/60)+':'+('0'+(s%60)).slice(-2)}
  function pct(v){return v==null?'':Math.round(100*v)+'%'}
  var tip=document.getElementById('tip');
  function showTip(ev,htmlText){tip.innerHTML=htmlText;tip.style.opacity=1;moveTip(ev)}
  function moveTip(ev){var x=ev.clientX+14,y=ev.clientY+14;var r=tip.getBoundingClientRect();if(x+r.width>window.innerWidth-8)x=ev.clientX-r.width-14;if(y+r.height>window.innerHeight-8)y=ev.clientY-r.height-14;tip.style.left=x+'px';tip.style.top=y+'px'}
  function hideTip(){tip.style.opacity=0}
  function bindTip(node,textFn){node.addEventListener('mouseenter',function(e){showTip(e,textFn())});node.addEventListener('mousemove',moveTip);node.addEventListener('mouseleave',hideTip)}

  var overlay=document.getElementById('overlay'),minimap=document.getElementById('minimap');
  var canSeek=D.video_offset_s!=null&&overlay;
  function seekTo(t){if(!canSeek)return;var v=Math.max(0,t-D.video_offset_s-2.5);[overlay,minimap].forEach(function(m){if(!m)return;try{m.currentTime=v;m.play()}catch(e){}});overlay.scrollIntoView({behavior:'smooth',block:'nearest'})}
  function shotTip(s){var tm=team(s.team);return '<b>'+(s.made?'Made':'Missed')+'</b> by '+s.label+' <span class="sub">'+tm.short+'</span><br><span class="sub">at '+clock(s.t)+(s.flags&&s.flags.length?', '+s.flags.join(', '):'')+(canSeek?', click to watch':'')+'</span>'}

  /* score timeline */
  (function(){
    var svg=document.getElementById('timeline'); if(!svg)return;
    var W=1000,H=250,L=44,R=16,T=14,B=44,PH=10; var dur=Math.max(D.duration_s,1);
    var maxPts=Math.max(2,D.score['0'],D.score['1']); maxPts=Math.ceil(maxPts/2)*2;
    var x=function(t){return L+(t/dur)*(W-L-R)}, y=function(p){return T+(1-p/maxPts)*(H-T-B-PH-6)};
    svg.setAttribute('viewBox','0 0 '+W+' '+H);
    var stepPts=maxPts<=10?2:(maxPts<=30?5:10);
    for(var p=0;p<=maxPts;p+=stepPts){el('line',{x1:L,x2:W-R,y1:y(p),y2:y(p),'class':'tl-grid'},svg);var t=el('text',{x:L-8,y:y(p)+4,'class':'tl-axis','text-anchor':'end'},svg);t.textContent=p}
    var ticks=5;for(var i=0;i<=ticks;i++){var tt=dur*i/ticks;var lab=el('text',{x:x(tt),y:H-B+16,'class':'tl-axis','text-anchor':i==0?'start':(i==ticks?'end':'middle')},svg);lab.textContent=clock(tt)}
    var base=y(0);
    (D.possessions||[]).forEach(function(ps){if(ps.end<=ps.start)return;var rr=el('rect',{x:x(ps.start),y:base+10,width:Math.max(1.5,x(ps.end)-x(ps.start)),height:PH,rx:2,fill:team(ps.team).color,'class':'tl-poss','data-team':ps.team},svg);bindTip(rr,function(){return '<b>Possession</b> '+team(ps.team).short+(ps.player_id!=null?', player '+ps.player_id:'')+'<br><span class="sub">'+clock(ps.start)+' to '+clock(ps.end)+'</span>'})});
    if((D.possessions||[]).length){var pl=el('text',{x:L-8,y:base+10+PH-1,'class':'tl-axis','text-anchor':'end'},svg);pl.textContent='ball'}
    ['0','1'].forEach(function(tk){var tm=TEAM[tk];var pts=0,d='M'+x(0)+' '+y(0);var made=D.shots.filter(function(s){return String(s.team)===tk&&s.made});
      made.forEach(function(s){d+=' L'+x(s.t)+' '+y(pts);pts+=s.points;d+=' L'+x(s.t)+' '+y(pts)});d+=' L'+x(dur)+' '+y(pts);
      el('path',{d:d,'class':'tl-line','stroke':tm.color,'data-team':tk},svg);
      D.shots.filter(function(s){return String(s.team)===tk&&!s.made}).forEach(function(s){var m=el('line',{x1:x(s.t),x2:x(s.t),y1:base-9,y2:base-1,stroke:tm.color,'class':'tl-miss','data-team':tk},svg);bindTip(m,function(){return shotTip(s)});m.addEventListener('click',function(){seekTo(s.t)})});
      pts=0;made.forEach(function(s){pts+=s.points;var c=el('circle',{cx:x(s.t),cy:y(pts),r:5.5,fill:tm.color,'class':'tl-dot','data-team':tk},svg);bindTip(c,function(){return shotTip(s)});c.addEventListener('click',function(){seekTo(s.t)})});
    });
    ((D.cuts||[]).length<=20?D.cuts:[]).forEach(function(c){if(c<=0||c>=dur)return;var l=el('line',{x1:x(c),x2:x(c),y1:T,y2:base,'class':'tl-cut'},svg);bindTip(l,function(){return '<b>Camera cut</b><br><span class="sub">at '+clock(c)+'</span>'})});
    var play=el('line',{x1:x(0),x2:x(0),y1:T,y2:base,'class':'tl-play'},svg);
    if(canSeek){overlay.addEventListener('timeupdate',function(){var t=overlay.currentTime+D.video_offset_s;play.setAttribute('x1',x(Math.min(t,dur)));play.setAttribute('x2',x(Math.min(t,dur)));play.style.opacity=overlay.paused&&overlay.currentTime===0?0:1})}
  })();

  /* shot chart */
  (function(){
    var svg=document.getElementById('court'); if(!svg)return; var C=D.court,S=30,M=1.0;
    var W=(C.length+2*M)*S,H=(C.width+2*M)*S; svg.setAttribute('viewBox','0 0 '+W+' '+H);
    var X=function(m){return (m+M)*S}, Y=function(m){return H-(m+M)*S};
    el('rect',{x:M*S,y:M*S,width:C.length*S,height:C.width*S,rx:3,'class':'floor'},svg);
    C.lines.forEach(function(poly){el('polyline',{points:poly.map(function(p){return X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)}).join(' '),'class':'mark'},svg)});
    C.hoops.forEach(function(h){var bx=h[0]<C.length/2?h[0]-0.375:h[0]+0.375;el('line',{x1:X(bx),x2:X(bx),y1:Y(h[1]-0.9),y2:Y(h[1]+0.9),'class':'board'},svg);el('circle',{cx:X(h[0]),cy:Y(h[1]),r:(0.225*S).toFixed(1),'class':'rim'},svg)});
    D.shots.forEach(function(s){if(!s.court_m)return;var tm=team(s.team);
      if(s.estimated)el('circle',{cx:X(s.court_m[0]).toFixed(1),cy:Y(s.court_m[1]).toFixed(1),r:14,'class':'shot-est','stroke':tm.color,'data-team':s.team},svg);
      var c=el('circle',{cx:X(s.court_m[0]).toFixed(1),cy:Y(s.court_m[1]).toFixed(1),r:s.made?9.5:8.5,'class':'shot '+(s.made?'made':'miss')+(s.unconfirmed?' unconfirmed':''),'data-team':s.team},svg);
      if(s.made)c.setAttribute('fill',tm.color);else c.setAttribute('stroke',tm.color);
      bindTip(c,function(){return shotTip(s)});c.addEventListener('click',function(){seekTo(s.t)})});
    document.querySelectorAll('.chip').forEach(function(ch){var s=D.shots[+ch.getAttribute('data-i')];if(!s)return;bindTip(ch,function(){return shotTip(s)});ch.addEventListener('click',function(){seekTo(s.t)})});
  })();

  var tog=document.getElementById('toggle-inactive');
  if(tog){tog.addEventListener('click',function(){var tb=tog.parentNode.querySelector('table');var on=tb.classList.toggle('all');tog.textContent=(on?'Hide ':'Show ')+tog.getAttribute('data-n')+' more tracks'})}

  /* team filter, shared by chart, chips, table, timeline */
  var buttons=document.querySelectorAll('[data-filter]');
  function apply(f){document.querySelectorAll('[data-team]').forEach(function(n){var t=n.getAttribute('data-team');var off=f!=='all'&&t!==f;
      if(n.classList.contains('tl-line')||n.classList.contains('tl-dot')||n.classList.contains('tl-miss')||n.classList.contains('tl-poss'))n.classList.toggle('dim',off);else n.classList.toggle('hidden',off)});
    buttons.forEach(function(b){b.classList.toggle('on',b.getAttribute('data-filter')===f)});
    var vis=D.shots.filter(function(s){return f==='all'||String(s.team)===f});var made=vis.filter(function(s){return s.made}).length;
    var c=document.getElementById('chart-count');if(c)c.textContent=vis.length?(made+' of '+vis.length+' made'+(vis.length?', '+Math.round(100*made/vis.length)+'%':'')):'';}
  buttons.forEach(function(b){b.addEventListener('click',function(){apply(b.getAttribute('data-filter'))})});
  apply('all');
})();
"""


def build(*, events, stats, cal, shots, players, teams, poss, cuts, duration_s, minimap, overlay, clip, calib_note,
          video_offset_s, source_meta) -> str:
    sc = score(shots)
    has_events = bool(events)
    placed = sum(1 for s in shots if s.get("court_m"))
    has_distance = any(p["distance_m"] is not None for p in players)
    fps = (events or {}).get("fps") or (source_meta or {}).get("source_fps")
    T0, T1 = TEAMS[0], TEAMS[1]

    def team_line(t: dict) -> str:
        if not (t["fga"] or t["fgm"]):
            return "no shots detected yet" if not has_events else "no shots detected"
        return f'{t["fgm"]} of {t["fga"]} field goals, {100 * t["fgm"] / t["fga"]:.0f}%'

    # header -----------------------------------------------------------------
    header = f"""
<header>
  <h1 class="sr">{html.escape(T0['name'])} vs {html.escape(T1['name'])}, FollowCam coach report</h1>
  <div class="kicker"><span>Landesliga Berlin, {html.escape(clip or "clip pending")}{f", {fmt_clock(duration_s)} min" if duration_s else ""}{f", {fps:g} fps" if fps else ""}</span></div>
  <div class="team"><div class="name"><span class="swatch" style="background:{T0['color']}"></span>{T0['name']}</div><div class="line">{team_line(teams[0])}</div></div>
  <div class="scoreboard{'' if has_events else ' empty'}" aria-label="Final score"><span class="pts" style="color:{T0['color'] if has_events else 'inherit'}">{sc[0]}</span><span class="colon">:</span><span class="pts" style="color:{T1['color'] if has_events else 'inherit'}">{sc[1]}</span></div>
  <div class="team right"><div class="name"><span class="swatch" style="background:{T1['color']}"></span>{T1['name']}</div><div class="line">{team_line(teams[1])}</div></div>
</header>"""

    # timeline ----------------------------------------------------------------
    tl_note = "" if has_events else '<p class="muted small" style="margin:8px 0 0">Shot events not available yet, the timeline fills in once events.json exists.</p>'
    timeline = f"""
<section><h2>Score timeline</h2>
<svg id="timeline" class="chart" role="img" aria-label="Points over time for both teams"></svg>
<div class="legend"><span><i class="swatch" style="background:{T0['color']}"></i>{T0['short']}</span><span><i class="swatch" style="background:{T1['color']}"></i>{T1['short']}</span><span class="faint">dots are made shots, ticks are misses{', dotted lines are camera cuts' if cuts and len(cuts) <= 20 else ''}</span></div>
{tl_note}</section>"""

    # shot chart --------------------------------------------------------------
    chips = ""
    if shots and placed < len(shots):
        items = []
        for i, s in enumerate(shots):
            if s.get("court_m"):
                continue
            tm = TEAMS.get(s["team"], TEAMS[-1])
            items.append(f'<span class="chip" data-i="{i}" data-team="{s["team"]}"><i class="m {"made" if s["made"] else "miss"}" style="background:{tm["color"]};border-color:{tm["color"]}"></i>{fmt_clock(s["t"])}<span class="muted">{html.escape(s["label"])}</span></span>')
        n_noshooter = sum(1 for s in shots if not s.get("court_m") and s.get("no_shooter"))
        why = ("the shooter could not be identified" if n_noshooter == len(items)
               else "calibration pending" if not n_noshooter else f"{n_noshooter} without an identified shooter, the rest waits for calibration")
        chips = f'<p class="muted small" style="margin:14px 0 0">{len(items)} {"shot" if len(items) == 1 else "shots"} without a court position, {why}:</p><div class="pending">{"".join(items)}</div>'
    estimated = sum(1 for s in shots if s.get("estimated"))
    notes = []
    if estimated:
        notes.append(f"{estimated} of {len(shots)} shot positions estimated from the image (dashed ring): distance from the basket is roughly right, side is not. Team A is drawn at the left basket, Team B at the right one.")
    if not has_events:
        notes.append("Shot events not available yet.")
    elif not shots:
        notes.append("No shots detected in this clip.")
    if cal is None and shots and placed < len(shots):
        notes.append("Court calibration pending, shots are placed once court_calib.json exists.")
    elif cal is None and not shots and not has_events:
        notes.append("Shots appear on the court once events.json and court_calib.json exist.")
    elif calib_note:
        notes.append(calib_note)
    shot_chart = f"""
<section><h2>Shot chart</h2>
<div class="toolbar"><button class="f on" data-filter="all">Both teams</button><button class="f" data-filter="0"><i class="swatch" style="background:{T0['color']}"></i>{T0['short']}</button><button class="f" data-filter="1"><i class="swatch" style="background:{T1['color']}"></i>{T1['short']}</button><span class="spacer"></span><span id="chart-count" class="muted small"></span></div>
<svg id="court" class="court" role="img" aria-label="Shot chart on a top-down court"></svg>
<div class="legend"><span><svg width="14" height="14"><circle cx="7" cy="7" r="6" fill="{T0['color']}"/></svg>made</span><span><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="{T0['color']}" stroke-width="2"/></svg>missed</span>{f'<span><svg width="16" height="16"><circle cx="8" cy="8" r="6.5" fill="none" stroke="{T0["color"]}" stroke-width="1.5" stroke-dasharray="3 2"/></svg>position estimated from image, no calibration</span>' if estimated else ''}<span class="faint">hover a shot for time and player</span></div>
{chips}
{f'<p class="muted small" style="margin:12px 0 0">{" ".join(html.escape(n) for n in notes)}</p>' if notes else ''}
</section>"""

    # player table ------------------------------------------------------------
    rows = []
    for p in players:
        tm = TEAMS.get(p["team"], TEAMS[-1])
        fg = p["fg_pct"]
        rows.append(
            f'<tr data-team="{p["team"]}"{"" if p["active"] else " class=\"inactive\""}><td class="id{"" if p["number"] is not None else " muted"}">{html.escape(p["label"])}</td>'
            f'<td><i class="tdot" style="background:{tm["color"]}"></i>{tm["short"]}</td>'
            f'<td class="num">{p["fga"]}</td><td class="num">{p["fgm"]}</td>'
            f'<td class="num">{"" if fg is None else f"{100 * fg:.0f}%"}<span class="bar"><span style="width:{(100 * fg) if fg else 0:.0f}%;background:{tm["color"]}"></span></span></td>'
            f'<td class="num">{"" if p["possession_s"] is None else f"{p["possession_s"]:.0f} s"}</td>'
            + (f'<td class="num">{"" if p["distance_m"] is None else f"{p["distance_m"]:.0f} m"}</td>' if has_distance else "")
            + "</tr>")
    ncol = 7 if has_distance else 6
    n_inactive = sum(1 for p in players if not p["active"])
    if not rows:
        rows.append(f'<tr><td colspan="{ncol}" class="muted">Player stats not available yet, the table fills in once stats.json exists.</td></tr>')
    table = f"""
<section><h2>Players</h2>
<div class="tablewrap"><table><thead><tr><th>Player</th><th>Team</th><th class="num">FGA</th><th class="num">FGM</th><th class="num">FG%</th><th class="num">Possession</th>{'<th class="num">Distance</th>' if has_distance else ''}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
{f'<button class="f" id="toggle-inactive" style="margin-top:12px" data-n="{n_inactive}">Show {n_inactive} more tracks</button>' if n_inactive else ''}
<p class="faint small" style="margin:12px 0 0">{"Jersey numbers read from the video, tracker id where no number was read." if any(p["number"] is not None for p in players) else "Players are tracker ids until jersey numbers are read."} Team by jersey colour. Possession is time as the closest player to the ball.{' Distance counts only the seconds with a court calibration.' if has_distance and cal is not None and cal.mode != 'single' else '' if has_distance else ' Distance follows once the court calibration exists.'}</p>
</section>"""

    # videos ------------------------------------------------------------------
    ov = (f'<video id="overlay" src="{html.escape(overlay)}" controls muted playsinline preload="metadata"></video>'
          if overlay else '<div class="placeholder">Tracking overlay arrives with overlay.mp4</div>')
    meta_clip = Path(str((source_meta or {}).get("clip") or "")).name
    ov_note = "ids, teams and ball trail on the broadcast frame"
    if overlay and meta_clip and clip and meta_clip != clip:
        ov_note = f"{html.escape(meta_clip)}, the stats above are from {html.escape(clip)}"
    mm = (f'<video id="minimap" src="{html.escape(minimap)}" controls muted playsinline preload="metadata"></video>'
          if minimap else '<div class="placeholder">2D minimap for this clip is not rendered yet</div>')
    videos = f"""
<section><h2>Video</h2>
<div class="videos">
  <div>{ov}<div class="vcap"><span>Tracking overlay</span><span>{ov_note}</span></div></div>
  <div>{mm}<div class="vcap"><span>Minimap</span><span>the same seconds on the 2D court</span></div></div>
</div></section>"""

    # method ------------------------------------------------------------------
    steps = "".join(f'<div class="step"><div class="n">0{i + 1}</div><div class="t">{html.escape(t)}</div><div class="d">{html.escape(d)}</div></div>'
                    for i, (t, d) in enumerate(METHOD))
    method = f'<section><h2>How it was made</h2><div class="method">{steps}</div></section>'

    data = {
        "teams": {str(k): v for k, v in TEAMS.items()},
        "score": {str(k): v for k, v in sc.items()},
        "shots": shots, "possessions": poss, "players": players, "cuts": cuts,
        "duration_s": round(float(duration_s), 2),
        "court": court_geometry(),
        "video_offset_s": video_offset_s,
    }
    data_js = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    built = dt.datetime.now().strftime("%d.%m.%Y %H:%M")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FollowCam, {html.escape(T0['name'])} vs {html.escape(T1['name'])}</title><style>{CSS}</style></head>
<body><main class="stack">
{header}
{timeline}
<div class="grid2">
{shot_chart}
{table}
</div>
{videos}
{method}
<footer><span>Built {built} by vision/dashboard/build.py, everything on this page is computed from the video.</span><span>FollowCam, CODE Hackathon Berlin 2026</span></footer>
</main>
<div id="tip" role="tooltip"></div>
<script>window.DASH={data_js};</script>
<script>{JS}</script></body></html>
"""


def pick_calibration(default: Path, events: dict | None):
    """COURT writes one calibration per clip (out/court_calib_<clip>.json) and a
    contract copy out/court_calib.json that may belong to a different clip.
    Prefer the per-clip file for the events clip, fall back to the contract copy
    only when its clip matches (or the clip is unknown)."""
    clip = Path(str((events or {}).get("clip") or "")).stem
    candidates = ([default.parent / f"court_calib_{clip}.json"] if clip else []) + [default]
    for path in candidates:
        if not path.exists():
            continue
        try:
            cal = load_calibration(path)
        except Exception as exc:  # noqa: BLE001, a broken file must not take the page down
            print(f"calibration {path.name} unreadable: {exc}", file=sys.stderr)
            continue
        cal_clip = Path(str(cal.meta.get("clip") or "")).stem
        if not clip or not cal_clip or cal_clip == clip:
            return cal, path
    return None, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=Path, default=ROOT / "out" / "events.json")
    ap.add_argument("--stats", type=Path, default=ROOT / "out" / "stats.json")
    ap.add_argument("--calib", type=Path, default=ROOT / "out" / "court_calib.json")
    ap.add_argument("--tracks", type=Path, default=ROOT / "out" / "tracks.jsonl", help="for distance_m if stats.json lacks it, and the clip length")
    ap.add_argument("--tracks-meta", type=Path, default=ROOT / "out" / "tracks_meta.json", help="clip and frame range of the overlay, for video seeking")
    ap.add_argument("--minimap", default="minimap.mp4", help="relative to the html, empty to omit")
    ap.add_argument("--overlay", default="overlay.mp4", help="relative to the html, empty to omit")
    ap.add_argument("--identities", type=Path, default=ROOT / "out" / "identities.json", help="track id to jersey number (NUMBERS)")
    ap.add_argument("--team-a", default=None, help="club name for team 0 (blue jerseys), default stays neutral")
    ap.add_argument("--team-b", default=None, help="club name for team 1 (black/red jerseys), default stays neutral")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "dashboard.html")
    args = ap.parse_args(argv)
    set_team_names(args.team_a, args.team_b)

    events, stats, meta = load_json(args.events), load_json(args.stats), load_json(args.tracks_meta)
    ids = Identities(load_json(args.identities), (events or {}).get("clip"))
    cal, calib_path = pick_calibration(args.calib, events)
    calib_note = ""
    if cal is None and args.calib.exists():
        calib_note = "Court calibration on disk belongs to another clip, not used."
    if cal is not None:
        calib_note = {"per_frame": "Court calibration per frame, the camera pan is tracked.",
                      "keyframes": f"Court calibration from {len(cal.keyframes)} keyframes, blended by time.",
                      "single": "One court calibration for the whole clip."}[cal.mode]
    minimap = args.minimap if args.minimap and (args.out.parent / args.minimap).exists() else None
    overlay = args.overlay if args.overlay and (args.out.parent / args.overlay).exists() else None
    if minimap and overlay and same_length(args.out.parent / args.minimap, args.out.parent / args.overlay) is False:
        print(f"minimap {args.minimap} has a different length than {args.overlay}, treated as another clip and left out", file=sys.stderr)
        minimap = None
    clip = (events or {}).get("clip") or (meta or {}).get("clip") or (cal.meta.get("clip") if cal else "") or ""

    shots = place_shots(events, cal, ids)
    distances: dict[int, float] = {}
    needs_distance = any(p.get("distance_m") is None for p in (stats or {}).get("players") or [])
    if cal is not None and needs_distance and args.tracks.exists():
        distances = cal.player_distances(args.tracks)
    players = player_rows(stats, distances, ids)
    teams = team_totals(shots, stats)
    poss = possessions(events)
    duration_s = clip_duration(events, meta, args.tracks, shots)

    # The overlay video starts at meta.first_frame of meta.clip. Seeking from a shot
    # only makes sense when the events were computed on the same clip.
    video_offset_s = None
    if overlay and meta and meta.get("source_fps"):
        same_clip = (not events) or Path(str(events.get("clip") or "")).name == Path(str(meta.get("clip") or "")).name
        if same_clip:
            video_offset_s = round(float(meta.get("first_frame") or 0) / float(meta["source_fps"]), 3)

    page = build(events=events, stats=stats, cal=cal, shots=shots, players=players, teams=teams, poss=poss, cuts=cuts_s(events),
                 duration_s=duration_s, minimap=minimap, overlay=overlay, clip=Path(clip).name if clip else "",
                 calib_note=calib_note, video_offset_s=video_offset_s, source_meta=meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"saved: {args.out} ({len(page) // 1024} kB, events={'yes' if events else 'no'} shots={len(shots)} placed={sum(1 for s in shots if s.get('court_m'))}, "
          f"stats={'yes' if stats else 'no'} players={len(players)} active={sum(1 for p in players if p['active'])} numbered={sum(1 for p in players if p['number'] is not None)}, calib={cal.mode + ' ' + calib_path.name if cal else 'no'}, "
          f"minimap={'yes' if minimap else 'no'}, overlay={'yes' if overlay else 'no'}, seek_offset={video_offset_s})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
