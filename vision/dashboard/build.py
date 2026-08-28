"""Build out/dashboard.html: one self-contained page, no external requests.

    .venv/bin/python vision/dashboard/build.py [--events out/events.json] [--stats out/stats.json]
        [--calib out/court_calib.json] [--minimap minimap.mp4] [--overlay overlay.mp4] [--out out/dashboard.html]

Shot chart on a top-down court (shooter_foot projected to metres with the COURT
helper), per-player and per-team table from stats.json, minimap video embedded
by relative path (the html and the mp4 both live in out/). Missing inputs
degrade to an empty section with a note instead of failing, so the page can be
rebuilt at every milestone.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision.court.geometry import FIBA, polylines  # noqa: E402
from vision.court.project import Calibration, load_calibration  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEAM_NAME = {0: "Team A", 1: "Team B", -1: "Unknown"}
TEAM_COLOR = {0: "#3B82F6", 1: "#EF4444", -1: "#9CA3AF"}


def load_json(path: Path | None) -> dict | None:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return None


# --- court svg ---------------------------------------------------------------

SCALE = 30.0  # svg units per metre
MARGIN = 1.0


def court_svg(shots: list[dict], cal: Calibration | None) -> str:
    w = (FIBA.length_m + 2 * MARGIN) * SCALE
    h = (FIBA.width_m + 2 * MARGIN) * SCALE

    def P(x, y):
        return f"{(x + MARGIN) * SCALE:.1f},{h - (y + MARGIN) * SCALE:.1f}"

    parts = [f'<svg class="court" viewBox="0 0 {w:.0f} {h:.0f}" role="img" aria-label="Shot chart">']
    parts.append(f'<rect x="{MARGIN * SCALE}" y="{MARGIN * SCALE}" width="{FIBA.length_m * SCALE}" '
                 f'height="{FIBA.width_m * SCALE}" class="floor"/>')
    for poly in polylines(FIBA):
        parts.append('<polyline class="mark" points="' + " ".join(P(x, y) for x, y in poly) + '"/>')
    for hx, hy in FIBA.hoops:
        cx, cy = P(hx, hy).split(",")
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{0.225 * SCALE:.1f}" class="hoop"/>')

    placed = 0
    for s in shots:
        xy = s.get("court_m")
        if xy is None:
            continue
        cx, cy = P(xy[0], xy[1]).split(",")
        team = int(s.get("team", -1))
        made = bool(s.get("made"))
        label = (f"{'Made' if made else 'Miss'}, player {s.get('player_id')}, {TEAM_NAME.get(team, 'Unknown')}, "
                 f"t={s.get('t', 0):.1f}s" + (", unconfirmed" if s.get("unconfirmed") else ""))
        cls = f"shot team{team} {'made' if made else 'miss'}"
        if made:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="7" class="{cls}" data-team="{team}"><title>{html.escape(label)}</title></circle>')
        else:
            x, y = float(cx), float(cy)
            parts.append(f'<g class="{cls}" data-team="{team}"><title>{html.escape(label)}</title>'
                         f'<line x1="{x - 6:.1f}" y1="{y - 6:.1f}" x2="{x + 6:.1f}" y2="{y + 6:.1f}"/>'
                         f'<line x1="{x - 6:.1f}" y1="{y + 6:.1f}" x2="{x + 6:.1f}" y2="{y - 6:.1f}"/></g>')
        placed += 1
    parts.append("</svg>")
    return "\n".join(parts), placed


def place_shots(events: dict | None, cal: Calibration | None) -> list[dict]:
    if not events:
        return []
    shots = []
    for s in events.get("shots", []):
        s = dict(s)
        foot = s.get("shooter_foot")
        if s.get("court_m") is None and foot and cal is not None:
            xy = cal.project(int(s.get("frame", 0)) if s.get("frame") is not None else None, [foot])[0]
            s["court_m"] = [float(xy[0]), float(xy[1])] if np.isfinite(xy).all() else None
        if s.get("shooter_confirmed") is False:
            s["unconfirmed"] = True
        shots.append(s)
    return shots


# --- tables ------------------------------------------------------------------


def pct(v) -> str:
    return "" if v is None else f"{100 * float(v):.0f}%"


def num(v, digits=0, unit="") -> str:
    if v is None:
        return ""
    return f"{float(v):.{digits}f}{unit}"


def player_rows(stats: dict | None, distances: dict[int, float]) -> str:
    if not stats or not stats.get("players"):
        return '<tr><td colspan="8" class="muted">stats.json not available yet</td></tr>'
    rows = []
    players = sorted(stats["players"], key=lambda p: (int(p.get("team", -1)), -float(p.get("fga") or 0), int(p["id"])))
    for p in players:
        team = int(p.get("team", -1))
        fga, fgm = int(p.get("fga") or 0), int(p.get("fgm") or 0)
        fg = p.get("fg_pct") if p.get("fg_pct") is not None else (fgm / fga if fga else None)
        dist = p.get("distance_m") if p.get("distance_m") is not None else distances.get(int(p["id"]))
        rows.append(
            f'<tr data-team="{team}"><td><span class="dot" style="background:{TEAM_COLOR.get(team)}"></span>{TEAM_NAME.get(team)}</td>'
            f'<td class="num">{p["id"]}</td><td class="num">{fga}</td><td class="num">{fgm}</td>'
            f'<td class="num">{pct(fg)}</td><td class="num">{num(p.get("possession_s"), 0, " s")}</td>'
            f'<td class="num">{num(dist, 0, " m")}</td>'
            f'<td><div class="bar"><span style="width:{(100 * float(fg)) if fg else 0:.0f}%;background:{TEAM_COLOR.get(team)}"></span></div></td></tr>')
    return "\n".join(rows)


def team_rows(stats: dict | None, shots: list[dict]) -> str:
    teams = (stats or {}).get("teams")
    if not teams and shots:
        agg: dict[int, dict] = {}
        for s in shots:
            t = agg.setdefault(int(s.get("team", -1)), {"team": int(s.get("team", -1)), "fga": 0, "fgm": 0})
            t["fga"] += 1
            t["fgm"] += 1 if s.get("made") else 0
        teams = list(agg.values())
    if not teams:
        return '<tr><td colspan="4" class="muted">no team totals yet</td></tr>'
    out = []
    for t in sorted(teams, key=lambda t: int(t.get("team", -1))):
        team = int(t.get("team", -1))
        fga, fgm = int(t.get("fga") or 0), int(t.get("fgm") or 0)
        out.append(f'<tr><td><span class="dot" style="background:{TEAM_COLOR.get(team)}"></span>{TEAM_NAME.get(team)}</td>'
                   f'<td class="num">{fga}</td><td class="num">{fgm}</td><td class="num">{pct(fgm / fga) if fga else ""}</td></tr>')
    return "\n".join(out)


# --- page --------------------------------------------------------------------

CSS = """
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--text:#e6e8ee;--muted:#8b93a7;--floor:#1f232c;--mark:#cfd4dd;--a:#3B82F6;--b:#EF4444}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px 24px 60px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:20px}
h1{font-size:22px;margin:0;font-weight:650;letter-spacing:.01em}h2{font-size:15px;margin:0 0 12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.sub{color:var(--muted);font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:28px;font-weight:650;line-height:1.1}.kpi .l{color:var(--muted);font-size:13px;margin-top:4px}
.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:16px}@media(max-width:960px){.grid{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;min-width:0}
.toolbar{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap}
button{background:transparent;color:var(--text);border:1px solid var(--line);border-radius:999px;padding:5px 12px;font:inherit;font-size:13px;cursor:pointer}
button.on{background:#232838;border-color:#3a4256}
svg.court{width:100%;height:auto;display:block}.floor{fill:var(--floor)}.mark{fill:none;stroke:var(--mark);stroke-width:1.6}.hoop{fill:none;stroke:#f59e0b;stroke-width:2}
.shot.made{stroke:#0f1115;stroke-width:1.2}.shot.team0{fill:var(--a)}.shot.team1{fill:var(--b)}.shot.team-1{fill:#9CA3AF}
g.shot line{stroke-width:2.2}g.shot.team0 line{stroke:var(--a)}g.shot.team1 line{stroke:var(--b)}g.shot.team-1 line{stroke:#9CA3AF}
.hidden{display:none}
.legend{display:flex;gap:16px;color:var(--muted);font-size:13px;margin-top:8px;flex-wrap:wrap}.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:8px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.muted{color:var(--muted)}
.bar{width:90px;height:6px;background:#262b36;border-radius:3px;overflow:hidden}.bar span{display:block;height:100%}
.tablewrap{overflow-x:auto}
video{width:100%;height:auto;display:block;border-radius:8px;background:#000}
.videos{display:grid;grid-template-columns:1fr;gap:16px;margin-top:16px}.videos.two{grid-template-columns:1fr 1fr}@media(max-width:960px){.videos.two{grid-template-columns:1fr}}
footer{color:var(--muted);font-size:12px;margin-top:20px}
"""

JS = """
(function(){
  var buttons=document.querySelectorAll('[data-filter]');
  function apply(team){
    document.querySelectorAll('.shot,[data-team]').forEach(function(el){
      if(el.hasAttribute('data-filter'))return;
      var t=el.getAttribute('data-team');
      el.classList.toggle('hidden',team!=='all'&&t!==team);
    });
    buttons.forEach(function(b){b.classList.toggle('on',b.getAttribute('data-filter')===team)});
  }
  buttons.forEach(function(b){b.addEventListener('click',function(){apply(b.getAttribute('data-filter'))})});
  apply('all');
})();
"""


def build(events, stats, cal, minimap: str | None, overlay: str | None, clip: str, calib_note: str,
          distances: dict[int, float] | None = None) -> str:
    shots = place_shots(events, cal)
    svg, placed = court_svg(shots, cal)
    fga = len(shots)
    fgm = sum(1 for s in shots if s.get("made"))
    unconfirmed = sum(1 for s in shots if s.get("unconfirmed"))
    distances = distances or {}
    n_players = len((stats or {}).get("players") or [])
    fps = (events or {}).get("fps")

    kpis = [
        (str(fga), "Field goal attempts"),
        (str(fgm), "Made"),
        (pct(fgm / fga) if fga else "", "Team FG%"),
        (str(n_players), "Players tracked"),
    ]
    if unconfirmed:
        kpis.append((str(unconfirmed), "Unconfirmed shots"))
    kpi_html = "".join(f'<div class="kpi"><div class="v">{html.escape(v) or "&nbsp;"}</div><div class="l">{l}</div></div>' for v, l in kpis)

    notes = []
    if not events:
        notes.append("events.json not available yet, shot chart is empty.")
    elif placed < fga:
        notes.append(f"{fga - placed} shots without a court position (no calibration for their frame).")
    if cal is None:
        notes.append("court_calib.json missing, shots cannot be placed on the court.")
    else:
        notes.append(calib_note)
    note_html = " ".join(html.escape(n) for n in notes)

    videos = []
    if minimap:
        videos.append(f'<div><h2>Minimap</h2><video src="{html.escape(minimap)}" controls muted loop autoplay playsinline></video></div>')
    if overlay:
        videos.append(f'<div><h2>Tracking overlay</h2><video src="{html.escape(overlay)}" controls muted loop playsinline></video></div>')
    videos_html = f'<div class="videos{" two" if len(videos) == 2 else ""}">{"".join(videos)}</div>' if videos else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FollowCam Coach Dashboard</title><style>{CSS}</style></head>
<body><main>
<header><div><h1>FollowCam Coach Dashboard</h1><div class="sub">{html.escape(clip)}{f" at {fps:g} fps" if fps else ""}</div></div>
<div class="sub">BC Lions Moabit vs Weddinger Wiesel, Landesliga Berlin</div></header>
<div class="kpis">{kpi_html}</div>
<div class="grid">
<section><h2>Shot chart</h2>
<div class="toolbar"><button data-filter="all" class="on">All</button><button data-filter="0"><span class="dot" style="background:{TEAM_COLOR[0]}"></span>Team A</button><button data-filter="1"><span class="dot" style="background:{TEAM_COLOR[1]}"></span>Team B</button></div>
{svg}
<div class="legend"><span><svg width="14" height="14"><circle cx="7" cy="7" r="6" fill="{TEAM_COLOR[0]}"/></svg> made</span><span><svg width="14" height="14"><line x1="2" y1="2" x2="12" y2="12" stroke="{TEAM_COLOR[0]}" stroke-width="2"/><line x1="2" y1="12" x2="12" y2="2" stroke="{TEAM_COLOR[0]}" stroke-width="2"/></svg> miss</span><span>hover a shot for details</span></div>
<p class="muted" style="font-size:13px;margin:10px 0 0">{note_html}</p>
</section>
<section><h2>Teams</h2>
<div class="tablewrap"><table><thead><tr><th>Team</th><th class="num">FGA</th><th class="num">FGM</th><th class="num">FG%</th></tr></thead>
<tbody>{team_rows(stats, shots)}</tbody></table></div>
<h2 style="margin-top:18px">Players</h2>
<div class="tablewrap"><table><thead><tr><th>Team</th><th class="num">Id</th><th class="num">FGA</th><th class="num">FGM</th><th class="num">FG%</th><th class="num">Possession</th><th class="num">Distance</th><th></th></tr></thead>
<tbody>{player_rows(stats, distances)}</tbody></table></div>
</section>
</div>
{videos_html}
<footer>Generated by vision/dashboard/build.py. Player ids are tracker ids, teams by jersey colour.</footer>
</main><script>{JS}</script></body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=Path, default=ROOT / "out" / "events.json")
    ap.add_argument("--stats", type=Path, default=ROOT / "out" / "stats.json")
    ap.add_argument("--calib", type=Path, default=ROOT / "out" / "court_calib.json")
    ap.add_argument("--tracks", type=Path, default=ROOT / "out" / "tracks.jsonl", help="für distance_m, falls stats.json es nicht füllt")
    ap.add_argument("--minimap", default="minimap.mp4", help="relative to the html, empty to omit")
    ap.add_argument("--overlay", default="overlay.mp4", help="relative to the html, empty to omit")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "dashboard.html")
    args = ap.parse_args(argv)

    events, stats = load_json(args.events), load_json(args.stats)
    cal = load_calibration(args.calib) if args.calib.exists() else None
    calib_note = ""
    if cal is not None:
        calib_note = {"per_frame": "Court calibration per frame (camera tracked).",
                      "keyframes": f"Court calibration from {len(cal.keyframes)} keyframes, blended by time.",
                      "single": "Single court calibration for the whole clip."}[cal.mode]
    minimap = args.minimap if args.minimap and (args.out.parent / args.minimap).exists() else None
    overlay = args.overlay if args.overlay and (args.out.parent / args.overlay).exists() else None
    clip = (events or {}).get("clip") or (cal.meta.get("clip") if cal else "") or ""
    distances: dict[int, float] = {}
    needs_distance = any(p.get("distance_m") is None for p in (stats or {}).get("players") or [])
    if cal is not None and needs_distance and args.tracks.exists():
        distances = cal.player_distances(args.tracks)
    page = build(events, stats, cal, minimap, overlay, clip, calib_note, distances)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"gespeichert: {args.out} ({len(page) // 1024} kB, events={'ja' if events else 'nein'}, stats={'ja' if stats else 'nein'}, "
          f"calib={cal.mode if cal else 'nein'}, minimap={'ja' if minimap else 'nein'}, overlay={'ja' if overlay else 'nein'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
