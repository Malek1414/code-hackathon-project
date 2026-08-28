"""Big Ball Baller end-of-game summary data: efficiency table per player and team.

    .venv/bin/python broadcast/end_summary_data.py [--stats out/game10/stats.json] [--events out/game10/events.json]
        [--calib out/court_calib_game10.json] [--config broadcast/config.json] [--out broadcast/end_summary.json]

Points: 2 per made shot, 3 when the shooter's projected position is outside the
three-point line (needs the calibration; flagged "three_estimated" because depth
accuracy near the basket is about half a metre). Possession share = a player's
possession seconds over the team's total. Players are the identified keys
(number known) plus every unidentified track with a shot; the remaining
unidentified tracks of a team are folded into one "others" row so possession
and distance totals stay complete.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.court.geometry import FIBA  # noqa: E402
from vision.court.project import load_calibration  # noqa: E402

BRAND = "Big Ball Baller"
DEFAULT_TEAMS = {0: {"name": "Team A", "color": "#2f6fdb"}, 1: {"name": "Team B", "color": "#c8102e"}}
THREE_R = 6.75
CORNER_Y = 0.9


def load_teams(config: Path | None) -> dict[int, dict]:
    teams = {k: dict(v) for k, v in DEFAULT_TEAMS.items()}
    if config and config.exists():
        for t in json.loads(config.read_text()).get("teams", []):
            teams.setdefault(int(t["id"]), {}).update({k: v for k, v in t.items() if k in ("name", "color")})
    return teams


def is_three(xy_m) -> bool | None:
    """Outside the three-point line of the nearer basket, None if unknown."""
    if xy_m is None or not np.isfinite(xy_m).all():
        return None
    x, y = float(xy_m[0]), float(xy_m[1])
    hx, hy = min(FIBA.hoops, key=lambda h: abs(h[0] - x))
    if y < CORNER_Y or y > FIBA.width_m - CORNER_Y:
        return True  # corner: beyond the straight section
    return math.hypot(x - hx, y - hy) > THREE_R


def shot_points(events: dict | None, cal) -> tuple[dict[str, dict], list[dict]]:
    """Per player key: pts, threes; plus the shot list with points."""
    per_key: dict[str, dict] = {}
    shots_out = []
    for s in (events or {}).get("shots", []):
        key = s.get("player_key") or f"{'AB'[int(s.get('team', 0))] if s.get('team') in (0, 1) else '?'}?{s.get('player_id')}"
        frame = s.get("release_frame") if s.get("release_frame") is not None else s.get("frame")
        xy = None
        if cal is not None and s.get("shooter_foot") and frame is not None:
            p = cal.project(int(frame), [s["shooter_foot"]])[0]
            xy = p if np.isfinite(p).all() and cal.on_court(p, 1.5)[0] else None
        if s.get("points") is not None:  # STATS' rule is the single source of truth when present
            pts = int(s["points"])
            three = bool(s.get("three_estimated")) if s.get("three_estimated") is not None else (pts == 3)
        else:
            three = is_three(xy)
            pts = (3 if three else 2) if s.get("made") else 0
        row = per_key.setdefault(key, {"pts": 0, "threes": 0, "fga": 0, "fgm": 0})
        row["fga"] += 1
        row["fgm"] += 1 if s.get("made") else 0
        row["pts"] += pts
        row["threes"] += 1 if (three and s.get("made")) else 0
        shots_out.append({"t": s.get("t"), "frame": frame, "player_key": key, "team": s.get("team"), "made": bool(s.get("made")),
                          "points": pts, "three_estimated": three, "court_m": None if xy is None else [round(float(xy[0]), 2), round(float(xy[1]), 2)],
                          "shooter_confirmed": s.get("shooter_confirmed"), "made_confirmed": s.get("made_confirmed")})
    return per_key, shots_out


def build(stats: dict, events: dict | None, cal, teams: dict[int, dict], clip: str) -> dict:
    per_key, shots = shot_points(events, cal)
    team_rows = {t: {"id": t, "name": teams[t]["name"], "color": teams[t]["color"], "pts": 0, "fga": 0, "fgm": 0,
                     "possession_s": 0.0, "distance_m": 0.0, "players_identified": 0, "tracks": 0, "threes": 0} for t in (0, 1)}
    players = []
    others = {t: {"key": f"{'AB'[t]} others", "number": None, "team": t, "pts": 0, "fga": 0, "fgm": 0, "possession_s": 0.0, "distance_m": 0.0, "tracks": 0} for t in (0, 1)}
    for p in stats.get("players", []):
        t = int(p.get("team", -1))
        if t not in team_rows:
            continue
        key = p.get("key") or f"{'AB'[t]}?{p['id']}"
        sp = per_key.get(key, {})
        fga = int(p.get("fga") or sp.get("fga", 0))
        fgm = int(p.get("fgm") or sp.get("fgm", 0))
        pts = int(p["pts"]) if p.get("pts") is not None else int(sp.get("pts", 2 * fgm))
        poss = float(p.get("possession_s") or 0.0)
        dist = float(p.get("distance_m") or 0.0)
        tr = team_rows[t]
        tr["pts"] += pts; tr["fga"] += fga; tr["fgm"] += fgm; tr["possession_s"] += poss; tr["distance_m"] += dist
        tr["tracks"] += len(p.get("track_ids") or [1]); tr["threes"] += int(sp.get("threes", 0))
        identified = p.get("number") is not None
        if identified or fga > 0:
            tr["players_identified"] += 1 if identified else 0
            players.append({"key": key, "number": p.get("number"), "team": t, "pts": pts, "fga": fga, "fgm": fgm,
                            "fg_pct": round(fgm / fga, 3) if fga else None, "threes": int(sp.get("threes", 0)),
                            "possession_s": round(poss, 1), "distance_m": round(dist, 1), "identified": identified,
                            "tracks": len(p.get("track_ids") or [1])})
        else:
            o = others[t]
            o["possession_s"] += poss; o["distance_m"] += dist; o["tracks"] += len(p.get("track_ids") or [1])
    for t, o in others.items():
        if o["tracks"]:
            players.append({**o, "fg_pct": None, "threes": 0, "possession_s": round(o["possession_s"], 1),
                            "distance_m": round(o["distance_m"], 1), "identified": False})
    for pl in players:
        tot = team_rows[pl["team"]]["possession_s"]
        pl["possession_share"] = round(pl["possession_s"] / tot, 3) if tot else None
    for st in stats.get("teams", []):  # STATS' team totals win over the per-player sum
        t = int(st.get("team", -1))
        if t in team_rows:
            if st.get("score") is not None:
                team_rows[t]["pts"] = int(st["score"])
            if st.get("fga") is not None:
                team_rows[t]["fga"], team_rows[t]["fgm"] = int(st["fga"]), int(st.get("fgm") or 0)
    for tr in team_rows.values():
        tr["fg_pct"] = round(tr["fgm"] / tr["fga"], 3) if tr["fga"] else None
        tr["possession_s"] = round(tr["possession_s"], 1)
        tr["distance_m"] = round(tr["distance_m"], 1)
    total_poss = sum(tr["possession_s"] for tr in team_rows.values())
    for tr in team_rows.values():
        tr["possession_share"] = round(tr["possession_s"] / total_poss, 3) if total_poss else None
    players.sort(key=lambda p: (p["team"], -(p["pts"]), -(p["fga"]), -(p["possession_s"])))
    return {
        "brand": BRAND, "clip": clip,
        "teams": [team_rows[0], team_rows[1]],
        "players": players,
        "shots": shots,
        "columns": ["pts", "fga", "fgm", "fg_pct", "possession_share", "distance_m"],
        "notes": [
            "points come from STATS (events.json shots[].points, stats.json teams[].score / players[].pts); the local 2/3 rule is only a fallback when those fields are missing",
            "possession_share is the player's share of the team's possession seconds",
            "distance_m comes from stats.json (calibrated tracks, uncertain camera stretches excluded)",
            "unidentified tracks with a shot keep their track key (A?661); tracks without a shot are folded into the others row",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", type=Path, default=ROOT / "out" / "game10" / "stats.json")
    ap.add_argument("--events", type=Path, default=ROOT / "out" / "game10" / "events.json")
    ap.add_argument("--calib", type=Path, default=ROOT / "out" / "court_calib_game10.json")
    ap.add_argument("--config", type=Path, default=ROOT / "broadcast" / "config.json")
    ap.add_argument("--out", type=Path, default=ROOT / "broadcast" / "end_summary.json")
    args = ap.parse_args(argv)
    stats = json.loads(args.stats.read_text())
    events = json.loads(args.events.read_text()) if args.events.exists() else None
    cal = load_calibration(args.calib) if args.calib.exists() else None
    data = build(stats, events, cal, load_teams(args.config), (events or {}).get("clip", ""))
    args.out.write_text(json.dumps(data, indent=1, allow_nan=False))
    for tr in data["teams"]:
        print(f"{tr['name']}: {tr['pts']} pts, {tr['fgm']}/{tr['fga']} FG ({tr['fg_pct']}), possession {tr['possession_s']} s ({tr['possession_share']}), {tr['distance_m']} m")
    top = [p for p in data["players"] if p["fga"]][:8]
    for p in top:
        print(f"  {p['key']:>10} team {p['team']} pts {p['pts']} {p['fgm']}/{p['fga']} poss {p['possession_share']} dist {p['distance_m']}")
    print(f"gespeichert: {args.out} ({len(data['players'])} Spielerzeilen, {len(data['shots'])} Würfe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
