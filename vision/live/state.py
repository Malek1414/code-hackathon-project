"""Big Ball Baller live state: out/live_state.json every second, also served
at 127.0.0.1:8501/state.json (schema: docs/ORCHESTRATION.md, Broadcast package).

Team names and colors come from --team-a/--team-b/--color-a/--color-b or
broadcast/config.json (FRONTEND's start menu); the fallback is the neutral
"Team A"/"Team B" with the overlay colors, never a guessed club name."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

BRAND = "Big Ball Baller"
DEFAULT_TEAMS = [{"id": 0, "name": "Team A", "color": "#2f6fdb"}, {"id": 1, "name": "Team B", "color": "#c8102e"}]


def load_team_config(path: str | Path = "broadcast/config.json", *, team_a=None, team_b=None, color_a=None, color_b=None) -> list[dict]:
    """Two team dicts {id, name, color}. CLI flags win over the menu json."""
    teams = [dict(t) for t in DEFAULT_TEAMS]
    p = Path(path)
    if p.exists():
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            d = {}
        lst = d.get("teams")
        if isinstance(lst, list):
            for i, t in enumerate(lst[:2]):
                if isinstance(t, dict):
                    teams[i]["name"] = str(t.get("name", teams[i]["name"]))
                    teams[i]["color"] = str(t.get("color", teams[i]["color"]))
        for i, key in enumerate(("team_a", "team_b")):
            t = d.get(key)
            if isinstance(t, dict):
                teams[i]["name"] = str(t.get("name", teams[i]["name"]))
                teams[i]["color"] = str(t.get("color", teams[i]["color"]))
            elif isinstance(t, str):
                teams[i]["name"] = t
        for i, key in enumerate(("color_a", "color_b")):
            if isinstance(d.get(key), str):
                teams[i]["color"] = d[key]
    for i, (name, color) in enumerate(((team_a, color_a), (team_b, color_b))):
        if name:
            teams[i]["name"] = name
        if color:
            teams[i]["color"] = color
    return teams


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) != 6:
        return (200, 200, 200)
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (b, g, r)


class LiveStats:
    """Per-player and per-team aggregates from the engine's shots, the score
    board's actions and the possession segments; keyed like stats.json."""

    def __init__(self, numbers: dict[int, str] | None = None) -> None:
        self.numbers = numbers or {}

    def key_of(self, pid: int | None, team: int) -> str | None:
        if pid is None:
            return None
        letter = {0: "A", 1: "B"}.get(team, "?")
        num = self.numbers.get(int(pid))
        return f"{letter}{num}" if num else f"{letter}?{pid}"

    def players(self, engine, board, distances: dict[int, float] | None = None) -> list[dict]:
        rows: dict[str, dict] = {}
        for s in engine.shots:
            if s.player_id is None:
                continue
            key = self.key_of(s.player_id, s.team)
            row = rows.setdefault(key, {"key": key, "number": self.numbers.get(int(s.player_id)), "team": s.team,
                                        "pts": 0, "fga": 0, "fgm": 0, "fg_pct": None, "possession_s": 0.0,
                                        "distance_m": None, "track_ids": []})
            row["fga"] += 1
            if s.made and s.made_confirmed:
                row["fgm"] += 1
                row["pts"] += 2
            if s.player_id not in row["track_ids"]:
                row["track_ids"].append(s.player_id)
        for seg in engine.possession.segments:
            key = self.key_of(seg.player_id, seg.team)
            row = rows.setdefault(key, {"key": key, "number": self.numbers.get(int(seg.player_id)), "team": seg.team,
                                        "pts": 0, "fga": 0, "fgm": 0, "fg_pct": None, "possession_s": 0.0,
                                        "distance_m": None, "track_ids": [seg.player_id]})
            row["possession_s"] += seg.duration_s(engine.possession.dt)
        for row in rows.values():
            row["number"] = int(row["number"]) if row["number"] and str(row["number"]).isdigit() else row["number"]
            row["fg_pct"] = round(row["fgm"] / row["fga"], 3) if row["fga"] else None
            row["possession_s"] = round(row["possession_s"], 1)
            if distances:
                d = sum(distances.get(t, 0.0) for t in row["track_ids"])
                row["distance_m"] = round(d) if d else None
        return sorted(rows.values(), key=lambda r: (-r["pts"], -r["fga"], r["key"]))

    def teams(self, engine, board, team_cfg: list[dict]) -> list[dict]:
        poss = Counter(seg.team for seg in engine.possession.segments)
        out = []
        for t in team_cfg:
            ts = board.teams[t["id"]]
            out.append({"id": t["id"], "name": t["name"], "color": t["color"], "score": ts.points, "fga": ts.fga,
                        "fgm": ts.fgm, "fg_pct": round(ts.fgm / ts.fga, 3) if ts.fga else None,
                        "possessions": poss.get(t["id"], 0)})
        return out


def build_state(*, source: str, t: float, teams: list[dict], players: list[dict], last_event: dict | None,
                pan_deg: float | None, camera_ok: bool, period: int = 1) -> dict:
    return {
        "schema": 1,
        "brand": BRAND,
        "clip_or_source": source,
        "t": round(t, 1),
        "period": period,
        "clock": f"{int(t // 60):02d}:{int(t % 60):02d}",
        "teams": teams,
        "players": players,
        "last_event": last_event,
        "pan_deg": None if pan_deg is None else round(pan_deg),
        "camera": "ok" if camera_ok else "no-frame",
    }


def write_state(state: dict, path: str | Path = "out/live_state.json") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(p)
