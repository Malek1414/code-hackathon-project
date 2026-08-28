"""Build `out/events.json` and `out/stats.json` from `out/tracks.jsonl`.

    .venv/bin/python -m vision.stats.build --tracks out/tracks.jsonl --clip data/clips/game10.mp4
    .venv/bin/python -m vision.stats.build --fixture made      # synthetic smoke run

Output follows the contract in docs/ORCHESTRATION.md (fg_pct is null when fga is 0). Extras that readers may
ignore: `possessions` in events.json (for overlays), `shooter_confirmed` and
`release_frame` per shot. `distance_m` is filled when a court calibration
(`out/court_calib.json`, key `H_px_to_m`, optional per-keyframe `frames`) is
available, otherwise null.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .court import distances_m
from .engine import StatsEngine
from .io import Frame, infer_fps, median_dt, read_tracks, synthetic_scenario
from .possession import PossessionParams, PossessionResult, possession_seconds
from .shots import ShotEvent, ShotParams

MIN_SEEN_S = 2.0  # players seen shorter than this (and without a shot) are tracker fragments


def build(
    frames: list[Frame],
    *,
    fps: float,
    clip: str,
    calib: dict | None = None,
    distances: dict[int, float] | None = None,
    cuts: list[int] | None = None,
    identities: dict[int, "Identity"] | None = None,
    image_height: float = 1080.0,
    on_court_filter: bool = True,
    bench_line_frac: float = 0.0,
    possession_params: PossessionParams = PossessionParams(),
    shot_params: ShotParams = ShotParams(),
) -> tuple[dict, dict]:
    """`distances` (player id -> metres) wins over `calib` (own projection).
    `cuts` = frame numbers where the footage jumps (engine state is reset).
    `identities` = track id -> real player (NUMBERS); stats are per player key.
    Bench players / spectators are removed per frame (OnCourtFilter)."""
    engine = StatsEngine(
        dt=median_dt(frames), possession_params=possession_params, shot_params=shot_params, cuts=cuts,
        calib=calib, image_height=image_height, on_court_filter=on_court_filter, bench_line_frac=bench_line_frac,
    )
    for fr in frames:
        engine.push(fr)
    engine.finish()
    possession = engine.possession.result()
    shots = engine.shots
    ident = identities or {}
    for s in shots:  # NUMBERS knows the team of a track even when TRACK's colour rule said -1
        if s.player_id in ident and ident[s.player_id].team >= 0:
            s.team = ident[s.player_id].team
            s.team_source = "identity"

    def key_of(pid: int | None, team: int) -> str | None:
        if pid is None:
            return None
        if pid in ident:
            return ident[pid].key
        return f"{TEAM_LETTER.get(team, '?')}?{pid}"

    events = {
        "fps": fps,
        "clip": clip,
        "cuts": sorted(set(int(c) for c in (cuts or []))),
        "off_court_track_ids": sorted(engine.removed_ids),
        "shots": [{**s.to_dict(), "player_key": key_of(s.player_id, s.team)} for s in shots],
        "possessions": [
            {
                "player_id": s.player_id,
                "player_key": key_of(s.player_id, s.team),
                "team": s.team,
                "start_t": round(s.start_t, 3),
                "end_t": round(s.end_t, 3),
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
            }
            for s in possession.segments
        ],
    }
    stats = player_stats(frames, possession, shots, calib=calib, distances=distances, identities=ident)
    return events, stats


TEAM_LETTER = {0: "A", 1: "B"}


class Identity:
    """One real player from out/identities.json (NUMBERS)."""

    __slots__ = ("key", "team", "number")

    def __init__(self, key: str, team: int, number: int | None) -> None:
        self.key, self.team, self.number = key, team, number


def load_identities(path: Path) -> dict[int, Identity]:
    """track id -> Identity, from the `players` list (preferred) or `tracks`."""
    d = json.loads(path.read_text())
    out: dict[int, Identity] = {}
    for pl in d.get("players") or []:
        ident = Identity(str(pl["key"]), int(pl.get("team", -1)), pl.get("number"))
        for tid in pl.get("track_ids") or []:
            out[int(tid)] = ident
    for tid, tr in (d.get("tracks") or {}).items():
        tid = int(tid)
        if tid in out:
            continue
        team = int(tr.get("team", -1))
        number = tr.get("number")
        letter = TEAM_LETTER.get(team, "?")
        key = f"{letter}{number}" if number is not None else f"{letter}?{tid}"
        out[tid] = Identity(key, team, number)
    return out


def identities_match_tracks(ident: dict[int, Identity], frames: list[Frame], min_agree: float = 0.7) -> bool:
    """identities.json belongs to a previous tracker run if its track ids do
    not exist in these tracks or their teams disagree with the tracks'
    majority colours (ids are re-numbered on every run)."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for fr in frames:
        for p in fr.players:
            if p.team >= 0:
                votes[p.id][p.team] += 1
    known = [tid for tid, it in ident.items() if it.team >= 0]
    if not known:
        return True
    present = [tid for tid in known if tid in votes]
    if len(present) < 0.5 * len(known):
        return False
    agree = sum(1 for tid in present if votes[tid].most_common(1)[0][0] == ident[tid].team)
    return agree >= min_agree * len(present)


def find_cuts(clip: str, out_dir: Path) -> list[int]:
    """COURT's cut list for this clip, if any: out/cuts_<stem>.json (list of
    frames, or {"cuts": [...]} / {"frames": [...]} with ints or {"frame": n}),
    or out/court_cuts_<stem>.txt with one frame number per line."""
    stem = Path(clip).stem
    for name in (f"cuts_{stem}.json", f"court_cuts_{stem}.json", f"cuts_{stem}.txt", f"court_cuts_{stem}.txt"):
        path = out_dir / name
        if path.exists():
            return load_cuts(path)
    return []


def load_cuts(path: Path) -> list[int]:
    text = path.read_text().strip()
    if not text:
        return []
    frames: list[int] = []
    if path.suffix == ".json":
        d = json.loads(text)
        items = d if isinstance(d, list) else (d.get("cuts") or d.get("frames") or [])
        for it in items:
            frames.append(int(it["frame"]) if isinstance(it, dict) else int(it))
    else:
        for line in text.splitlines():
            tok = line.strip().split()
            if tok and tok[0].lstrip("-").isdigit():
                frames.append(int(tok[0]))
    return sorted(set(frames))


def player_stats(
    frames: list[Frame],
    possession: PossessionResult,
    shots: list[ShotEvent],
    *,
    calib: dict | None = None,
    distances: dict[int, float] | None = None,
    identities: dict[int, Identity] | None = None,
) -> dict:
    """One row per real player when identities are known (track ids merged
    under their `key`), else one row per track id with key `A?<id>`."""
    dt = median_dt(frames)
    ident = identities or {}
    seen: Counter[int] = Counter()
    teams: dict[int, Counter] = defaultdict(Counter)
    for fr in frames:
        for p in fr.players:
            seen[p.id] += 1
            if p.team >= 0:
                teams[p.id][p.team] += 1

    fga: Counter[int] = Counter()
    fgm: Counter[int] = Counter()
    for s in shots:
        if s.player_id is not None:
            fga[s.player_id] += 1
            fgm[s.player_id] += int(s.made)
    poss_s = possession_seconds(frames, possession)
    distance = distances if distances is not None else (distances_m(frames, calib) if calib else {})

    rows: dict[str, dict] = {}
    for pid, n in seen.items():
        track_team = teams[pid].most_common(1)[0][0] if teams[pid] else -1
        if pid in ident:
            key, team, number = ident[pid].key, ident[pid].team, ident[pid].number
        else:
            key, team, number = f"{TEAM_LETTER.get(track_team, '?')}?{pid}", track_team, None
        row = rows.setdefault(
            key,
            {"id": pid, "key": key, "number": number, "team": team, "track_ids": [], "fga": 0, "fgm": 0,
             "fg_pct": None, "possession_s": 0.0, "distance_m": None, "_seen": 0},
        )
        row["id"] = min(row["id"], pid)
        row["track_ids"].append(pid)
        row["fga"] += fga[pid]
        row["fgm"] += fgm[pid]
        row["possession_s"] += poss_s.get(pid, 0.0)
        row["_seen"] += n
        if pid in distance:
            row["distance_m"] = (row["distance_m"] or 0.0) + distance[pid]

    players = []
    for row in rows.values():
        if row["fga"] == 0 and row["_seen"] * dt < MIN_SEEN_S:
            continue
        row.pop("_seen")
        row["track_ids"].sort()
        row["fg_pct"] = round(row["fgm"] / row["fga"], 3) if row["fga"] else None
        row["possession_s"] = round(row["possession_s"], 1)
        if row["distance_m"] is not None:
            row["distance_m"] = round(row["distance_m"], 1)
        players.append(row)
    players.sort(key=lambda p: (-p["fga"], -p["possession_s"], p["id"]))

    team_fga: Counter[int] = Counter()
    team_fgm: Counter[int] = Counter()
    for s in shots:
        team_fga[s.team] += 1
        team_fgm[s.team] += int(s.made)
    team_ids = sorted({0, 1} | {t for t in team_fga if t >= 0} | ({-1} if team_fga[-1] else set()))
    team_rows = [{"team": t, "fga": team_fga[t], "fgm": team_fgm[t]} for t in team_ids]
    return {"players": players, "teams": team_rows}


def court_distances(calib_path: Path, tracks_path: Path) -> dict[int, float] | None:
    """COURT's projection helper (vision/court/project.py), if it is importable."""
    try:
        from vision.court.project import load_calibration  # owned by COURT

        cal = load_calibration(str(calib_path))
        return {int(k): float(v) for k, v in cal.player_distances(str(tracks_path)).items()}
    except Exception as exc:  # noqa: BLE001 - any failure there must not block stats
        print(f"court projection unavailable ({exc}); using own projection", file=sys.stderr)
        return None


def meta_value(tracks_path: Path, key: str) -> float | None:
    meta = tracks_path.parent / "tracks_meta.json"
    if not meta.exists():
        return None
    try:
        d = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return None
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def meta_fps(tracks_path: Path) -> float | None:
    """Video fps from TRACK's out/tracks_meta.json next to the tracks, if present.
    Frame numbers in the tracks index the *video*, so the source rate wins over
    the (strided) track rate `fps`."""
    meta = tracks_path.parent / "tracks_meta.json"
    if not meta.exists():
        return None
    try:
        d = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return None
    for key in ("source_fps", "video_fps", "fps"):
        if isinstance(d.get(key), (int, float)) and d[key] > 0:
            return float(d[key])
    return None


# --- CLI ------------------------------------------------------------------------


def summary(events: dict, stats: dict) -> str:
    lines = [
        f"shots: {len(events['shots'])}  possessions: {len(events['possessions'])}  "
        f"off-court track ids removed: {len(events.get('off_court_track_ids', []))}"
    ]
    for s in events["shots"]:
        who = f"#{s['player_id']}" if s["player_id"] is not None else "?"
        flag = "" if s["shooter_confirmed"] else " (unconfirmed)"
        lines.append(f"  {s['t']:8.2f}s  {'MADE' if s['made'] else 'miss'}  {who} team {s['team']}{flag}")
    lines.append("players (key team fga fgm fg% poss_s tracks):")
    for p in stats["players"][:12]:
        lines.append(
            f"  {p['key']:>6} {p['team']:>2} {p['fga']:>3} {p['fgm']:>3} "
            f"{'  - ' if p['fg_pct'] is None else f'{p['fg_pct']:.2f}'} {p['possession_s']:6.1f}"
            f"  {len(p['track_ids'])}"
        )
    for t in stats["teams"]:
        lines.append(f"team {t['team']}: {t['fgm']}/{t['fga']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", default="out/tracks.jsonl")
    ap.add_argument("--clip", default="data/clips/game10.mp4")
    ap.add_argument("--fps", type=float, default=None, help="default: inferred from tracks")
    ap.add_argument("--calib", default="out/court_calib.json", help="used if the file exists")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--cuts", default=None, help="cut list; default: out/cuts_<clip>.json if present")
    ap.add_argument("--identities", default="out/identities.json", help="used if the file exists")
    ap.add_argument("--no-court-filter", action="store_true", help="keep bench players and spectators")
    ap.add_argument("--bench-line-frac", type=float, default=0.0,
                    help="interim without calibration: feet above this share of the image height are off court "
                         "(COURT for dev60 segment 853-1562: 505/1080 = 0.47)")
    ap.add_argument("--fixture", choices=["made", "miss", "pass"], help="run on a synthetic scenario")
    ap.add_argument("--min-hold", type=float, default=PossessionParams.min_hold_s, help="seconds")
    ap.add_argument("--max-dist", type=float, default=PossessionParams.max_dist_heights)
    args = ap.parse_args(argv)

    image_height = 1080.0
    if args.fixture:
        fps = args.fps or 50.0
        frames = synthetic_scenario(args.fixture, fps=fps)
        clip = f"fixture:{args.fixture}"
    else:
        fps = args.fps or meta_fps(Path(args.tracks))
        image_height = meta_value(Path(args.tracks), "height") or 1080.0
        frames = read_tracks(args.tracks, fps=fps)
        if not frames:
            print(f"no frames in {args.tracks}", file=sys.stderr)
            return 1
        fps = fps or infer_fps(frames) or 50.0
        clip = args.clip

    calib = None
    distances = None
    calib_path = Path(args.calib)
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
        if not args.fixture:
            distances = court_distances(calib_path, Path(args.tracks))

    out = Path(args.out_dir)
    tracks_dir = Path(args.tracks).parent  # cuts/identities live next to the tracks, not in --out-dir
    cuts = load_cuts(Path(args.cuts)) if args.cuts else (find_cuts(clip, tracks_dir) if not args.fixture else [])
    identities = None
    ident_path = Path(args.identities)
    if ident_path.exists() and not args.fixture:
        meta_path = Path(args.tracks).parent / "tracks_meta.json"
        candidate = load_identities(ident_path)
        if meta_path.exists() and ident_path.stat().st_mtime < meta_path.stat().st_mtime:
            print(f"{ident_path} is older than {meta_path}: track ids belong to a previous run, ignoring it",
                  file=sys.stderr)
        elif not identities_match_tracks(candidate, frames):
            print(f"{ident_path}: track ids/teams do not match these tracks (previous run?), ignoring it",
                  file=sys.stderr)
        else:
            identities = candidate

    events, stats = build(
        frames,
        fps=fps,
        clip=clip,
        calib=calib,
        distances=distances,
        cuts=cuts,
        identities=identities,
        image_height=image_height,
        on_court_filter=not args.no_court_filter,
        bench_line_frac=args.bench_line_frac,
        possession_params=PossessionParams(max_dist_heights=args.max_dist, min_hold_s=args.min_hold),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.json").write_text(json.dumps(events, indent=1))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"{len(frames)} frames @ {fps:g} fps, {len(cuts)} cuts, identities {'yes' if identities else 'no'} "
          f"-> {out / 'events.json'}, {out / 'stats.json'}")
    print(summary(events, stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
