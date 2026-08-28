"""Build `out/events.json` and `out/stats.json` from `out/tracks.jsonl`.

    .venv/bin/python -m vision.stats.build --tracks out/tracks.jsonl --clip data/clips/game10.mp4
    .venv/bin/python -m vision.stats.build --fixture made      # synthetic smoke run

Output follows the contract in docs/ORCHESTRATION.md. Extras that readers may
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

from .engine import StatsEngine
from .io import Frame, infer_fps, median_dt, read_tracks, synthetic_scenario
from .possession import PossessionParams, PossessionResult, possession_seconds
from .shots import ShotEvent, ShotParams

MIN_SEEN_S = 2.0  # players seen shorter than this (and without a shot) are tracker fragments
MAX_PLAYER_SPEED_MS = 12.0  # steps faster than this are id swaps, not running


def build(
    frames: list[Frame],
    *,
    fps: float,
    clip: str,
    calib: dict | None = None,
    distances: dict[int, float] | None = None,
    possession_params: PossessionParams = PossessionParams(),
    shot_params: ShotParams = ShotParams(),
) -> tuple[dict, dict]:
    """`distances` (player id -> metres) wins over `calib` (own projection)."""
    engine = StatsEngine(dt=median_dt(frames), possession_params=possession_params, shot_params=shot_params)
    for fr in frames:
        engine.push(fr)
    engine.finish()
    possession = engine.possession.result()
    shots = engine.shots
    events = {
        "fps": fps,
        "clip": clip,
        "shots": [s.to_dict() for s in shots],
        "possessions": [
            {
                "player_id": s.player_id,
                "team": s.team,
                "start_t": round(s.start_t, 3),
                "end_t": round(s.end_t, 3),
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
            }
            for s in possession.segments
        ],
    }
    stats = player_stats(frames, possession, shots, calib=calib, distances=distances)
    return events, stats


def player_stats(
    frames: list[Frame],
    possession: PossessionResult,
    shots: list[ShotEvent],
    *,
    calib: dict | None = None,
    distances: dict[int, float] | None = None,
) -> dict:
    dt = median_dt(frames)
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

    players = []
    for pid, n in seen.items():
        if fga[pid] == 0 and n * dt < MIN_SEEN_S:
            continue
        team = teams[pid].most_common(1)[0][0] if teams[pid] else -1
        players.append(
            {
                "id": pid,
                "team": team,
                "fga": fga[pid],
                "fgm": fgm[pid],
                "fg_pct": round(fgm[pid] / fga[pid], 3) if fga[pid] else 0.0,
                "possession_s": round(poss_s.get(pid, 0.0), 1),
                "distance_m": round(distance[pid], 1) if pid in distance else None,
            }
        )
    players.sort(key=lambda p: (-p["fga"], -p["possession_s"], p["id"]))

    team_fga: Counter[int] = Counter()
    team_fgm: Counter[int] = Counter()
    for s in shots:
        team_fga[s.team] += 1
        team_fgm[s.team] += int(s.made)
    team_ids = sorted({0, 1} | {t for t in team_fga if t >= 0} | ({-1} if team_fga[-1] else set()))
    team_rows = [{"team": t, "fga": team_fga[t], "fgm": team_fgm[t]} for t in team_ids]
    return {"players": players, "teams": team_rows}


# --- distance via court calibration --------------------------------------------


def _homography_for(calib: dict, frame_no: int) -> list[list[float]] | None:
    per_frame = calib.get("frames")
    if per_frame:
        nearest = min(per_frame, key=lambda k: abs(int(k) - frame_no))
        return per_frame[nearest]
    return calib.get("H_px_to_m")


def project(H: list[list[float]], p: tuple[float, float]) -> tuple[float, float]:
    x, y = p
    w = H[2][0] * x + H[2][1] * y + H[2][2]
    if abs(w) < 1e-9:
        return (math.nan, math.nan)
    return ((H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w)


def distances_m(frames: list[Frame], calib: dict) -> dict[int, float]:
    """Path length of every player's projected foot point, in metres."""
    last: dict[int, tuple[float, tuple[float, float]]] = {}
    total: dict[int, float] = defaultdict(float)
    for fr in frames:
        H = _homography_for(calib, fr.frame)
        if H is None:
            continue
        for p in fr.players:
            m = project(H, p.foot)
            if not all(math.isfinite(v) for v in m):
                continue
            if p.id in last:
                t0, m0 = last[p.id]
                dt = fr.t - t0
                d = math.hypot(m[0] - m0[0], m[1] - m0[1])
                if 0 < dt <= 0.5 and d / dt <= MAX_PLAYER_SPEED_MS:
                    total[p.id] += d
            last[p.id] = (fr.t, m)
    return dict(total)


def court_distances(calib_path: Path, tracks_path: Path) -> dict[int, float] | None:
    """COURT's projection helper (vision/court/project.py), if it is importable."""
    try:
        from vision.court.project import load_calibration  # owned by COURT

        cal = load_calibration(str(calib_path))
        return {int(k): float(v) for k, v in cal.player_distances(str(tracks_path)).items()}
    except Exception as exc:  # noqa: BLE001 - any failure there must not block stats
        print(f"court projection unavailable ({exc}); using own projection", file=sys.stderr)
        return None


def meta_fps(tracks_path: Path) -> float | None:
    """Video fps from TRACK's out/tracks_meta.json next to the tracks, if present."""
    meta = tracks_path.parent / "tracks_meta.json"
    if not meta.exists():
        return None
    try:
        d = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return None
    for key in ("fps", "video_fps", "source_fps"):
        if isinstance(d.get(key), (int, float)) and d[key] > 0:
            return float(d[key])
    return None


# --- CLI ------------------------------------------------------------------------


def summary(events: dict, stats: dict) -> str:
    lines = [f"shots: {len(events['shots'])}  possessions: {len(events['possessions'])}"]
    for s in events["shots"]:
        who = f"#{s['player_id']}" if s["player_id"] is not None else "?"
        flag = "" if s["shooter_confirmed"] else " (unconfirmed)"
        lines.append(f"  {s['t']:8.2f}s  {'MADE' if s['made'] else 'miss'}  {who} team {s['team']}{flag}")
    lines.append("players (id team fga fgm fg% poss_s):")
    for p in stats["players"][:12]:
        lines.append(
            f"  {p['id']:>4} {p['team']:>2} {p['fga']:>3} {p['fgm']:>3} {p['fg_pct']:.2f} {p['possession_s']:6.1f}"
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
    ap.add_argument("--fixture", choices=["made", "miss", "pass"], help="run on a synthetic scenario")
    ap.add_argument("--min-hold", type=float, default=PossessionParams.min_hold_s, help="seconds")
    ap.add_argument("--max-dist", type=float, default=PossessionParams.max_dist_heights)
    args = ap.parse_args(argv)

    if args.fixture:
        fps = args.fps or 50.0
        frames = synthetic_scenario(args.fixture, fps=fps)
        clip = f"fixture:{args.fixture}"
    else:
        fps = args.fps or meta_fps(Path(args.tracks))
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

    events, stats = build(
        frames,
        fps=fps,
        clip=clip,
        calib=calib,
        distances=distances,
        possession_params=PossessionParams(max_dist_heights=args.max_dist, min_hold_s=args.min_hold),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.json").write_text(json.dumps(events, indent=1))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"{len(frames)} frames @ {fps:g} fps -> {out / 'events.json'}, {out / 'stats.json'}")
    print(summary(events, stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
