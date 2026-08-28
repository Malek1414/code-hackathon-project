"""Quick data-quality report for an out/tracks.jsonl before trusting the stats.

    .venv/bin/python -m vision.stats.diagnose out/tracks.jsonl

Tells you whether the ball and the hoop are seen often enough for shot
detection to mean anything, how fragmented the player ids are, and where the
ball was near a hoop (candidate moments to check by hand).
"""

from __future__ import annotations

import sys
from collections import Counter

from .io import Frame, infer_fps, median_dt, read_tracks
from .shots import ShotParams, in_up_zone


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def report(frames: list[Frame]) -> str:
    n = len(frames)
    if n == 0:
        return "no frames"
    fps = infer_fps(frames)
    dt = median_dt(frames)
    span = frames[-1].t - frames[0].t
    with_ball = sum(1 for f in frames if f.ball)
    with_hoop = sum(1 for f in frames if f.hoops)
    hoop_counts = Counter(len(f.hoops) for f in frames)
    players_per_frame = [len(f.players) for f in frames]
    seen: Counter[int] = Counter()
    team_of: dict[int, Counter] = {}
    for f in frames:
        for p in f.players:
            seen[p.id] += 1
            team_of.setdefault(p.id, Counter())[p.team] += 1
    long_ids = [pid for pid, c in seen.items() if c * dt >= 2.0]
    teams = Counter(t.most_common(1)[0][0] for t in team_of.values())

    # longest ball gaps
    gaps = []
    last = None
    for f in frames:
        if f.ball:
            if last is not None and f.t - last > 1.0:
                gaps.append((last, f.t))
            last = f.t
    gaps.sort(key=lambda g: g[0] - g[1])

    # ball near a hoop (zone hits), merged into episodes
    p = ShotParams()
    episodes: list[list[float]] = []
    for f in frames:
        if not f.ball or not f.hoops:
            continue
        if any(in_up_zone(f.ball.center, h, p) for h in f.hoops):
            if episodes and f.t - episodes[-1][1] < 1.0:
                episodes[-1][1] = f.t
            else:
                episodes.append([f.t, f.t])

    lines = [
        f"frames {n}  span {span:.1f}s  fps {fps}  frame spacing {dt * 1000:.0f} ms",
        f"ball detected in {with_ball / n:.0%} of frames, {len(gaps)} gaps > 1 s"
        + (f" (longest {gaps[0][1] - gaps[0][0]:.1f}s at {_mmss(gaps[0][0])})" if gaps else ""),
        f"hoop box in {with_hoop / n:.0%} of frames, hoops per frame {dict(sorted(hoop_counts.items()))}",
        f"players per frame min/median/max {min(players_per_frame)}/{sorted(players_per_frame)[n // 2]}/{max(players_per_frame)}",
        f"player ids: {len(seen)} total, {len(long_ids)} seen >= 2 s, teams of those {dict(teams)}",
        f"ball-in-hoop-zone episodes: {len(episodes)}",
    ]
    for a, b in episodes[:40]:
        lines.append(f"  {_mmss(a)} - {_mmss(b)}")
    if len(episodes) > 40:
        lines.append(f"  ... {len(episodes) - 40} more")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else "out/tracks.jsonl"
    fps = float(argv[1]) if len(argv) > 1 else None
    print(report(read_tracks(path, fps=fps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
