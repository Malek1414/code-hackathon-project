"""Ball possession from per-frame tracks.

Rule (docs/ORCHESTRATION.md, STATS milestone 12:00): the holder is the player
whose bbox is nearest to the ball center, but only if the ball center lies
within `max_dist_heights` bbox heights of that player's bbox center. A change
of holder has to persist for `min_frames` processed frames before it counts
(hysteresis against flicker); once confirmed it is backdated to the first
frame of the streak. Frames without a ball detection carry the previous state.

Everything is measured inside one frame (ball vs. player in the same image),
so a panning camera does not disturb it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .io import Frame, Player, median_dt


_NO_PENDING = object()  # distinct from None, which is a valid candidate (loose ball)


@dataclass
class PossessionParams:
    max_dist_heights: float = 1.5
    min_frames: int = 10


@dataclass
class Possession:
    player_id: int
    team: int
    start_frame: int
    end_frame: int
    start_t: float
    end_t: float
    start_index: int  # indices into the frames list
    end_index: int

    def duration_s(self, dt: float) -> float:
        return self.end_t - self.start_t + dt


@dataclass
class PossessionResult:
    holder: list[int | None]  # per frames-list index: player id or None (loose ball)
    segments: list[Possession]

    def holder_at(self, index: int) -> int | None:
        return self.holder[index]


def normalized_distance(player: Player, point: tuple[float, float]) -> float:
    """Ball-to-bbox-center distance in units of the player's bbox height."""
    h = max(player.height, 1.0)
    cx, cy = player.center
    return math.hypot(point[0] - cx, point[1] - cy) / h


def nearest_player(fr: Frame, params: PossessionParams = PossessionParams()) -> Player | None:
    """Candidate holder for one frame, or None (no ball / ball loose)."""
    if fr.ball is None or not fr.players:
        return None
    best, best_d = None, math.inf
    for p in fr.players:
        d = normalized_distance(p, fr.ball.center)
        if d < best_d:
            best, best_d = p, d
    return best if best_d <= params.max_dist_heights else None


def track_possession(frames: list[Frame], params: PossessionParams = PossessionParams()) -> PossessionResult:
    holder: list[int | None] = []
    current: int | None = None
    pending: object = _NO_PENDING  # candidate (id or None) that differs from `current`
    pending_start = 0
    pending_len = 0

    for i, fr in enumerate(frames):
        if fr.ball is None:
            holder.append(current)  # no information: carry the state, don't count
            continue
        cand = nearest_player(fr, params)
        cand_id = cand.id if cand else None

        if cand_id == current:
            pending, pending_len = _NO_PENDING, 0
            holder.append(current)
            continue

        if cand_id == pending:
            pending_len += 1
        else:
            pending, pending_start, pending_len = cand_id, i, 1

        if pending_len >= params.min_frames:
            current = pending
            for j in range(pending_start, i):  # backdate to the start of the streak
                holder[j] = current
            pending, pending_len = _NO_PENDING, 0
        holder.append(current)

    return PossessionResult(holder=holder, segments=_segments(frames, holder))


def _segments(frames: list[Frame], holder: list[int | None]) -> list[Possession]:
    segments: list[Possession] = []
    start = None
    for i in range(len(frames) + 1):
        pid = holder[i] if i < len(frames) else None
        if start is not None and (i == len(frames) or pid != holder[start]):
            segments.append(_make_segment(frames, holder[start], start, i - 1))
            start = None
        if pid is not None and start is None:
            start = i
    return segments


def _make_segment(frames: list[Frame], pid: int, a: int, b: int) -> Possession:
    teams = Counter()
    for fr in frames[a : b + 1]:
        p = fr.player(pid)
        if p is not None and p.team >= 0:
            teams[p.team] += 1
    team = teams.most_common(1)[0][0] if teams else -1
    return Possession(
        player_id=pid,
        team=team,
        start_frame=frames[a].frame,
        end_frame=frames[b].frame,
        start_t=frames[a].t,
        end_t=frames[b].t,
        start_index=a,
        end_index=b,
    )


def possession_seconds(frames: list[Frame], result: PossessionResult) -> dict[int, float]:
    dt = median_dt(frames)
    out: dict[int, float] = {}
    for s in result.segments:
        out[s.player_id] = out.get(s.player_id, 0.0) + s.duration_s(dt)
    return out
