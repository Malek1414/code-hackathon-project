"""Ball possession from per-frame tracks, incremental.

Rule (docs/ORCHESTRATION.md, STATS milestone 12:00): the holder is the player
whose bbox is nearest to the ball center, but only if the ball center lies
within `max_dist_heights` bbox heights of that player's bbox center. A change
of holder has to persist for `min_hold_s` (10 frames at 50 fps; tracks come
at 10-25 fps, so it is measured in seconds) before it counts (hysteresis
against flicker), a loose ball for `min_loose_s`; once confirmed it is
backdated to the first frame of the streak. Frames without a ball detection
carry the previous state.

Everything is measured inside one frame (ball vs. player in the same image),
so a panning camera does not disturb it. `PossessionTracker.push` takes one
frame at a time (live mode); `track_possession` runs it over a list.
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
    min_hold_s: float = 0.2  # a new holder must be nearest for this long (10 frames at 50 fps)
    min_loose_s: float = 0.5  # "nobody" needs more evidence: stray ball boxes are common
    min_frames: int | None = None  # override: count processed frames instead of seconds

    def frames_for(self, dt: float, loose: bool) -> int:
        if self.min_frames is not None:
            return self.min_frames
        if dt <= 0:
            return 1
        secs = self.min_loose_s if loose else self.min_hold_s
        return max(2, int(round(secs / dt)))


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


class PossessionTracker:
    """Feed frames in order; `holder[i]` is the holder of the i-th pushed frame.

    `dt` is the typical frame spacing in seconds (drives the hysteresis
    lengths). Live mode passes its processing interval; batch passes the
    median spacing of the file.
    """

    def __init__(self, params: PossessionParams = PossessionParams(), dt: float = 0.1) -> None:
        self.params = params
        self.dt = dt
        self.frames: list[Frame] = []
        self.holder: list[int | None] = []
        self._current: int | None = None
        self._pending: object = _NO_PENDING
        self._pending_start = 0
        self._pending_len = 0
        self.cut_indices: list[int] = []  # frame-list indices where a segment must break

    @property
    def current(self) -> int | None:
        return self._current

    def reset(self) -> None:
        """Forget the holder and any pending streak (cut in the footage); the
        next frame starts a new segment even if the same player holds on."""
        self._current = None
        self._pending, self._pending_len = _NO_PENDING, 0
        self.cut_indices.append(len(self.frames))

    def push(self, fr: Frame) -> int | None:
        self.frames.append(fr)
        i = len(self.frames) - 1
        if fr.ball is None:
            self.holder.append(self._current)  # no information: carry the state, don't count
            return self._current
        cand = nearest_player(fr, self.params)
        cand_id = cand.id if cand else None

        if cand_id == self._current:
            self._pending, self._pending_len = _NO_PENDING, 0
            self.holder.append(self._current)
            return self._current

        if cand_id == self._pending:
            self._pending_len += 1
        else:
            self._pending, self._pending_start, self._pending_len = cand_id, i, 1

        need = self.params.frames_for(self.dt, loose=self._pending is None)
        if self._pending_len >= need:
            self._current = self._pending
            for j in range(self._pending_start, i):  # backdate to the start of the streak
                self.holder[j] = self._current
            self._pending, self._pending_len = _NO_PENDING, 0
        self.holder.append(self._current)
        return self._current

    @property
    def segments(self) -> list[Possession]:
        """Possession segments so far; the last one may still be open."""
        segments: list[Possession] = []
        frames, holder = self.frames, self.holder
        cuts = set(self.cut_indices)
        start = None
        for i in range(len(frames) + 1):
            pid = holder[i] if i < len(frames) else None
            if start is not None and (i == len(frames) or pid != holder[start] or i in cuts):
                segments.append(_make_segment(frames, holder[start], start, i - 1))
                start = None
            if pid is not None and start is None:
                start = i
        return segments

    def result(self) -> PossessionResult:
        return PossessionResult(holder=list(self.holder), segments=self.segments)


def track_possession(frames: list[Frame], params: PossessionParams = PossessionParams()) -> PossessionResult:
    tracker = PossessionTracker(params, dt=median_dt(frames))
    for fr in frames:
        tracker.push(fr)
    return tracker.result()


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
