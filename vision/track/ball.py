"""Ball plausibility gate.

The ball/hoop detector fires on orange wall fixtures next to the backboard
(fire alarm, orange box) at 0.48-0.75, a real ball scores ~0.86 in flight but
drops below 0.5 when it is blurred on the rim, and a plain threshold cannot
separate them. What separates them is time and motion:

* hoop-relative recurrence (LABEL's rule, vision/label/clean_balls.py): a
  fixture keeps the same offset to the hoop box in every frame, a ball does
  not. A candidate whose hoop-relative offset recurred within `rel_px` in at
  least `rel_min` earlier frames of the last `rel_window` processed frames,
  spread over at least `rel_span` frames (0.6 s), is static and dropped; its
  offset is blacklisted for the rest of the clip. A ball held still for that
  long loses those frames, nothing else.
* absolute static filter for frames without a hoop: an accepted ball that has
  not moved more than `static_px` in `static_frames` frames is a fixture.
* trajectory: the ball's next position is extrapolated from the last two
  accepted positions (damped). Candidates within `near_px` (+ growth per
  missed frame) of that prediction form the first tier and always beat
  candidates outside it, whatever their confidence: measured on dev60 57.0 s,
  a 0.75 fixture 320 px away must not outscore a 0.5 ball on the rim. Inside a
  tier, confidence minus a distance penalty decides.
* gate: a candidate must lie within `base + per_frame * gap` px of the last
  accepted ball; the gate grows with the gap, so a lost ball (out of frame,
  occluded) is re-acquired anywhere, but a wall object while the ball is in
  view is not.

Inspired by courtside/engine/analytics/ball.py (max-speed outlier drop).
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _center(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)


class BallGate:
    def __init__(self, *, base_px: float = 120.0, per_frame_px: float = 80.0,
                 max_gate_px: float = 900.0, near_px: float = 120.0, near_grow_px: float = 40.0,
                 static_px: float = 6.0, static_frames: int = 30, blacklist_px: float = 25.0,
                 rel_px: float = 8.0, rel_min: int = 4, rel_window: int = 150, rel_span: int = 15,
                 blacklist_rel: list[np.ndarray] | None = None) -> None:
        self.base_px, self.per_frame_px, self.max_gate_px = base_px, per_frame_px, max_gate_px
        self.near_px, self.near_grow_px = near_px, near_grow_px
        self.static_px, self.static_frames, self.blacklist_px = static_px, static_frames, blacklist_px
        self.rel_px, self.rel_min, self.rel_window, self.rel_span = rel_px, rel_min, rel_window, rel_span

        self.step_no = 0
        self.last: np.ndarray | None = None
        self.last_step: int = -10**9
        self.prev: np.ndarray | None = None  # accepted position before `last`
        self.prev_step: int = -10**9
        self.history: deque[np.ndarray] = deque(maxlen=static_frames)
        self.blacklist_abs: list[np.ndarray] = []
        self.blacklist_rel: list[np.ndarray] = list(blacklist_rel or [])
        self.rel_seen: deque[tuple[int, np.ndarray]] = deque()  # (step, offset) of all candidates
        self.rejected_gate = 0
        self.rejected_static = 0

    def _is_static_rel(self, off: np.ndarray) -> bool:
        if any(np.linalg.norm(off - b) < self.rel_px * 2 for b in self.blacklist_rel):
            return True
        hits = [s for s, o in self.rel_seen if np.linalg.norm(off - o) < self.rel_px]
        if len(hits) >= self.rel_min and hits[-1] - hits[0] >= self.rel_span:
            self.blacklist_rel.append(off)
            return True
        return False

    def predict(self, gap: int) -> np.ndarray | None:
        """Where the ball should be `gap` processed frames after the last fix."""
        if self.last is None:
            return None
        if self.prev is None or self.last_step - self.prev_step > 3:
            return self.last
        v = (self.last - self.prev) / (self.last_step - self.prev_step)
        # Damped: a bouncing or caught ball does not keep its velocity for long.
        return self.last + v * min(gap, 4) * 0.7

    def pick(self, candidates: list[tuple[float, list[float]]],
             hoop: list[float] | None = None) -> tuple[float, list[float]] | None:
        """candidates = [(conf, [x1,y1,x2,y2])]; returns the accepted one or None."""
        self.step_no += 1
        while self.rel_seen and self.step_no - self.rel_seen[0][0] > self.rel_window:
            self.rel_seen.popleft()
        hoop_c = _center(hoop) if hoop else None
        gap = self.step_no - self.last_step
        pred = self.predict(gap)
        gate = min(self.base_px + self.per_frame_px * gap, self.max_gate_px)
        near = self.near_px + self.near_grow_px * min(gap, 10)

        tier1, tier2 = [], []
        for conf, box in candidates:
            c = _center(box)
            if hoop_c is not None:
                off = c - hoop_c
                static = self._is_static_rel(off)
                self.rel_seen.append((self.step_no, off))
                if static:
                    self.rejected_static += 1
                    continue
            if any(np.linalg.norm(c - b) < self.blacklist_px for b in self.blacklist_abs):
                self.rejected_static += 1
                continue
            if pred is None:
                tier2.append((conf, conf, box, c))
                continue
            dist_last = float(np.linalg.norm(c - self.last))
            if dist_last > gate:
                self.rejected_gate += 1
                continue
            dist_pred = float(np.linalg.norm(c - pred))
            score = conf - 0.5 * dist_pred / gate
            (tier1 if dist_pred <= near else tier2).append((score, conf, box, c))
        pool = tier1 or tier2
        if not pool:
            return None
        _s, conf, box, c = max(pool, key=lambda x: x[0])

        self.history.append(c)
        if len(self.history) == self.static_frames:
            pts = np.stack(self.history)
            if np.ptp(pts, axis=0).max() < self.static_px:
                self.blacklist_abs.append(pts.mean(axis=0))
                self.history.clear()
                self.last, self.last_step, self.prev = None, -10**9, None
                self.rejected_static += 1
                return None
        self.prev, self.prev_step = self.last, self.last_step
        self.last, self.last_step = c, self.step_no
        return conf, box

    def summary(self) -> dict:
        return {"ball_rejected_gate": self.rejected_gate,
                "ball_rejected_static": self.rejected_static,
                "ball_blacklist_abs": [b.round(1).tolist() for b in self.blacklist_abs],
                "ball_blacklist_rel": [b.round(1).tolist() for b in self.blacklist_rel]}
