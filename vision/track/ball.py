"""Ball plausibility gate.

The ball/hoop detector fires on orange wall fixtures next to the backboard
(fire alarm, orange box) at 0.48-0.69, a real ball scores ~0.86, and a plain
threshold cannot separate them. What separates them is time:

* hoop-relative recurrence (LABEL's rule, vision/label/clean_balls.py): a
  fixture keeps the same offset to the hoop box in every frame, a ball does
  not. A candidate whose hoop-relative offset recurred within `rel_px` in at
  least `rel_min` earlier frames of the last `rel_window` processed frames,
  spread over at least `rel_span` frames, is static and dropped; its offset is
  blacklisted for the rest of the clip. The span requirement is what keeps a
  ball held still before a free throw (1-2 s) alive.
* absolute static filter for frames without a hoop: an accepted ball that has
  not moved more than `static_px` in `static_frames` frames is a fixture.
* gating: a candidate must lie within `base + per_frame * gap` px of the last
  accepted ball; the gate grows with the gap, so a lost ball (out of frame,
  occluded) is re-acquired anywhere, but a wall object while the ball is in
  view is not. Among survivors the one nearest the previous ball wins, weighted
  by confidence.

Inspired by courtside/engine/analytics/ball.py (max-speed outlier drop).
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _center(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)


class BallGate:
    def __init__(self, *, base_px: float = 120.0, per_frame_px: float = 80.0,
                 max_gate_px: float = 900.0, static_px: float = 6.0, static_frames: int = 75,
                 blacklist_px: float = 25.0, rel_px: float = 8.0, rel_min: int = 4,
                 rel_window: int = 150, rel_span: int = 75) -> None:
        self.base_px = base_px
        self.per_frame_px = per_frame_px
        self.max_gate_px = max_gate_px
        self.static_px = static_px
        self.static_frames = static_frames
        self.blacklist_px = blacklist_px
        self.rel_px = rel_px
        self.rel_min = rel_min
        self.rel_window = rel_window
        self.rel_span = rel_span

        self.step_no = 0
        self.last: np.ndarray | None = None
        self.last_step: int = -10**9
        self.history: deque[np.ndarray] = deque(maxlen=static_frames)
        self.blacklist_abs: list[np.ndarray] = []
        self.blacklist_rel: list[np.ndarray] = []
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

    def pick(self, candidates: list[tuple[float, list[float]]],
             hoop: list[float] | None = None) -> tuple[float, list[float]] | None:
        """candidates = [(conf, [x1,y1,x2,y2])]; returns the accepted one or None."""
        self.step_no += 1
        while self.rel_seen and self.step_no - self.rel_seen[0][0] > self.rel_window:
            self.rel_seen.popleft()
        hoop_c = _center(hoop) if hoop else None

        scored = []
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
            if self.last is not None:
                gap = self.step_no - self.last_step
                gate = min(self.base_px + self.per_frame_px * gap, self.max_gate_px)
                dist = float(np.linalg.norm(c - self.last))
                if dist > gate:
                    self.rejected_gate += 1
                    continue
                scored.append((conf - 0.3 * dist / gate, conf, box, c))
            else:
                scored.append((conf, conf, box, c))
        if not scored:
            return None
        _s, conf, box, c = max(scored, key=lambda x: x[0])

        self.history.append(c)
        if len(self.history) == self.static_frames:
            pts = np.stack(self.history)
            if np.ptp(pts, axis=0).max() < self.static_px:
                self.blacklist_abs.append(pts.mean(axis=0))
                self.history.clear()
                self.last, self.last_step = None, -10**9
                self.rejected_static += 1
                return None
        self.last, self.last_step = c, self.step_no
        return conf, box

    def summary(self) -> dict:
        return {"ball_rejected_gate": self.rejected_gate,
                "ball_rejected_static": self.rejected_static,
                "ball_blacklist_abs": [b.round(1).tolist() for b in self.blacklist_abs],
                "ball_blacklist_rel": [b.round(1).tolist() for b in self.blacklist_rel]}
