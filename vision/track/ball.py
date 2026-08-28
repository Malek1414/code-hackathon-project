"""Ball plausibility gate.

The ball/hoop detector fires on the two orange wall fixtures next to the
backboard at 0.48-0.75, a real ball scores ~0.86 in flight but drops below
0.5 when it is blurred on the rim, and a plain threshold cannot separate
them. What separates them is time and motion:

* hoop-relative recurrence (LABEL's rule, vision/label/clean_balls.py): a
  fixture keeps the same offset to the hoop box in every frame, a ball does
  not. A candidate whose hoop-relative offset recurred within `rel_px` in at
  least `rel_min` earlier frames of the last `rel_window` processed frames,
  spread over at least `rel_span` frames (0.6 s), is static and dropped; its
  offset is blacklisted for the rest of the camera segment (the Tracker
  builds a fresh gate at every cut). A ball held still for that long loses
  those frames, nothing else.
* absolute blacklist while the camera is not panning (hoop box moved < 4 px
  since the previous frame): every static reject is also blacklisted as a
  pixel position, so it cannot win even in frames without a hoop. Cleared
  when the hoop box jumps > 10 px (pan) or at cuts. An accepted ball that has
  not moved > `static_px` in `static_frames` frames is blacklisted too.
* radius consistency: the ball does not change size between frames. A
  candidate whose radius is off by more than `radius_tol` (40 %) from the
  running median of the last accepted radii is a head, a hand or a sign.
* heads: a candidate whose center lies in the top 20 % of any player box is a
  head, unless the previous ball was already there (a ball held overhead).
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

Every reject is recorded with its reason (`rejects`, one list per step) so
run.py can write out/<clip>/ball_rejects.jsonl for QA.
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
                 radius_tol: float = 0.4, radius_min_samples: int = 5, head_frac: float = 0.2,
                 pan_px: float = 10.0, still_px: float = 4.0) -> None:
        self.base_px, self.per_frame_px, self.max_gate_px = base_px, per_frame_px, max_gate_px
        self.near_px, self.near_grow_px = near_px, near_grow_px
        self.static_px, self.static_frames, self.blacklist_px = static_px, static_frames, blacklist_px
        self.rel_px, self.rel_min, self.rel_window, self.rel_span = rel_px, rel_min, rel_window, rel_span
        self.radius_tol, self.radius_min_samples, self.head_frac = radius_tol, radius_min_samples, head_frac
        self.pan_px, self.still_px = pan_px, still_px

        self.step_no = 0
        self.last: np.ndarray | None = None
        self.last_step: int = -10**9
        self.prev: np.ndarray | None = None  # accepted position before `last`
        self.prev_step: int = -10**9
        self.last_known: np.ndarray | None = None  # survives the static reset
        self.radii: deque[float] = deque(maxlen=25)
        self.history: deque[np.ndarray] = deque(maxlen=static_frames)
        self.blacklist_abs: list[np.ndarray] = []
        self.blacklist_rel: list[np.ndarray] = []
        self.rel_seen: deque[tuple[int, np.ndarray]] = deque()  # (step, offset) of all candidates
        self.prev_hoop_c: np.ndarray | None = None
        self.camera_still = False
        self.rejects: list[dict] = []
        self.counts = {"gate": 0, "static_rel": 0, "blacklist_abs": 0, "blacklist_rel": 0,
                       "radius": 0, "head": 0, "static_abs": 0}
        self.accepted_near_blacklist = 0

    # ----- helpers -------------------------------------------------------------
    def _reject(self, box, conf: float, reason: str) -> None:
        self.counts[reason] += 1
        self.rejects.append({"bbox": [round(float(v), 1) for v in box], "conf": round(conf, 3),
                             "reason": reason})

    def _add_abs(self, c: np.ndarray) -> None:
        if not any(np.linalg.norm(c - b) < self.blacklist_px for b in self.blacklist_abs):
            self.blacklist_abs.append(c.copy())

    def _bad_radius(self, box) -> bool:
        if len(self.radii) < self.radius_min_samples:
            return False
        r = max(box[2] - box[0], box[3] - box[1]) / 2
        med = float(np.median(self.radii))
        return abs(r - med) > self.radius_tol * med

    def _is_head(self, c: np.ndarray, players) -> bool:
        if self.last is not None and float(np.linalg.norm(c - self.last)) < 60:
            return False  # ball was already there (held overhead)
        for b in players:
            if b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[1] + self.head_frac * (b[3] - b[1]):
                return True
        return False

    def _static_rel(self, off: np.ndarray) -> str | None:
        if any(np.linalg.norm(off - b) < self.rel_px * 2 for b in self.blacklist_rel):
            return "blacklist_rel"
        hits = [s for s, o in self.rel_seen if np.linalg.norm(off - o) < self.rel_px]
        if len(hits) >= self.rel_min and hits[-1] - hits[0] >= self.rel_span:
            self.blacklist_rel.append(off)
            return "static_rel"
        return None

    def _near_blacklist(self, c: np.ndarray, hoop_c: np.ndarray | None, px: float = 40.0) -> bool:
        if any(np.linalg.norm(c - b) < px for b in self.blacklist_abs):
            return True
        if hoop_c is not None:
            off = c - hoop_c
            return any(np.linalg.norm(off - b) < px for b in self.blacklist_rel)
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

    @property
    def gap(self) -> int:
        return self.step_no - self.last_step

    # ----- one processed frame -------------------------------------------------
    def begin_step(self, hoop: list[float] | None) -> None:
        """Call once per processed frame, before pick()."""
        self.step_no += 1
        self.rejects = []
        while self.rel_seen and self.step_no - self.rel_seen[0][0] > self.rel_window:
            self.rel_seen.popleft()
        hoop_c = _center(hoop) if hoop else None
        if hoop_c is not None and self.prev_hoop_c is not None:
            moved = float(np.linalg.norm(hoop_c - self.prev_hoop_c))
            self.camera_still = moved < self.still_px
            if moved > self.pan_px:
                self.blacklist_abs.clear()  # pixels moved, pixel blacklist is stale
        else:
            self.camera_still = False
        self.prev_hoop_c = hoop_c

    def pick(self, candidates: list[tuple[float, list[float]]],
             hoop: list[float] | None = None, players=()) -> tuple[float, list[float]] | None:
        """candidates = [(conf, [x1,y1,x2,y2])], players = player boxes; may be
        called more than once per step (crop re-detection). Returns the
        accepted candidate or None."""
        hoop_c = _center(hoop) if hoop else None
        gap = self.gap
        pred = self.predict(gap)
        gate = min(self.base_px + self.per_frame_px * gap, self.max_gate_px)
        near = self.near_px + self.near_grow_px * min(gap, 10)

        tier1, tier2 = [], []
        for conf, box in candidates:
            c = _center(box)
            if hoop_c is not None:
                off = c - hoop_c
                reason = self._static_rel(off)
                self.rel_seen.append((self.step_no, off))
                if reason:
                    self._reject(box, conf, reason)
                    if self.camera_still:
                        self._add_abs(c)
                    continue
            if any(np.linalg.norm(c - b) < self.blacklist_px for b in self.blacklist_abs):
                self._reject(box, conf, "blacklist_abs")
                continue
            if self._bad_radius(box):
                self._reject(box, conf, "radius")
                continue
            if self._is_head(c, players):
                self._reject(box, conf, "head")
                continue
            if pred is None:
                tier2.append((conf, conf, box, c))
                continue
            dist_last = float(np.linalg.norm(c - self.last))
            if dist_last > gate:
                self._reject(box, conf, "gate")
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
                self._add_abs(pts.mean(axis=0))
                if hoop_c is not None:
                    self.blacklist_rel.append(pts.mean(axis=0) - hoop_c)
                self.history.clear()
                self.last, self.last_step, self.prev = None, -10**9, None
                self._reject(box, conf, "static_abs")
                return None
        if self._near_blacklist(c, hoop_c):
            self.accepted_near_blacklist += 1
        self.prev, self.prev_step = self.last, self.last_step
        self.last, self.last_step = c, self.step_no
        self.last_known = c
        self.radii.append(max(box[2] - box[0], box[3] - box[1]) / 2)
        return conf, box

    def summary(self) -> dict:
        return {"ball_rejected": dict(self.counts),
                "ball_accepted_near_blacklist": self.accepted_near_blacklist,
                "ball_blacklist_abs": [b.round(1).tolist() for b in self.blacklist_abs],
                "ball_blacklist_rel": [b.round(1).tolist() for b in self.blacklist_rel]}
