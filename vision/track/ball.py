"""Ball plausibility gate.

The ball/hoop detector fires on the two orange wall fixtures next to the
backboard at 0.48-0.75, a real ball scores ~0.86 in flight but drops below
0.5 when it is blurred on the rim, and a plain threshold cannot separate
them. What separates them is time and motion:

* hoop-relative recurrence (LABEL's rule, vision/label/clean_balls.py): a
  fixture keeps the same offset to the hoop box in every frame, a ball does
  not. Offsets are measured in hoop widths, so they survive zoom, pans and
  cuts (the wall fixture is 2.5 hoop widths left of the rim in every shot),
  and the blacklist is handed from segment to segment. A candidate whose
  offset recurred within `rel_tol` hoop widths in at least `rel_min` earlier
  frames of the last `rel_window` processed frames, spread over at least
  `rel_span` frames (0.3 s), is static and dropped and its offset is
  blacklisted. A ball held still for that long loses those frames, nothing
  else; Sami's call (28.08. 13:59): strict against the fixtures even at the
  cost of a still ball. A run can be seeded with the offsets of an earlier
  run in the same hall (`blacklist_rel`, hoop widths), so the fixtures are
  known from frame 0. Measured on dev60 v3: without the hand-over, the fixture won the
  first 0.6 s after the cut at 2798 and the real shot arc (0.83-0.88) was
  gated out.
* static takeover: when the last 5 accepted positions did not move (< 8 px)
  the gate opens fully and confidence alone decides, so a high-confidence
  candidate anywhere beats what is by then most likely a fixture (a ball held
  still keeps winning as long as it scores higher than the alternatives).
* absolute blacklist while the camera is not panning (hoop box moved < 4 px
  since the previous frame): every static reject is also blacklisted as a
  pixel position, so it cannot win even in frames without a hoop. Cleared
  when the hoop box jumps > 10 px (pan) or at cuts. An accepted ball that has
  not moved > `static_px` in `static_frames` frames is blacklisted too.
* wall rule (hard): a candidate whose center is at or above the rim line
  (hoop box top), more than 2.5 hoop widths from the rim horizontally, with
  no player box within 150 px, is a wall object. Exempt when the candidate
  continues the predicted trajectory, otherwise the rule would reject a
  shot at its apex (high, far from the rim, nobody near).
* radius plausibility, perspective-aware: a ball is ~24 cm next to ~190 cm
  players, so its radius must be 3-14 % of the height of the nearest player
  box (no check without a player within 500 px). A rolling-median rule was
  tried first and measured wrong: dev60 frames 1800-1816 the real ball at
  0.87-0.91 conf was rejected because it flew towards the camera and doubled
  in size while the median came from far-away frames.
* heads: a candidate with conf < `head_conf` whose center lies in the top
  20 % and the central 60 % of a player box is a head, unless the previous
  ball was already there. High-confidence candidates are exempt on purpose:
  measured on dev60 frame 2336 the rule threw away the ball at the release
  of a shot (0.87 conf, above the shooter's hands, top of his box).
* prediction first, Kalman style (Sami's spec, 28.08. 14:02/14:08): the ball
  has its own constant-velocity Kalman filter (x, y, vx, vy) with a gravity
  term on vy. Each processed frame the filter predicts; a detection is
  accepted only inside the gate around the prediction (`near_px`, growing
  with the frames since the last update), the filter updates on a hit and
  COASTS on the prediction for up to `coast_frames` (0.5 s) on a miss: the
  predicted point is emitted with predicted=True (drawn hollow). After that
  the ball is lost and may be re-acquired anywhere (occlusion, out of frame),
  which re-initialises the filter. A far high-confidence detection can never
  win while the filter is alive, so wall objects do not "take the video".

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
                 rel_tol: float = 0.12, rel_min: int = 3, rel_window: int = 150, rel_span: int = 8,
                 radius_frac: tuple[float, float] = (0.03, 0.14), head_frac: float = 0.2,
                 head_conf: float = 0.75, wall_hoop_widths: float = 2.5, wall_player_px: float = 150.0,
                 coast_frames: int = 12, gravity_px: float = 2.5, process_noise: float = 4.0,
                 measurement_noise: float = 6.0,
                 pan_px: float = 10.0, still_px: float = 4.0, takeover_frames: int = 5,
                 takeover_px: float = 8.0, counts: dict | None = None,
                 blacklist_rel: list[np.ndarray] | None = None) -> None:
        self.base_px, self.per_frame_px, self.max_gate_px = base_px, per_frame_px, max_gate_px
        self.near_px, self.near_grow_px = near_px, near_grow_px
        self.static_px, self.static_frames, self.blacklist_px = static_px, static_frames, blacklist_px
        self.rel_tol, self.rel_min, self.rel_window, self.rel_span = rel_tol, rel_min, rel_window, rel_span
        self.takeover_frames, self.takeover_px = takeover_frames, takeover_px
        self.radius_frac, self.head_frac, self.head_conf = radius_frac, head_frac, head_conf
        self.wall_hoop_widths, self.wall_player_px = wall_hoop_widths, wall_player_px
        self.coast_frames, self.gravity_px = coast_frames, gravity_px
        self.q, self.r_meas = process_noise, measurement_noise
        self.kf_x: np.ndarray | None = None  # [x, y, vx, vy]
        self.kf_P: np.ndarray | None = None
        self.misses = 0
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
        self.blacklist_rel: list[np.ndarray] = list(blacklist_rel or [])  # in hoop widths
        self.rel_seen: deque[tuple[int, np.ndarray]] = deque()  # (step, offset) of all candidates
        self.prev_hoop_c: np.ndarray | None = None
        self.camera_still = False
        self.rejects: list[dict] = []
        self.counts = counts if counts is not None else {  # shared across cut resets
            "gate": 0, "static_rel": 0, "blacklist_abs": 0, "blacklist_rel": 0,
            "radius": 0, "head": 0, "wall": 0, "far": 0, "static_abs": 0, "accepted_near_blacklist": 0}
        self.coasted = 0
        self.accepted_near_blacklist = 0

    # ----- helpers -------------------------------------------------------------
    def _reject(self, box, conf: float, reason: str) -> None:
        self.counts[reason] += 1
        self.rejects.append({"bbox": [round(float(v), 1) for v in box], "conf": round(conf, 3),
                             "reason": reason})

    def _add_abs(self, c: np.ndarray) -> None:
        if not any(np.linalg.norm(c - b) < self.blacklist_px for b in self.blacklist_abs):
            self.blacklist_abs.append(c.copy())

    def _bad_radius(self, box, players) -> bool:
        if not len(players):
            return False
        c = _center(box)
        r = max(box[2] - box[0], box[3] - box[1]) / 2
        best, best_h = 1e9, 0.0
        for b in players:
            d = float(np.hypot((b[0] + b[2]) / 2 - c[0], (b[1] + b[3]) / 2 - c[1]))
            if d < best:
                best, best_h = d, float(b[3] - b[1])
        if best > 500 or best_h <= 0:
            return False
        frac = r / best_h
        return not (self.radius_frac[0] <= frac <= self.radius_frac[1])

    def _is_head(self, c: np.ndarray, conf: float, players) -> bool:
        if conf >= self.head_conf:
            return False  # a confident ball above the hands is a shot, not a head
        if self.last is not None and self.kf_x is not None and float(np.linalg.norm(c - self.last)) < 60:
            return False  # ball was already there (held overhead)
        for b in players:
            w = b[2] - b[0]
            if b[0] + 0.2 * w <= c[0] <= b[2] - 0.2 * w \
                    and b[1] <= c[1] <= b[1] + self.head_frac * (b[3] - b[1]):
                return True
        return False

    def _is_wall(self, c: np.ndarray, hoop, players) -> bool:
        if hoop is None:
            return False
        hoop_w = max(float(hoop[2] - hoop[0]), 1.0)
        rim_y, rim_cx = float(hoop[1]), (hoop[0] + hoop[2]) / 2
        if c[1] > rim_y or abs(c[0] - rim_cx) <= self.wall_hoop_widths * hoop_w:
            return False
        for b in players:
            dx = max(b[0] - c[0], 0, c[0] - b[2])
            dy = max(b[1] - c[1], 0, c[1] - b[3])
            if float(np.hypot(dx, dy)) <= self.wall_player_px:
                return False
        return True

    def _static_rel(self, off: np.ndarray) -> str | None:
        if any(np.linalg.norm(off - b) < self.rel_tol * 2 for b in self.blacklist_rel):
            return "blacklist_rel"
        hits = [s for s, o in self.rel_seen if np.linalg.norm(off - o) < self.rel_tol]
        if len(hits) >= self.rel_min and hits[-1] - hits[0] >= self.rel_span:
            self.blacklist_rel.append(off)
            return "static_rel"
        return None

    def _near_blacklist(self, c: np.ndarray, hoop_c: np.ndarray | None, hoop_w: float,
                        px: float = 40.0) -> bool:
        if any(np.linalg.norm(c - b) < px for b in self.blacklist_abs):
            return True
        if hoop_c is not None:
            off = (c - hoop_c) / hoop_w
            return any(np.linalg.norm(off - b) * hoop_w < px for b in self.blacklist_rel)
        return False

    def _stuck(self) -> bool:
        if len(self.history) < self.takeover_frames:
            return False
        pts = np.stack(list(self.history)[-self.takeover_frames:])
        return float(np.ptp(pts, axis=0).max()) < self.takeover_px

    # ----- Kalman filter -------------------------------------------------------
    def _kf_init(self, c: np.ndarray) -> None:
        self.kf_x = np.array([c[0], c[1], 0.0, 0.0], dtype=np.float64)
        self.kf_P = np.diag([self.r_meas ** 2, self.r_meas ** 2, 400.0, 400.0])
        self.misses = 0

    def _kf_predict(self) -> None:
        """One processed frame ahead: constant velocity plus gravity on vy."""
        if self.kf_x is None:
            return
        F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.kf_x = F @ self.kf_x + np.array([0.0, 0.5 * self.gravity_px, 0.0, self.gravity_px])
        Q = np.diag([self.q ** 2, self.q ** 2, (2 * self.q) ** 2, (2 * self.q) ** 2])
        self.kf_P = F @ self.kf_P @ F.T + Q

    def _kf_update(self, c: np.ndarray) -> None:
        if self.kf_x is None:
            self._kf_init(c)
            return
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        R = np.eye(2) * self.r_meas ** 2
        S = H @ self.kf_P @ H.T + R
        K = self.kf_P @ H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + K @ (np.asarray(c, dtype=np.float64) - H @ self.kf_x)
        self.kf_P = (np.eye(4) - K @ H) @ self.kf_P
        self.misses = 0

    def predict(self, gap: int = 0) -> np.ndarray | None:
        """Predicted ball position for the current frame (after begin_step)."""
        if self.kf_x is None:
            return None
        return self.kf_x[:2].astype(np.float32)

    @property
    def velocity(self) -> np.ndarray | None:
        return None if self.kf_x is None else self.kf_x[2:].astype(np.float32)

    @property
    def gap(self) -> int:
        return self.step_no - self.last_step

    # ----- one processed frame -------------------------------------------------
    def begin_step(self, hoop: list[float] | None) -> None:
        """Call once per processed frame, before pick()."""
        self.step_no += 1
        self.rejects = []
        if self.kf_x is not None:
            self._kf_predict()
            if self.misses >= self.coast_frames:
                self.kf_x, self.kf_P = None, None  # coast expired: lost
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
             hoop: list[float] | None = None, players=()) -> tuple[float, list[float], bool] | None:
        """candidates = [(conf, [x1,y1,x2,y2])], players = player boxes; may be
        called more than once per step (crop re-detection). Returns
        (conf, box, predicted) for an accepted detection, (0, box, True) while
        coasting on the prediction, None when lost."""
        hoop_c = _center(hoop) if hoop else None
        hoop_w = max(float(hoop[2] - hoop[0]), 1.0) if hoop else 1.0
        pred = self.predict()
        stuck = self._stuck()
        gate = float("inf") if stuck else float("inf")  # the KF gate below replaces the old distance gate
        near = self.near_px + self.near_grow_px * min(self.misses, 10)

        tier1, tier2 = [], []
        for conf, box in candidates:
            c = _center(box)
            if hoop_c is not None:
                off = (c - hoop_c) / hoop_w
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
            if self._bad_radius(box, players):
                self._reject(box, conf, "radius")
                continue
            if self._is_head(c, conf, players):
                self._reject(box, conf, "head")
                continue
            on_track = pred is not None and float(np.linalg.norm(c - pred)) <= near
            if not on_track and self._is_wall(c, hoop, players):
                self._reject(box, conf, "wall")
                continue
            if pred is None or stuck:
                tier2.append((conf, conf, box, c))  # lost or parked: confidence decides
                continue
            dist_pred = float(np.linalg.norm(c - pred))
            if dist_pred > near:
                self._reject(box, conf, "far")  # prediction first: coast, never jump
                continue
            tier1.append((conf - 0.5 * dist_pred / near, conf, box, c))
        pool = tier1 or tier2
        if not pool:
            if pred is not None:
                self.misses += 1
                self.coasted += 1
                r = float(np.median(self.radii)) if self.radii else 15.0
                box = [float(pred[0] - r), float(pred[1] - r), float(pred[0] + r), float(pred[1] + r)]
                return 0.0, box, True  # coasting on the prediction
            return None
        _s, conf, box, c = max(pool, key=lambda x: x[0])

        self.history.append(c)
        if len(self.history) == self.static_frames:
            pts = np.stack(self.history)
            if np.ptp(pts, axis=0).max() < self.static_px:
                self._add_abs(pts.mean(axis=0))
                if hoop_c is not None:
                    self.blacklist_rel.append((pts.mean(axis=0) - hoop_c) / hoop_w)
                self.history.clear()
                self.last, self.last_step, self.prev = None, -10**9, None
                self.kf_x, self.kf_P = None, None
                self._reject(box, conf, "static_abs")
                return None
        if self._near_blacklist(c, hoop_c, hoop_w):
            self.accepted_near_blacklist += 1
            self.counts["accepted_near_blacklist"] += 1
        if stuck and self.last is not None and float(np.linalg.norm(c - self.last)) > 60:
            self._kf_init(c)  # takeover: the old track was a parked object
        else:
            self._kf_update(c)
        self.prev, self.prev_step = self.last, self.last_step
        self.last, self.last_step = c, self.step_no
        self.last_known = c
        self.radii.append(max(box[2] - box[0], box[3] - box[1]) / 2)
        return conf, box, False

    def summary(self) -> dict:
        return {"ball_rejected": {k: v for k, v in self.counts.items() if k != "accepted_near_blacklist"},
                "ball_coasted_frames": self.coasted,
                "ball_accepted_near_blacklist": self.counts["accepted_near_blacklist"],
                "ball_blacklist_abs": [b.round(1).tolist() for b in self.blacklist_abs],
                "ball_blacklist_rel_hoopwidths": [b.round(2).tolist() for b in self.blacklist_rel]}
