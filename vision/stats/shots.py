"""Shot detection: attempts, made / missed, and who shot. Incremental.

Rule (docs/ORCHESTRATION.md, STATS milestone 12:45), adapted to sparse tracks
(10-25 fps, gappy ball) after comparing with reference/thirdparty/utils.py
(Avi Shah's detect_up / detect_down / score for the same hoop model):

* "up": the ball is seen above the rim line inside a zone around the hoop
  (`zone_width_scale` x hoop width, up to `zone_above_scale` hoop heights
  above the rim). Over the last `arc_window_s` the ball must have MOVED
  (hoop-relative extent >= `min_move_widths` hoop widths: static false balls
  on wall fixtures never count) and shown a rise of `arc_min_rise_hoops`
  hoop heights before its apex.
* attempt: within `attempt_window_s` after an "up" sighting the ball is seen
  below the rim line ("down"). Event time = last sample above the rim.
* made: (a) a sample lies in the net (inside the hoop x-range, between
  0.5 and 2.0 hoop heights below the rim) until `made_window_s` after "down",
  or (b) the straight line from the last sample above the rim to the first
  below it crosses the rim line inside the central 80 % of the rim
  (reference `score()`). Otherwise miss. (a) means an event is final at most
  `made_window_s` after the ball dropped.
* shooter: holder of the last possession before the "up" sighting; release =
  first frame in which the ball is more than one bbox width from his bbox.

The camera pans, so every test is done in hoop-relative coordinates (ball
minus hoop in the *same* frame) and only in frames that contain a hoop box.
Frames without a hoop box never advance the state machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .io import BBox, Frame, Point
from .possession import Possession, PossessionResult, PossessionTracker


@dataclass
class ShotParams:
    zone_width_scale: float = 4.0  # "up" zone width = scale * hoop width, centred on the hoop
    zone_above_scale: float = 4.0  # "up" zone reaches this many hoop heights above the rim (10 fps: ball moves ~1 h per sample)
    rim_frac: float = 0.0  # rim line = y1 + rim_frac * hoop height (TRACK: box = rim + net)
    net_depth: tuple[float, float] = (0.5, 2.0)  # "in the net" = this many hoop heights below the rim (generous: sparse samples)
    rim_inner_frac: float = 0.8  # central share of the rim width that counts as "through"
    arc_window_s: float = 1.0
    arc_min_rise_hoops: float = 1.0  # ball must have climbed this many hoop heights before the apex
    arc_min_samples: int = 3  # fewer samples in the window: the rise test is skipped, not failed
    min_move_widths: float = 0.5  # hoop-relative extent over the window; static "balls" never count
    attempt_window_s: float = 1.5  # "down" must follow "up" within this
    made_window_s: float = 0.5  # net samples are accepted until this long after "down"
    cooldown_s: float = 1.5  # one attempt per ... (rim rattles)
    shooter_lookback_s: float = 4.0
    release_width_scale: float = 1.0
    hoop_match_widths: float = 1.5  # same hoop in two frames if centers are this close


@dataclass
class ShotEvent:
    frame: int
    t: float
    player_id: int | None
    team: int
    made: bool
    shooter_foot: Point | None
    hoop_bbox: BBox
    shooter_confirmed: bool
    release_frame: int | None = None
    decided_t: float | None = None  # when the made/miss verdict became final (live latency)

    def to_dict(self) -> dict:
        return {
            "t": round(self.t, 3),
            "frame": self.frame,
            "player_id": self.player_id,
            "team": self.team,
            "made": self.made,
            "shooter_foot": [round(v, 1) for v in self.shooter_foot] if self.shooter_foot else None,
            "hoop_bbox": [round(v, 1) for v in self.hoop_bbox],
            "shooter_confirmed": self.shooter_confirmed,
            "release_frame": self.release_frame,
        }


# --- hoop geometry (all relative to the hoop of the same frame) ---------------


def hoop_center(h: BBox) -> Point:
    return ((h[0] + h[2]) / 2, (h[1] + h[3]) / 2)


def rim_y(h: BBox, p: ShotParams) -> float:
    return h[1] + p.rim_frac * (h[3] - h[1])


def in_up_zone(ball: Point, h: BBox, p: ShotParams) -> bool:
    w, hh = h[2] - h[0], h[3] - h[1]
    cx = (h[0] + h[2]) / 2
    half = p.zone_width_scale * w / 2
    rim = rim_y(h, p)
    return cx - half <= ball[0] <= cx + half and rim - p.zone_above_scale * hh <= ball[1] < rim


def above_rim(ball: Point, h: BBox, p: ShotParams) -> bool:
    return ball[1] < rim_y(h, p)


def in_net(ball: Point, h: BBox, p: ShotParams) -> bool:
    hh = h[3] - h[1]
    rim = rim_y(h, p)
    lo, hi = rim + p.net_depth[0] * hh, rim + p.net_depth[1] * hh
    return h[0] <= ball[0] <= h[2] and lo < ball[1] <= hi


def crosses_rim(above: Point, below: Point, h: BBox, p: ShotParams) -> bool:
    """Does the segment above→below cross the rim line inside the rim?"""
    rim = rim_y(h, p)
    dy = below[1] - above[1]
    if dy <= 0:
        return False
    s = (rim - above[1]) / dy
    x = above[0] + (below[0] - above[0]) * s
    cx, w = (h[0] + h[2]) / 2, h[2] - h[0]
    return abs(x - cx) <= p.rim_inner_frac * w / 2


def match_hoop(fr: Frame, ref: BBox, p: ShotParams) -> BBox | None:
    """The hoop in `fr` that corresponds to `ref` (nearest center), if any."""
    if not fr.hoops:
        return None
    rc = hoop_center(ref)
    limit = p.hoop_match_widths * max(ref[2] - ref[0], 20.0)
    best, best_d = None, math.inf
    for h in fr.hoops:
        c = hoop_center(h)
        d = math.hypot(c[0] - rc[0], c[1] - rc[1])
        if d < best_d:
            best, best_d = h, d
    return best if best_d <= limit else None


def _rel(fr: Frame, hoop: BBox) -> Point:
    """Ball position relative to the hoop's top-left corner (pan-proof)."""
    return (fr.ball.center[0] - hoop[0], fr.ball.center[1] - hoop[1])


# --- detector -------------------------------------------------------------------


@dataclass
class _Pending:
    event: ShotEvent
    hoop: BBox
    deadline_t: float


class ShotDetector:
    """State machine fed with frame indices in order (the frames live in the
    shared PossessionTracker). `push` returns the events that became final."""

    def __init__(self, possession: PossessionTracker, params: ShotParams = ShotParams()) -> None:
        self.p = params
        self.possession = possession
        self.shots: list[ShotEvent] = []
        self._ball_idx: list[int] = []  # frame indices with a ball
        self._up_k: int | None = None  # position in _ball_idx of the first "up" sighting
        self._last_above_k: int | None = None
        self._last_event_t = -math.inf
        self._pending: _Pending | None = None

    @property
    def frames(self) -> list[Frame]:
        return self.possession.frames

    def push(self, index: int) -> list[ShotEvent]:
        fr = self.frames[index]
        if fr.ball is None:
            return []
        self._ball_idx.append(index)
        k = len(self._ball_idx) - 1
        p = self.p
        done: list[ShotEvent] = []

        if self._pending is not None:
            done += self._check_pending(fr)

        if not fr.hoops:
            return done
        ball = fr.ball.center
        hoop = fr.hoops[0]

        if self._up_k is not None:
            up_fr = self.frames[self._ball_idx[self._up_k]]
            same_hoop = match_hoop(fr, up_fr.hoops[0], p) is not None
            if not same_hoop or fr.t - up_fr.t > p.attempt_window_s:
                self._up_k = self._last_above_k = None  # episode expired (or the camera moved on)

        if self._up_k is None:
            if fr.t - self._last_event_t < p.cooldown_s:
                return done
            if in_up_zone(ball, hoop, p) and self._arc_ok(k, hoop):
                self._up_k = self._last_above_k = k
            return done

        if above_rim(ball, hoop, p):
            self._last_above_k = k
            return done

        # "down": below the rim after "up" -> attempt
        rim_k = self._last_above_k if self._last_above_k is not None else self._up_k
        rim_fr = self.frames[self._ball_idx[rim_k]]
        rim_hoop = rim_fr.hoops[0]
        shooter, foot, release, confirmed = self._shooter(self._ball_idx[self._up_k])
        event = ShotEvent(
            frame=rim_fr.frame,
            t=rim_fr.t,
            player_id=shooter.player_id if shooter else None,
            team=shooter.team if shooter else -1,
            made=False,
            shooter_foot=foot,
            hoop_bbox=rim_hoop,
            shooter_confirmed=confirmed,
            release_frame=release,
        )
        self._last_event_t = fr.t
        self._up_k = self._last_above_k = None

        local = (0.0, 0.0, rim_hoop[2] - rim_hoop[0], rim_hoop[3] - rim_hoop[1])
        below_hoop = match_hoop(fr, rim_hoop, p) or rim_hoop
        if crosses_rim(_rel(rim_fr, rim_hoop), _rel(fr, below_hoop), local, p) or in_net(ball, below_hoop, p):
            event.made = True
            event.decided_t = fr.t
            self.shots.append(event)
            done.append(event)
        else:
            self._pending = _Pending(event=event, hoop=rim_hoop, deadline_t=fr.t + p.made_window_s)
        return done

    def finish(self) -> list[ShotEvent]:
        """End of stream: a pending verdict becomes a miss."""
        if self._pending is None:
            return []
        ev = self._pending.event
        ev.decided_t = self.frames[-1].t if self.frames else ev.t
        self._pending = None
        self.shots.append(ev)
        return [ev]

    def _check_pending(self, fr: Frame) -> list[ShotEvent]:
        pend = self._pending
        ev = pend.event
        if fr.t > pend.deadline_t:
            ev.decided_t = fr.t
            self._pending = None
            self.shots.append(ev)
            return [ev]
        h = match_hoop(fr, pend.hoop, self.p) or pend.hoop
        if in_net(fr.ball.center, h, self.p):
            ev.made = True
            ev.decided_t = fr.t
            self._pending = None
            self.shots.append(ev)
            return [ev]
        return []

    def _arc_ok(self, k: int, hoop: BBox) -> bool:
        """The ball moved (not a wall fixture) and climbed towards an apex
        over the last `arc_window_s`, in hoop-relative coordinates."""
        p = self.p
        fr = self.frames[self._ball_idx[k]]
        rel: list[Point] = []
        for j in range(k, -1, -1):
            f = self.frames[self._ball_idx[j]]
            if fr.t - f.t > p.arc_window_s:
                break
            h = match_hoop(f, hoop, p)
            if h is not None:
                rel.append(_rel(f, h))
        if len(rel) < 2:
            return False  # a single sighting is never a shot
        hoop_w = max(hoop[2] - hoop[0], 1.0)
        hoop_h = max(hoop[3] - hoop[1], 1.0)
        xs, ys = [r[0] for r in rel], [r[1] for r in rel]
        extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if extent < p.min_move_widths * hoop_w:
            return False
        if len(rel) < p.arc_min_samples:
            return True  # too little information to judge the rise
        return (max(ys) - min(ys)) >= p.arc_min_rise_hoops * hoop_h

    def _shooter(self, up_index: int) -> tuple[Possession | None, Point | None, int | None, bool]:
        p = self.p
        frames = self.frames
        t_up = frames[up_index].t
        candidates = [
            s for s in self.possession.segments if s.start_t < t_up and s.end_t >= t_up - p.shooter_lookback_s
        ]
        if not candidates:
            return None, None, None, False
        seg = max(candidates, key=lambda s: s.start_t)

        release_index = None
        last_held = None
        for i in range(seg.start_index, up_index):
            fr = frames[i]
            pl = fr.player(seg.player_id)
            if pl is None or fr.ball is None:
                continue
            if pl.edge_distance(fr.ball.center) <= p.release_width_scale * pl.width:
                last_held = i
            elif last_held is not None:
                release_index = i
                break
        if release_index is None:
            release_index = last_held if last_held is not None else seg.end_index

        foot = _foot_at(frames, seg.player_id, release_index, seg.start_index)
        return seg, foot, frames[release_index].frame, True


def _foot_at(frames: list[Frame], pid: int, index: int, floor: int) -> Point | None:
    for i in range(index, floor - 1, -1):
        pl = frames[i].player(pid)
        if pl is not None:
            return pl.foot
    return None


def detect_shots(
    frames: list[Frame],
    possession: PossessionResult | PossessionTracker,
    params: ShotParams = ShotParams(),
) -> list[ShotEvent]:
    """Batch wrapper: runs the incremental detector over already-tracked frames."""
    if isinstance(possession, PossessionResult):
        tracker = PossessionTracker()
        tracker.frames = list(frames)
        tracker.holder = list(possession.holder)
    else:
        tracker = possession
    det = ShotDetector(tracker, params)
    for i in range(len(tracker.frames)):
        det.push(i)
    det.finish()
    return det.shots
