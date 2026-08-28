"""Shot detection: attempts, made / missed, and who shot.

Rule (docs/ORCHESTRATION.md, STATS milestone 12:45), adapted to sparse tracks
(10-25 fps, gappy ball) after comparing with reference/thirdparty/utils.py
(Avi Shah's detect_up / detect_down / score for the same hoop model):

* "up": the ball is seen above the rim line inside a zone around the hoop
  (`zone_width_scale` x hoop width, up to `zone_above_scale` hoop heights
  above the rim). The ball must have shown an up-then-down arc in the last
  `arc_window_s` (skipped when there are too few samples to judge).
* attempt: within `attempt_window_s` after an "up" sighting the ball is seen
  below the rim line ("down"). Event time = last sample above the rim.
* made: (a) a sample lies in the net (inside the hoop x-range, between
  0.5 and 2.0 hoop heights below the rim), or (b) the straight line from the
  last sample above the rim to the first below it crosses the rim line inside
  the central 80 % of the rim (reference `score()`). Otherwise miss.
* shooter: holder of the last possession before the "up" sighting; release =
  first frame in which the ball is more than one bbox width from his bbox.

The camera pans, so every test is done in hoop-relative coordinates (ball
minus hoop center in the *same* frame) and only in frames that contain a
hoop box. Frames without a hoop box never advance the state machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .io import BBox, Frame, Point
from .possession import Possession, PossessionResult


@dataclass
class ShotParams:
    zone_width_scale: float = 4.0  # "up" zone width = scale * hoop width, centred on the hoop
    zone_above_scale: float = 4.0  # "up" zone reaches this many hoop heights above the rim (10 fps: ball moves ~1 h per sample)
    rim_frac: float = 0.0  # rim line = y1 + rim_frac * hoop height (TRACK: box = rim + net)
    net_depth: tuple[float, float] = (0.5, 2.0)  # "in the net" = this many hoop heights below the rim (generous: sparse samples)
    rim_inner_frac: float = 0.8  # central share of the rim width that counts as "through"
    arc_window_s: float = 1.0
    arc_min_rise_hoops: float = 1.0  # ball must have climbed this many hoop heights before the apex
    arc_min_samples: int = 3  # fewer samples in the window: arc test is skipped, not failed
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


# --- detection ------------------------------------------------------------------


def detect_shots(
    frames: list[Frame],
    possession: PossessionResult,
    params: ShotParams = ShotParams(),
) -> list[ShotEvent]:
    ball_idx = [i for i, fr in enumerate(frames) if fr.ball is not None]
    shots: list[ShotEvent] = []
    up_k: int | None = None  # ball_idx position of the first "up" sighting of the episode
    last_above_k: int | None = None  # last sample above the rim since "up"
    last_event_t = -math.inf

    for k, i in enumerate(ball_idx):
        fr = frames[i]
        if not fr.hoops:
            continue
        ball = fr.ball.center
        hoop = fr.hoops[0]

        if up_k is not None:
            up_fr = frames[ball_idx[up_k]]
            same_hoop = match_hoop(fr, up_fr.hoops[0], params) is not None
            if not same_hoop or fr.t - up_fr.t > params.attempt_window_s:
                up_k = last_above_k = None  # episode expired (or the camera moved on)

        if up_k is None:
            if fr.t - last_event_t < params.cooldown_s:
                continue
            if in_up_zone(ball, hoop, params) and _arc_ok(frames, ball_idx, k, hoop, params):
                up_k = last_above_k = k
            continue

        if above_rim(ball, hoop, params):
            last_above_k = k
            continue

        # "down": below the rim after "up" -> attempt
        rim_k = last_above_k if last_above_k is not None else up_k
        rim_fr = frames[ball_idx[rim_k]]
        rim_hoop = rim_fr.hoops[0]
        made = _made(frames, ball_idx, rim_k, k, rim_hoop, params)
        shooter, foot, release, confirmed = _shooter(frames, possession, ball_idx[up_k], params)
        shots.append(
            ShotEvent(
                frame=rim_fr.frame,
                t=rim_fr.t,
                player_id=shooter.player_id if shooter else None,
                team=shooter.team if shooter else -1,
                made=made,
                shooter_foot=foot,
                hoop_bbox=rim_hoop,
                shooter_confirmed=confirmed,
                release_frame=release,
            )
        )
        last_event_t = fr.t
        up_k = last_above_k = None
    return shots


def _rel_y(fr: Frame, hoop: BBox, p: ShotParams) -> float | None:
    h = match_hoop(fr, hoop, p)
    return None if h is None else fr.ball.center[1] - hoop_center(h)[1]


def _arc_ok(frames: list[Frame], ball_idx: list[int], k: int, hoop: BBox, p: ShotParams) -> bool:
    """Up-then-down (or still rising towards the apex) arc in hoop-relative y
    over the last `arc_window_s`: the ball must have climbed at least
    `arc_min_rise_hoops` hoop heights. A pass or dribble never does."""
    fr = frames[ball_idx[k]]
    rel: list[float] = []
    for j in range(k, -1, -1):
        f = frames[ball_idx[j]]
        if fr.t - f.t > p.arc_window_s:
            break
        y = _rel_y(f, hoop, p)
        if y is not None:
            rel.append(y)
    if len(rel) < p.arc_min_samples:
        return True  # too little information to reject
    rel.reverse()
    apex = min(rel)
    hoop_h = max(hoop[3] - hoop[1], 1.0)
    return (max(rel) - apex) >= p.arc_min_rise_hoops * hoop_h


def _made(
    frames: list[Frame], ball_idx: list[int], rim_k: int, down_k: int, hoop: BBox, p: ShotParams
) -> bool:
    above = frames[ball_idx[rim_k]]
    below = frames[ball_idx[down_k]]
    below_hoop = match_hoop(below, hoop, p) or hoop
    # (b) straight line from the last sample above the rim to the first below it
    if crosses_rim(_rel(above, hoop), _rel(below, below_hoop), (0.0, 0.0, hoop[2] - hoop[0], hoop[3] - hoop[1]), p):
        return True
    # (a) any sample in the net shortly after
    t_down = below.t
    for j in range(rim_k + 1, len(ball_idx)):
        f = frames[ball_idx[j]]
        if f.t - t_down > p.made_window_s:
            break
        h = match_hoop(f, hoop, p) or hoop
        if in_net(f.ball.center, h, p):
            return True
    return False


def _rel(fr: Frame, hoop: BBox) -> Point:
    """Ball position relative to the hoop's top-left corner (pan-proof)."""
    return (fr.ball.center[0] - hoop[0], fr.ball.center[1] - hoop[1])


def _shooter(
    frames: list[Frame], possession: PossessionResult, up_index: int, p: ShotParams
) -> tuple[Possession | None, Point | None, int | None, bool]:
    t_up = frames[up_index].t
    candidates = [
        s for s in possession.segments if s.start_t < t_up and s.end_t >= t_up - p.shooter_lookback_s
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
