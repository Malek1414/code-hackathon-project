"""Shot detection: attempts, made / missed, and who shot.

Rule (docs/ORCHESTRATION.md, STATS milestone 12:45):

* Shot candidate: the ball center enters the hoop zone (hoop bbox grown to
  2x its width, extended 1.5x its height above the rim) from above (ball
  above the rim line, moving down), and the ball showed an up-then-down arc
  during the last `arc_window_s`.
* Made: within `made_window_s` after entry the ball is seen below the rim
  line, inside the hoop's x-range. Otherwise a miss.
* Shooter: holder of the last possession before the ball left him. Release =
  first frame in which the ball is more than one bbox width away from him.

The camera pans, so every test is done in hoop-relative coordinates (ball
minus hoop center in the *same* frame) and only in frames that contain a
hoop box. Frames without a hoop never start a shot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .io import BBox, Frame, Player, Point
from .possession import Possession, PossessionResult


@dataclass
class ShotParams:
    zone_width_scale: float = 2.0  # zone width = scale * hoop width, centred on the hoop
    zone_above_scale: float = 1.5  # zone reaches this many hoop heights above the rim
    rim_frac: float = 0.0  # rim line = y1 + rim_frac * hoop height
    made_depth_frac: float = 0.5  # "below the rim" = deeper than this fraction of hoop height
    arc_window_s: float = 1.0
    arc_min_rise_hoops: float = 1.0  # ball must have climbed at least this many hoop heights before the apex
    arc_min_samples: int = 3  # fewer samples in the window: arc test is skipped, not failed
    made_window_s: float = 0.5
    cooldown_s: float = 1.5  # one attempt per ... after a candidate (rim rattles)
    shooter_lookback_s: float = 4.0
    release_width_scale: float = 1.0
    hoop_match_widths: float = 1.5  # same hoop in adjacent frames if centers are this close


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


def in_zone(ball: Point, h: BBox, p: ShotParams) -> bool:
    w, hh = h[2] - h[0], h[3] - h[1]
    cx = (h[0] + h[2]) / 2
    half = p.zone_width_scale * w / 2
    top = rim_y(h, p) - p.zone_above_scale * hh
    return cx - half <= ball[0] <= cx + half and top <= ball[1] <= h[3]


def above_rim(ball: Point, h: BBox, p: ShotParams) -> bool:
    return ball[1] < rim_y(h, p)


def below_rim_in_net(ball: Point, h: BBox, p: ShotParams) -> bool:
    depth = rim_y(h, p) + p.made_depth_frac * (h[3] - h[1])
    return h[0] <= ball[0] <= h[2] and ball[1] > depth


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
    last_candidate_t = -math.inf

    for k, i in enumerate(ball_idx):
        fr = frames[i]
        if not fr.hoops or fr.t - last_candidate_t < params.cooldown_s:
            continue
        ball = fr.ball.center
        for hoop in fr.hoops:
            if not in_zone(ball, hoop, params) or not above_rim(ball, hoop, params):
                continue
            if k == 0:
                continue
            prev = frames[ball_idx[k - 1]]
            prev_hoop = match_hoop(prev, hoop, params) or hoop
            prev_rel_y = prev.ball.center[1] - hoop_center(prev_hoop)[1]
            rel_y = ball[1] - hoop_center(hoop)[1]
            if in_zone(prev.ball.center, prev_hoop, params) or rel_y <= prev_rel_y:
                continue  # already inside, or not moving down
            if not _arc_ok(frames, ball_idx, k, hoop, params):
                continue

            last_candidate_t = fr.t
            made = _made(frames, ball_idx, k, hoop, params)
            shooter, foot, release, confirmed = _shooter(frames, possession, i, params)
            shots.append(
                ShotEvent(
                    frame=fr.frame,
                    t=fr.t,
                    player_id=shooter.player_id if shooter else None,
                    team=shooter.team if shooter else -1,
                    made=made,
                    shooter_foot=foot,
                    hoop_bbox=hoop,
                    shooter_confirmed=confirmed,
                    release_frame=release,
                )
            )
            break
    return shots


def _arc_ok(frames: list[Frame], ball_idx: list[int], k: int, hoop: BBox, p: ShotParams) -> bool:
    """Up-then-down arc in hoop-relative y over the last `arc_window_s`."""
    fr = frames[ball_idx[k]]
    rel: list[float] = []
    for j in range(k, -1, -1):
        f = frames[ball_idx[j]]
        if fr.t - f.t > p.arc_window_s:
            break
        h = match_hoop(f, hoop, p)
        if h is None:
            continue
        rel.append(f.ball.center[1] - hoop_center(h)[1])
    rel.reverse()
    if len(rel) < p.arc_min_samples:
        return True  # too little information to reject
    apex = min(range(len(rel)), key=lambda n: rel[n])
    before = rel[: apex + 1]
    hoop_h = max(hoop[3] - hoop[1], 1.0)
    rose = (max(before) - rel[apex]) >= p.arc_min_rise_hoops * hoop_h
    descending = rel[-1] > rel[apex]
    return rose and descending


def _made(frames: list[Frame], ball_idx: list[int], k: int, hoop: BBox, p: ShotParams) -> bool:
    t0 = frames[ball_idx[k]].t
    for j in range(k + 1, len(ball_idx)):
        f = frames[ball_idx[j]]
        if f.t - t0 > p.made_window_s:
            break
        h = match_hoop(f, hoop, p) or hoop
        if below_rim_in_net(f.ball.center, h, p):
            return True
    return False


def _shooter(
    frames: list[Frame], possession: PossessionResult, entry_index: int, p: ShotParams
) -> tuple[Possession | None, Point | None, int | None, bool]:
    t_entry = frames[entry_index].t
    candidates = [
        s
        for s in possession.segments
        if s.start_t < t_entry and s.end_t >= t_entry - p.shooter_lookback_s
    ]
    if not candidates:
        return None, None, None, False
    seg = max(candidates, key=lambda s: s.start_t)

    release_index = None
    last_held = None
    for i in range(seg.start_index, entry_index):
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
