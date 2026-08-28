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
* vanished: the ball was last seen near the rim (`near_rim_*`) and then not
  at all for `vanish_window_s` (typical on dev60: the detector loses the ball
  at the ring). The attempt counts as a miss with `made_confirmed` False;
  `made_hint` says whether the last two samples extrapolate into the
  central 50 % of the rim. Only a hint: from a side camera the 2D crossing
  cannot tell a swish from a front-iron rattle-out (dev60 57 s: crossing
  0.1 widths from the center, rattled out, verified on video).
* shooter: the flight is followed backwards from the "up" sighting to its
  first sample (chain of plausible moves, strays skipped) and extrapolated
  `release_back_samples` further back to the hands; the player whose bbox
  (upper `release_box_top_frac`, widened `release_box_widen` per side for the
  arms) contains that release point is the shooter, else the player whose
  widened box contains the first flight sample (the ball left his box). QA on dev60: the nearest
  foot to the first flight sample was a bystander at the lane, the real
  shooter stood alone at the free-throw line. Fallback when no box contains
  the point: holder of the last possession (nearest foot), with
  `shooter_confirmed` False.

The camera pans, so every test is done in hoop-relative coordinates (ball
minus hoop in the *same* frame) and only in frames that contain a hoop box.
Frames without a hoop box never advance the state machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .io import BBox, Frame, Player, Point
from .possession import Possession, PossessionResult, PossessionTracker


@dataclass
class ShotParams:
    zone_width_scale: float = 4.0  # "up" zone width = scale * hoop width, centred on the hoop
    zone_above_scale: float = 4.0  # "up" zone reaches this many hoop heights above the rim (10 fps: ball moves ~1 h per sample)
    rim_frac: float = 0.0  # rim line = y1 + rim_frac * hoop height (TRACK: box = rim + net)
    net_depth: tuple[float, float] = (0.5, 2.0)  # "in the net" = this many hoop heights below the rim (generous: sparse samples)
    rim_inner_frac: float = 0.8  # central share of the rim width that counts as "through"
    rim_inner_frac_extrapolated: float = 0.5  # stricter when the crossing is only extrapolated (rim hits rattle out)
    arc_window_s: float = 1.0
    arc_min_rise_hoops: float = 1.0  # ball must have climbed this many hoop heights before the apex
    arc_min_samples: int = 3  # fewer samples in the window: the rise test is skipped, not failed
    min_move_widths: float = 0.5  # hoop-relative extent over the window; static "balls" never count
    attempt_window_s: float = 1.5  # "down" must follow "up" within this
    made_window_s: float = 0.5  # net samples are accepted until this long after "down"
    vanish_window_s: float = 0.6  # ball unseen this long after being near the rim: attempt, verdict extrapolated
    near_rim_widths: float = 1.5  # "near the rim" = within this many hoop widths of the hoop center ...
    near_rim_heights: float = 2.5  # ... and at most this many hoop heights above the rim
    cooldown_s: float = 1.5  # one attempt per ... (rim rattles)
    shooter_lookback_s: float = 4.0
    release_width_scale: float = 1.0
    release_chain_gap_s: float = 0.8  # flight samples further apart than this do not chain (best.pt run: 0.52 s hole)
    release_back_samples: float = 2.0  # extrapolate the flight this many sample gaps back to the hands ...
    release_back_max_s: float = 0.12  # ... but never further back than this (gappy tracks)
    release_box_top_frac: float = 0.6  # the release point must lie in the upper part of the shooter's box ...
    release_box_widen: float = 0.3  # ... widened by this share on each side (arms)
    release_max_skips: int = 4  # stray samples tolerated inside the flight chain
    release_max_speed_diam_s: float = 90.0  # a real shot stays under ~70 ball diameters per second; static-junk jumps are >100
    release_min_rise_diam_s: float = 5.0  # slower vertical motion than this is the ball in the hands, not in flight
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
    made_confirmed: bool = True  # False: ball vanished at the rim, result unknown (counted as miss)
    made_hint: bool | None = None  # vanished attempts: did the extrapolated arc aim at the middle of the rim?
    team_source: str = "unknown"  # identity | track_majority | nearby_players | possession | unknown

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
            "made_confirmed": self.made_confirmed,
            "made_hint": self.made_hint,
            "team_source": self.team_source,
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


def crosses_rim(above: Point, below: Point, h: BBox, p: ShotParams, inner_frac: float | None = None) -> bool:
    """Does the line above→below cross the rim line inside the central
    `inner_frac` of the rim? (Extended beyond `below` if needed.)"""
    rim = rim_y(h, p)
    dy = below[1] - above[1]
    if dy <= 0:
        return False
    s = (rim - above[1]) / dy
    x = above[0] + (below[0] - above[0]) * s
    cx, w = (h[0] + h[2]) / 2, h[2] - h[0]
    frac = p.rim_inner_frac if inner_frac is None else inner_frac
    return abs(x - cx) <= frac * w / 2


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
        self._floor_k = 0  # _ball_idx position of the last cut: never look back past it
        self._floor_index = 0  # frame-list index of the last cut

    @property
    def frames(self) -> list[Frame]:
        return self.possession.frames

    def reset(self) -> None:
        """Cut in the footage: drop the episode and never look back past here."""
        self._up_k = self._last_above_k = None
        self._pending = None
        self._last_event_t = -math.inf
        self._floor_k = len(self._ball_idx)
        self._floor_index = len(self.frames)

    def push(self, index: int) -> list[ShotEvent]:
        fr = self.frames[index]
        vanished = self._check_vanished(fr.t)
        if fr.ball is None:
            return vanished
        self._ball_idx.append(index)
        k = len(self._ball_idx) - 1
        p = self.p
        done: list[ShotEvent] = []

        done += vanished
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
        pid, team, foot, release, confirmed, team_source = self._shooter_any(self._up_k)
        event = ShotEvent(
            frame=rim_fr.frame,
            t=rim_fr.t,
            player_id=pid,
            team=team,
            made=False,
            shooter_foot=foot,
            hoop_bbox=rim_hoop,
            shooter_confirmed=confirmed,
            release_frame=release,
            team_source=team_source,
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
        """End of stream: a pending verdict becomes a miss; a ball that was
        near the rim when the stream ended counts as a vanished attempt."""
        done: list[ShotEvent] = []
        if self._up_k is not None and self._last_above_k is not None:
            done += self._vanished_attempt(self.frames[-1].t)
        if self._pending is not None:
            ev = self._pending.event
            ev.decided_t = self.frames[-1].t if self.frames else ev.t
            self._pending = None
            self.shots.append(ev)
            done.append(ev)
        return done

    def _check_vanished(self, t_now: float) -> list[ShotEvent]:
        if self._up_k is None or self._last_above_k is None:
            return []
        la = self.frames[self._ball_idx[self._last_above_k]]
        if t_now - la.t <= self.p.vanish_window_s:
            return []
        return self._vanished_attempt(t_now)

    def _vanished_attempt(self, t_now: float) -> list[ShotEvent]:
        """The ball has not been seen since `last_above`: attempt if it was
        near the rim, verdict by extrapolating the last two samples."""
        p = self.p
        la_k = self._last_above_k
        la = self.frames[self._ball_idx[la_k]]
        hoop = la.hoops[0]
        w, hh = hoop[2] - hoop[0], hoop[3] - hoop[1]
        cx = (hoop[0] + hoop[2]) / 2
        bx, by = la.ball.center
        near = abs(bx - cx) <= p.near_rim_widths * w and rim_y(hoop, p) - by <= p.near_rim_heights * hh
        if not near:
            self._up_k = self._last_above_k = None
            return []
        hint = None
        if la_k - 1 >= self._floor_k:
            prev = self.frames[self._ball_idx[la_k - 1]]
            prev_hoop = match_hoop(prev, hoop, p)
            if prev_hoop is not None and la.t - prev.t <= p.vanish_window_s:
                local = (0.0, 0.0, w, hh)
                a, b = _rel(prev, prev_hoop), _rel(la, hoop)
                if b[1] > a[1]:  # descending: extend the segment to the rim line
                    hint = crosses_rim(a, b, local, p, p.rim_inner_frac_extrapolated)
        pid, team, foot, release, confirmed, team_source = self._shooter_any(self._up_k)
        ev = ShotEvent(
            frame=la.frame,
            t=la.t,
            player_id=pid,
            team=team,
            made=False,
            shooter_foot=foot,
            hoop_bbox=hoop,
            shooter_confirmed=confirmed,
            release_frame=release,
            team_source=team_source,
            decided_t=t_now,
            made_confirmed=False,
            made_hint=hint,
        )
        self._last_event_t = t_now
        self._up_k = self._last_above_k = None
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
        for j in range(k, self._floor_k - 1, -1):
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

    def _shooter_any(self, up_k: int) -> tuple[int | None, int, Point | None, int | None, bool, str]:
        """Release-point rule first, possession rule as the unconfirmed fallback.
        Returns (player id, team, foot, release frame, confirmed, team source)."""
        hit = self._shooter_by_release(up_k)
        if hit is not None:
            return hit
        seg, foot, release, _ = self._shooter(self._ball_idx[up_k])
        if seg is None:
            return None, -1, None, None, False, "unknown"
        return seg.player_id, seg.team, foot, release, False, "possession" if seg.team >= 0 else "unknown"

    def _flight_chain(self, up_k: int) -> list[int]:
        """Ball-sample positions of the flight, earliest first, ending at `up_k`.
        Walking backwards from the rim the ball first climbs to the apex
        (y shrinks), then drops to the hands (y grows); the chain ends where
        the ball was lower than that again (dribble, pass) or a gap is too
        long. Strays (implausible jumps) are skipped."""
        p = self.p
        chain = [up_k]
        skips = 0
        past_apex = False
        j = up_k - 1
        while j >= self._floor_k and skips <= p.release_max_skips:
            cur = self.frames[self._ball_idx[chain[-1]]]
            prev = self.frames[self._ball_idx[j]]
            if cur.t - prev.t > p.release_chain_gap_s:
                break
            dt = max(cur.t - prev.t, 1e-3)
            d = math.hypot(cur.ball.center[0] - prev.ball.center[0], cur.ball.center[1] - prev.ball.center[1])
            diam = 20.0
            if cur.ball.bbox is not None:
                diam = max((cur.ball.bbox[2] - cur.ball.bbox[0] + cur.ball.bbox[3] - cur.ball.bbox[1]) / 2, 4.0)
            if d / dt > p.release_max_speed_diam_s * diam:
                skips += 1
                j -= 1
                continue
            dy = prev.ball.center[1] - cur.ball.center[1]  # > 0: the earlier sample was lower in the image
            rising = dy / dt >= p.release_min_rise_diam_s * diam
            if rising:
                past_apex = True  # clearly on the way up (seen backwards): the ascent has begun
            elif past_apex:
                break  # the ball is no longer rising fast: it was in the hands (or a dribble/pass)
            chain.append(j)
            j -= 1
        chain.reverse()
        return chain

    def _shooter_by_release(self, up_k: int) -> tuple[int | None, int, Point | None, int | None, bool, str] | None:
        p = self.p
        chain = self._flight_chain(up_k)
        if len(chain) < 2:
            return None
        a = self.frames[self._ball_idx[chain[0]]]
        b = self.frames[self._ball_idx[chain[1]]]
        dt = b.t - a.t
        if dt <= 0:
            return None
        vx = (b.ball.center[0] - a.ball.center[0]) / dt
        vy = (b.ball.center[1] - a.ball.center[1]) / dt
        back = min(p.release_back_samples * dt, p.release_back_max_s)
        release = (a.ball.center[0] - vx * back, a.ball.center[1] - vy * back)
        a_index = self._ball_idx[chain[0]]
        fr = None
        for i in range(a_index, max(self._floor_index, a_index - 10) - 1, -1):
            if self.frames[i].players:
                fr = self.frames[i]
                break
        if fr is None:
            return None
        first = a.ball.center
        best, best_rank = None, (9, math.inf)
        for pl in fr.players:
            w = pl.width
            x1, x2 = pl.bbox[0] - p.release_box_widen * w, pl.bbox[2] + p.release_box_widen * w
            top = pl.bbox[1] - 0.15 * pl.height
            upper = pl.bbox[1] + p.release_box_top_frac * pl.height
            cx, cy = pl.center
            if x1 <= release[0] <= x2 and top <= release[1] <= upper:
                rank = (0, math.hypot(release[0] - cx, release[1] - cy))  # release point in the shooting zone
            elif x1 <= first[0] <= x2 and top <= first[1] <= pl.bbox[3]:
                rank = (1, math.hypot(first[0] - cx, first[1] - cy))  # the ball left this player's box
            else:
                continue
            if rank < best_rank:
                best, best_rank = pl, rank
        if best is None:
            return None
        team, source = self._team_of(best.id), "track_majority"
        if team < 0:
            team, source = self._team_around(fr, best), "nearby_players"
        if team < 0:
            source = "unknown"
        return best.id, team, best.foot, fr.frame, True, source

    def _team_of(self, pid: int) -> int:
        """Majority team of a track over everything seen so far (TRACK's
        per-frame value can be -1 for stretches). build.py overrides with
        identities.json when NUMBERS knows the track."""
        votes: dict[int, int] = {}
        for fr in self.frames:
            pl = fr.player(pid)
            if pl is not None and pl.team >= 0:
                votes[pl.team] = votes.get(pl.team, 0) + 1
        return max(votes, key=votes.get) if votes else -1

    def _team_around(self, fr: Frame, shooter: Player) -> int:
        """Last resort for a track that never got a colour: the majority team
        of the players standing nearest to the shooter at release (a free-throw
        line-up has both teams, an open shooter has teammates around)."""
        others = [pl for pl in fr.players if pl.id != shooter.id and pl.team >= 0]
        if not others:
            return -1
        sx, sy = shooter.center
        others.sort(key=lambda pl: math.hypot(pl.center[0] - sx, pl.center[1] - sy))
        votes: dict[int, int] = {}
        for pl in others[:3]:
            votes[pl.team] = votes.get(pl.team, 0) + 1
        return max(votes, key=votes.get)

    def _shooter(self, up_index: int) -> tuple[Possession | None, Point | None, int | None, bool]:
        p = self.p
        frames = self.frames
        t_up = frames[up_index].t
        candidates = [
            s
            for s in self.possession.segments
            if s.start_t < t_up and s.end_t >= t_up - p.shooter_lookback_s and s.start_index >= self._floor_index
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
            release_index = last_held if last_held is not None else min(seg.end_index, up_index)

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
