"""StatsEngine: feed one tracks line (or Frame) at a time, get shot events
out as they become final. Batch mode (build.py) drives the same engine, so
live and offline results are identical by construction.

    engine = StatsEngine(dt=0.1)
    for line in tracks:              # dict per docs/ORCHESTRATION.md or io.Frame
        for shot in engine.push(line):
            ...                      # final verdict, at most made_window_s after the drop
    engine.finish()

Ball cleaning (see io.clean_ball) needs the *next* ball detection to judge a
jump, so a frame with a ball is held back until the next ball detection or
until `max_gap_s` has passed; frames are always processed in order.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict, deque

from .court import on_court
from .io import Frame, frame_from_dict, is_round_ball, plausible_move
from .possession import PossessionParams, PossessionTracker
from .shots import ShotDetector, ShotEvent, ShotParams


class OnCourtFilter:
    """Drops bench players and spectators from a frame's player list.

    With a calibration: keep a player only while the foot point projects
    inside the court plus `margin_m`. Without: a track is off court while,
    over the trailing `window_s`, its median bbox height is below
    `min_height_ratio` of the median height of all detections in that window
    AND its median foot y lies in the top `top_frac` of the image (bench and
    far sideline). QA finding on dev60: seated spectators were tracked as
    players and inflated FGA and possession. If TRACK already stamps every
    player with "on_court" (true/false), that verdict is used as is.
    """

    def __init__(
        self,
        calib: dict | None = None,
        *,
        image_height: float = 1080.0,
        window_s: float = 5.0,
        min_height_ratio: float = 0.6,
        top_frac: float = 0.35,
        margin_m: float = 1.0,
    ) -> None:
        self.calib = calib
        self.image_height = image_height
        self.window_s = window_s
        self.min_height_ratio = min_height_ratio
        self.top_frac = top_frac
        self.margin_m = margin_m
        self._window: deque[tuple[float, int, float, float]] = deque()  # t, id, height, foot_y
        self._by_id: dict[int, deque[tuple[float, float, float]]] = defaultdict(deque)  # t, height, foot_y
        self.removed: Counter[int] = Counter()  # id -> frames removed

    def apply(self, fr: Frame) -> list[int]:
        """Remove off-court players from `fr.players` in place; returns their ids."""
        if not fr.players:
            return []
        keep, removed = [], []
        if all(p.on_court is not None for p in fr.players):  # TRACK already decided for everyone
            for p in fr.players:
                (keep if p.on_court else removed).append(p)
            self._record(fr, keep, removed)
            return [p.id for p in removed]
        if self.calib is not None:
            for p in fr.players:
                verdict = on_court(self.calib, fr.frame, p.foot, self.margin_m)
                (removed if verdict is False else keep).append(p)
            if any(on_court(self.calib, fr.frame, p.foot, self.margin_m) is not None for p in fr.players):
                self._record(fr, keep, removed)
                return [p.id for p in removed]
            keep, removed = [], []  # no homography for this frame: fall through to the heuristic
        self._advance(fr)
        band = [e[2] for e in self._window]
        band_median = statistics.median(band) if band else 0.0
        for p in fr.players:
            hist = self._by_id[p.id]
            h_med = statistics.median(e[1] for e in hist)
            y_med = statistics.median(e[2] for e in hist)
            small = band_median > 0 and h_med < self.min_height_ratio * band_median
            high = y_med < self.top_frac * self.image_height
            (removed if (small and high) else keep).append(p)
        self._record(fr, keep, removed)
        return [p.id for p in removed]

    def _advance(self, fr: Frame) -> None:
        for p in fr.players:
            self._window.append((fr.t, p.id, p.height, p.foot[1]))
            self._by_id[p.id].append((fr.t, p.height, p.foot[1]))
        cutoff = fr.t - self.window_s
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
        for pid in [p.id for p in fr.players]:
            hist = self._by_id[pid]
            while hist and hist[0][0] < cutoff:
                hist.popleft()

    def _record(self, fr: Frame, keep: list, removed: list) -> None:
        fr.players = keep
        for p in removed:
            self.removed[p.id] += 1


class StatsEngine:
    def __init__(
        self,
        *,
        dt: float = 0.1,
        fps: float | None = None,
        possession_params: PossessionParams = PossessionParams(),
        shot_params: ShotParams = ShotParams(),
        max_gap_s: float = 0.6,
        cuts: list[int] | None = None,
        calib: dict | None = None,
        image_height: float = 1080.0,
        on_court_filter: bool = True,
    ) -> None:
        self.fps = fps
        self.max_gap_s = max_gap_s
        self.court_filter = OnCourtFilter(calib, image_height=image_height) if on_court_filter else None
        self._cuts = sorted(set(int(c) for c in (cuts or [])))  # frame numbers where the footage jumps
        self.cuts_applied = 0
        self.possession = PossessionTracker(possession_params, dt=dt)
        self.detector = ShotDetector(self.possession, shot_params)
        self._queue: list[Frame] = []  # frames not yet processed (first one may hold a ball under judgement)
        self._last_kept: Frame | None = None  # last processed frame that had a ball
        self.dropped_balls = 0

    # --- public ---------------------------------------------------------------

    @property
    def frames(self) -> list[Frame]:
        return self.possession.frames

    @property
    def shots(self) -> list[ShotEvent]:
        return self.detector.shots

    @property
    def holder(self) -> int | None:
        return self.possession.current

    def push(self, frame: Frame | dict) -> list[ShotEvent]:
        fr = frame if isinstance(frame, Frame) else frame_from_dict(frame, self.fps)
        done: list[ShotEvent] = []
        if self._cuts and fr.frame >= self._cuts[0]:
            while self._cuts and fr.frame >= self._cuts[0]:
                self._cuts.pop(0)
            done += self.reset()
        if fr.ball is not None and fr.ball.bbox is not None and not is_round_ball(fr.ball.bbox):
            fr.ball = None
            self.dropped_balls += 1
        if self.court_filter is not None:
            self.court_filter.apply(fr)
        self._queue.append(fr)
        return done + self._drain(final=False)

    @property
    def removed_ids(self) -> dict[int, int]:
        """Track id -> number of frames in which it was dropped as off-court."""
        return dict(self.court_filter.removed) if self.court_filter else {}

    def reset(self) -> list[ShotEvent]:
        """A cut in the footage (or a new camera position): everything that
        looks back in time starts over. Frames stay, so segments and stats
        before the cut are kept; a pending verdict becomes a miss."""
        done = self._drain(final=True)
        done += self.detector.finish()
        self.possession.reset()
        self.detector.reset()
        self._last_kept = None
        self.cuts_applied += 1
        return done

    def finish(self) -> list[ShotEvent]:
        done = self._drain(final=True)
        done += self.detector.finish()
        return done

    # --- internals ------------------------------------------------------------

    def _drain(self, final: bool) -> list[ShotEvent]:
        done: list[ShotEvent] = []
        while self._queue:
            head = self._queue[0]
            if head.ball is None:
                done += self._process(self._queue.pop(0))
                continue
            nxt = next((f for f in self._queue[1:] if f.ball is not None), None)
            if nxt is None:
                newest = self._queue[-1]
                if not final and newest.t - head.t <= self.max_gap_s:
                    break  # wait for the next ball detection to judge this one
            ok_prev = self._last_kept is None or plausible_move(self._last_kept, head, self.max_gap_s)
            ok_next = nxt is None or plausible_move(head, nxt, self.max_gap_s)
            if not (ok_prev or ok_next):
                head.ball = None
                self.dropped_balls += 1
                done += self._process(self._queue.pop(0))
                continue
            self._last_kept = head
            done += self._process(self._queue.pop(0))
        return done

    def _process(self, fr: Frame) -> list[ShotEvent]:
        self.possession.push(fr)
        return self.detector.push(len(self.frames) - 1)
