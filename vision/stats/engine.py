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

from .io import Frame, frame_from_dict, is_round_ball, plausible_move
from .possession import PossessionParams, PossessionTracker
from .shots import ShotDetector, ShotEvent, ShotParams


class StatsEngine:
    def __init__(
        self,
        *,
        dt: float = 0.1,
        fps: float | None = None,
        possession_params: PossessionParams = PossessionParams(),
        shot_params: ShotParams = ShotParams(),
        max_gap_s: float = 0.6,
    ) -> None:
        self.fps = fps
        self.max_gap_s = max_gap_s
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
        if fr.ball is not None and fr.ball.bbox is not None and not is_round_ball(fr.ball.bbox):
            fr.ball = None
            self.dropped_balls += 1
        self._queue.append(fr)
        return self._drain(final=False)

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
