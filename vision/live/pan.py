"""Pan servo link for live.py: thin adapter over the rig's shared control law
(software/pan_control.py, RIG). Law: KP 0.06 deg/px, deadband 25 px, EMA
0.35, 40..140 deg, centre 90, slew 90 deg/s, return to centre after 3 s
without a ball; protocol "A<angle>\\n" at 115200, <= 20 commands/s, only on
>= 1 deg change. Coasted (predicted) ball points are fed like detections.
Constants live in software/pan_control.py only (RIG owns them)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from software.pan_control import PanController as _Law  # noqa: E402
from software.pan_control import ServoSerial  # noqa: E402

log = logging.getLogger(__name__)


class PanController:
    def __init__(self, port: str | None, *, dry: bool = False, invert: bool = False, frame_width: int = 1920) -> None:
        self.law = _Law(invert=invert)
        self.frame_width = int(frame_width)
        self.dry = dry
        self.link = ServoSerial(port if not dry else None, dry=dry)
        if port and not dry:
            time.sleep(1.5)  # the Arduino resets on connect
            log.info("pan servo on %s", port)
        elif dry:
            log.info("pan servo dry run: commands go to stdout")

    @property
    def active(self) -> bool:
        return self.dry or not self.link.dry

    @property
    def angle(self) -> float:
        return self.law.angle

    @property
    def commands(self) -> int:
        return self.link.sent

    def update(self, ball_x: float | None, now: float | None = None) -> float:
        """Feed the ball's x (pixels, None when lost); returns the commanded angle."""
        now = time.monotonic() if now is None else now
        if ball_x is None:
            self.law.no_ball(now)
        else:
            self.law.update(float(ball_x), self.frame_width, now)
        try:
            self.link.send(self.law.angle, now)
        except Exception as exc:  # noqa: BLE001 - a flaky cable must not kill the overlay
            log.error("pan serial write failed: %s", exc)
        return self.law.angle

    def close(self) -> None:
        try:
            if not self.link.dry:
                self.link.send(self.law.center, time.monotonic() + 1.0)
            self.link.close()
        except Exception:  # noqa: BLE001
            pass
