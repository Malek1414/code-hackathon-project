"""Pan servo control for the rig: keep the ball near the frame centre.

Control law from Malek's software/ball_tracker.py (KP 0.06 deg per px of
horizontal error, deadband 25 px, EMA 0.35 on the ball x), protocol from
software/servo_pan/servo_pan.ino ("A<angle>\\n" at 115200, 40..140 deg,
centre 90). Commands go out at most `max_hz` per second and only when the
angle moved >= `min_step_deg`. Ball lost: hold the angle, after `return_s`
drift back to the centre. Coasted (predicted) ball points steer as well.
"""

from __future__ import annotations

import logging
import sys
import time

log = logging.getLogger(__name__)

KP = 0.06
DEADBAND_PX = 25.0
EMA = 0.35
SERVO_MIN, SERVO_MAX, SERVO_CENTER = 40.0, 140.0, 90.0


class PanController:
    def __init__(self, port: str | None, *, dry: bool = False, invert: bool = False, frame_width: int = 1920,
                 max_hz: float = 20.0, min_step_deg: float = 1.0, return_s: float = 3.0,
                 return_rate_deg_s: float = 10.0, max_rate_deg_s: float = 90.0) -> None:
        self.dry = dry
        self.invert = invert
        self.center_x = frame_width / 2
        self.min_interval = 1.0 / max_hz
        self.min_step = min_step_deg
        self.return_s = return_s
        self.return_rate = return_rate_deg_s
        self.max_rate = max_rate_deg_s  # slew limit: KP was tuned on a webcam, at 1920 px a far ball asks for 50 deg at once
        self.angle = SERVO_CENTER
        self.sent_angle: float | None = None
        self.smooth_x: float | None = None
        self.last_seen: float | None = None
        self.last_sent_t = 0.0
        self.last_update_t: float | None = None
        self.commands = 0
        self.ser = None
        if port and not dry:
            import serial  # pyserial

            self.ser = serial.Serial(port, 115200, timeout=0.05)
            time.sleep(1.5)  # the Arduino resets on connect
            log.info("pan servo on %s", port)
        elif dry:
            log.info("pan servo dry run: commands go to stdout")

    @property
    def active(self) -> bool:
        return self.dry or self.ser is not None

    def update(self, ball_x: float | None, now: float | None = None) -> float:
        """Feed the ball's x (pixels, None when lost); returns the commanded angle."""
        now = time.monotonic() if now is None else now
        dt = 0.0 if self.last_update_t is None else max(now - self.last_update_t, 0.0)
        self.last_update_t = now
        if ball_x is not None:
            self.smooth_x = ball_x if self.smooth_x is None else EMA * ball_x + (1 - EMA) * self.smooth_x
            err = self.smooth_x - self.center_x
            if abs(err) > DEADBAND_PX:
                step = KP * err * (1 if self.invert else -1)
                limit = self.max_rate * (dt if dt > 0 else 1.0 / 30.0)
                step = max(-limit, min(limit, step))
                self.angle = min(SERVO_MAX, max(SERVO_MIN, self.angle + step))
            self.last_seen = now
        elif self.last_seen is not None and now - self.last_seen > self.return_s:
            # ball lost for a while: drift back to the centre
            diff = SERVO_CENTER - self.angle
            move = min(abs(diff), self.return_rate * dt)
            self.angle += move if diff > 0 else -move
            self.smooth_x = None
        self._maybe_send(now)
        return self.angle

    def _maybe_send(self, now: float) -> None:
        if not self.active:
            return
        if now - self.last_sent_t < self.min_interval:
            return
        target = float(round(self.angle))
        if self.sent_angle is not None and abs(target - self.sent_angle) < self.min_step:
            return
        line = f"A{int(target)}\n"
        if self.dry:
            sys.stdout.write(f"{now:10.3f} {line}")
            sys.stdout.flush()
        else:
            try:
                self.ser.write(line.encode())
            except Exception as exc:  # noqa: BLE001 - a flaky cable must not kill the overlay
                log.error("pan serial write failed: %s", exc)
        self.sent_angle = target
        self.last_sent_t = now
        self.commands += 1

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.write(f"A{int(SERVO_CENTER)}\n".encode())
                self.ser.close()
            except Exception:  # noqa: BLE001
                pass
