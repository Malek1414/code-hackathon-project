"""Pan control law and serial writer for the FollowCam rig, shared by the trackers and live.py.

    from software.pan_control import PanController, ServoSerial

    ctl = PanController()                       # kp 0.06, deadband 25 px, ema 0.35, 40..140 deg
    ser = ServoSerial("/dev/tty.usbserial-XXXX")  # or ServoSerial(None, dry=True) to print instead
    angle = ctl.update(ball_x, frame_w)         # None when the error is inside the deadband
    ser.send(ctl.angle)                         # rate-limited: 20 Hz and only on >= 1 degree change

Same numbers and sign convention as Malek's ball_tracker.py: error = smoothed
ball x minus frame centre, angle -= kp * error (invert=True flips it if the
camera runs away from the ball), clipped to [lo, hi]. The EMA smooths the
ball x before the error is taken. `reset()` forgets the smoothing (use it when
the ball was lost for a while so the first new position is taken as is).
Protocol: `A<angle>\\n` at 115200 baud, integer degrees.
"""

from __future__ import annotations

import time


class PanController:
    def __init__(self, kp: float = 0.06, deadband_px: float = 25.0, ema: float = 0.35,
                 lo: float = 40.0, hi: float = 140.0, center: float = 90.0, invert: bool = False):
        self.kp, self.deadband_px, self.ema = kp, deadband_px, ema
        self.lo, self.hi, self.invert = lo, hi, invert
        self.angle = float(center)
        self.smooth_x: float | None = None

    def reset(self) -> None:
        self.smooth_x = None

    def update(self, ball_x: float, frame_w: int) -> float | None:
        """Feed one ball x (pixels). Returns the new angle, or None if inside the deadband
        (self.angle keeps the last value either way)."""
        self.smooth_x = ball_x if self.smooth_x is None else self.ema * ball_x + (1 - self.ema) * self.smooth_x
        err = self.smooth_x - frame_w / 2
        if abs(err) <= self.deadband_px:
            return None
        step = self.kp * err
        if self.invert:
            step = -step
        self.angle = min(self.hi, max(self.lo, self.angle - step))
        return self.angle


class ServoSerial:
    """Writes `A<angle>\\n`, at most `rate_hz` times per second and only when the
    integer angle changed by at least `min_step` degrees. `dry=True` prints
    instead of opening a port (the --dry-serial mode of the trackers)."""

    def __init__(self, port: str | None, baud: int = 115200, rate_hz: float = 20.0,
                 min_step: int = 1, dry: bool = False):
        self.dry = dry or port is None
        self.min_interval = 1.0 / rate_hz
        self.min_step = min_step
        self.last_sent: int | None = None
        self.last_t = 0.0
        self.sent = 0
        self.ser = None
        if not self.dry:
            import serial
            self.ser = serial.Serial(port, baud, timeout=0.05)
            time.sleep(2)  # Arduino auto-reset

    def send(self, angle: float, now: float | None = None) -> bool:
        """Returns True when a command went out."""
        now = time.time() if now is None else now
        a = int(round(angle))
        if self.last_sent is not None and abs(a - self.last_sent) < self.min_step:
            return False
        if now - self.last_t < self.min_interval:
            return False
        line = f"A{a}\n"
        if self.ser is not None:
            self.ser.write(line.encode())
        else:
            print(f"serial(dry): {line.strip()}", flush=True)
        self.last_sent, self.last_t, self.sent = a, now, self.sent + 1
        return True

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
