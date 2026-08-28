# FollowCam hardware — wiring & bring-up

The whole circuit is three connections. Do them in this order, test after
each step with `software/servo_test.py`.

## Circuit

```
 laptop ──USB──> Arduino Uno
                    │
                    ├─ pin 9  ──────────── servo SIGNAL (orange/yellow)
                    ├─ GND ─────┬───────── servo GND    (brown/black)
                    │           │
                    └─ 5V ──────┼───────── servo V+     (red)   ← SG90 ONLY
                                │
                 4xAA holder (+)┼───────── servo V+             ← MG990/995/996R
                 4xAA holder (–)┘   (battery – joins Arduino GND: COMMON GROUND)
```

- **SG90 (today's build)**: light enough to run off the Uno's 5V pin. Three
  wires total: signal→9, V+→5V, GND→GND.
- **MG990/MG995/MG996R upgrade**: NEVER off the 5V pin (stall >1A browns out
  the board). V+ from the 4xAA pack (≈6V), pack minus tied to Arduino GND.
  Signal and GND to the Arduino as before.
- Servo lead colors: orange/yellow = signal, red = V+, brown/black = GND.

## Flash + bench test (5 min, before any printing matters)

1. Arduino IDE → open `software/servo_pan/servo_pan.ino` → board "Arduino
   Uno" → port `/dev/cu.usbmodem*` → Upload.
2. `python3 software/servo_test.py` — auto-finds the port, centers at 90°,
   then: `s` sweep 40↔140, `j`/`k` nudge ±5°, number+Enter = absolute angle,
   `q` quit. If the horn buzzes at the limits, the angles are hitting the
   servo's own end stops — fine, firmware clamps to 40–140 anyway.
3. Attach the printed fork arm to the horn ONLY after centering at 90°, with
   the arm perpendicular to the servo body (so 40/140 land symmetric).

## Full-rig loop (at the court)

1. Clamp on the column, servo in the pocket, fork over the handle shaft.
2. `python3 software/pan_bridge.py --port /dev/cu.usbmodem*` on the laptop,
   phone app → link rig → tap the ball. OR laptop-only:
   `python3 software/ball_tracker.py` (webcam/Continuity Camera path).
3. First motion test at LOW speed: `servo_test.py` sweep with the phone
   mounted — watch the clamp for slip, the fork for binding, re-tighten.

## Outside checklist

laptop (charged) · Uno + USB cable · servo (+spare SG90) · 4xAA pack if
metal-gear servo · printed clamp + fork arm · M3 bolts + zip ties · ball ·
phone hotspot on (bridge needs phone and laptop on one network) · tape.
