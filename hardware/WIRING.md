# FollowCam hardware — wiring & bring-up

The whole circuit is three connections. Do them in this order, test after
each step with `software/servo_test.py`.

## Circuit — Pod v4 (MG996R)

Three connections. The servo sits in the pod on the tripod; the Uno sits in its
own case clipped to a tripod leg. **The servo is on the stationary part of the
pod, so none of these wires rotate** — no service loop, no twist limit.

```
 laptop ──USB──> Arduino Uno  (in 10_leg_saddle + 08_uno_tray on a tripod leg)
                    │
                    ├─ pin 9  ──────────── servo SIGNAL (orange/yellow)
                    ├─ GND ─────┬───────── servo GND    (brown/black)
                    │           │
                 4xAA holder (+)┼───────── servo V+     (red)
                 4xAA holder (–)┘   (pack – joins Arduino GND: COMMON GROUND)
```

**NEVER run the MG996R off the Uno's 5 V pin.** Stall current is >1 A and it
browns out the board mid-demo. V+ comes from the 4xAA pack (~6 V) in `11_aa_bay`,
pack minus tied to Arduino GND.

### Lead length — the stock servo lead is NOT long enough

| Leg | mm |
|---|---:|
| servo → pod wire exit (internal routing) | ~80 |
| pod exit → leg case, straight line | 181 |
| elevator column raise (`column_free_h`) | +160 |
| routing slack + service loop (25 %) | +86 |
| **required servo-to-Uno** | **507** |

The MG996R's own lead is 300 mm — only 220 mm of that is usable outside the pod.
Add **one 500 mm 3-pin servo extension** (800 mm total, 293 mm margin). Anything
shorter pulls tight as soon as the centre column is raised.

Strain-relieve the lead with a zip tie through the two tie slots beside the pod's
wire exit before it drops to the leg.

### SG90 (superseded)

The v1/v2 column-clamp rig used an SG90 off the Uno's 5 V pin. That build is
retired; `cad/followcam-rig.scad` is kept for reference only.

## Flash + bench test (5 min, before any printing matters)

1. Arduino IDE → open `software/servo_pan/servo_pan.ino` → board "Arduino
   Uno" → port `/dev/cu.usbmodem*` → Upload. (For the 180° pod firmware use
   `software/servo_pod180/servo_pod180.ino`.)
2. `python3 software/servo_test.py` — auto-finds the port, centers at 90°,
   then: `s` sweep 40↔140, `j`/`k` nudge ±5°, number+Enter = absolute angle,
   `q` quit. If the horn buzzes at the limits, the angles are hitting the
   servo's own end stops — fine, firmware clamps to 40–140 anyway.
3. Pod v4: fit `05_drive_dog` to the horn ONLY after centering at 90°, so the
   rotor's ±90° sweep lands symmetric.

## Full-rig loop (at the court)

1. Clamp on the column, servo in the pocket, fork over the handle shaft.
2. `python3 software/pan_bridge.py --port /dev/cu.usbmodem*` on the laptop,
   phone app → link rig → tap the ball. OR laptop-only:
   `python3 software/ball_tracker.py` (webcam/Continuity Camera path).
3. First motion test at LOW speed: `servo_test.py` sweep with the phone
   mounted — watch the clamp for slip, the fork for binding, re-tighten.

## Outside checklist

laptop (charged) · Uno + USB cable · MG996R (+spare) · **500 mm servo
extension** · 4xAA pack + fresh cells · the 11 printed parts · 1/4-20 nut ·
M3/M4 screws + zip ties + 2 straps · foam shim strip · ball · phone hotspot on
(bridge needs phone and laptop on one network) · tape.
