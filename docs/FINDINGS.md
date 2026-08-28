# FollowCam — findings & decisions log (up to Aug 28, 2026, 11:00)

Everything we've learned and decided so far, in one place. Newest context for
anyone joining the project (hi Sammy).

## 1. Concept & scope (locked Aug 27)

- **Hackathon build**: horizontal-only auto-pan. A printed actuator steers the
  tripod's own pan head via the pan handle; the phone does the filming.
- **Why the pan handle**: no modification to the tripod, works on any tripod,
  demo-visible motion, and the head's own friction damps servo jitter.
- **Long-term story**: same tracking engine feeds broadcasting overlays (B2C),
  tactical analysis (B2B), and auto-scorekeeping (roadmap). The build proves
  the capture layer. Fallback ladder A→D lives in `IDEA.md`.

## 2. Mechanism design v2 (locked Aug 27 evening)

Two printed parts (`cad/followcam-rig.scad`):

1. **Column clamp** — split ring around the center column just below the
   head, M3 bolts + zip-tie backup. Carries a shelf the servo drops into,
   output shaft UP, right next to the pan axis → **servo angle ≈ pan angle
   (near 1:1)**, no linkage math.
2. **Fork arm** — flat bar on the servo horn ending in a tall U-fork that
   captures the handle's chrome shaft from below. The open fork absorbs the
   axis offset and the handle's ~8° downward slope; tall prongs (45 mm) and a
   35 mm trough make vertical and radial alignment forgiving.

Earlier v1 (servo on tripod leg + pushrod to a handle clamp,
`cad/handle-clamp.scad`) is kept for reference but superseded.

## 3. Measured tripod geometry (Aug 27, ~21:45)

| Variable | Value | Note |
|----------|-------|------|
| `column_d` | 15 mm | exact |
| `column_free_h` | ~160 mm | after raising the elevator column |
| `shaft_d` | 8.5 mm | chrome handle shaft, 8–9 mm measured |
| `arm_reach` | 95 mm | fork window covers 60–95 mm from pan axis |
| `fork_drop` | 45 mm | tall prongs; vertical catch window is huge |

Function checks: pan friction loosens to a smooth two-finger pan; tilt locks
hard; ±45° sweep collides with nothing.

## 4. Servo selection

- **SG90** (Elegoo Uno kit) confirmed sufficient once pan friction is
  loosened; it's what the printed pocket is sized for today.
- **Upgrade path**: MG990 / MG995 / MG996R are the same standard ~40×20 mm
  footprint class — change `servo_l`/`servo_w` in the SCAD (40.7 / 20.2),
  reprint the clamp, keep the same firmware. Metal-gear class delivers
  ~9–13 kg·cm at 4.8–6 V vs SG90's ~1.8 kg·cm.
- **Power rule for the metal-gear servos**: external 5–6 V supply, common
  ground with the Arduino — stall current exceeds what the 5 V pin can feed.
  Budget clones (MG990 etc.): sweep-test for centering/jitter before install.

## 5. Firmware (`software/servo_pan/servo_pan.ino`)

- Serial 115200, protocol `A<angle>\n` (e.g. `A95`), input validated and
  clamped to **40°–140°** (mechanical linkage range).
- Slew-rate limiting: max 2.0°/15 ms tick (~133°/s) — smooth pans, no
  stick-slip snapping, less shock through the printed parts.
- Centers at 90° on power-up; servo on pin 9.

## 6. Tracker (`software/ball_tracker.py`)

HSV-threshold ball detection → horizontal error → pan-angle command over
serial. Demo insurance: neon ball or AprilTag sticker + live HSV tuning
(fallback D — and tags are part of the product thesis anyway).

## 7. Court-coverage geometry (validated Aug 28)

From the scorer's-table spot at the sideline midpoint of a 28×15 m court,
covering both baseline corners needs atan(14/15) ≈ **±43°**. The servo's
software range gives **±50°** — closes with margin. Visualized in
`viz/final/final.mov` (right panel).

## 8. Print status (Aug 27 night)

Sliced and ready in `print/`: `followcam_clamp_04` / `followcam_arm_04`
(0.4 mm) plus `_HF04` high-flow variants. PLA/PETG, 25–35% infill, no
supports needed.

## 9. Visual assets (Aug 28 morning)

- `viz/followcam_assembled_poster.png` — labeled 3-view render of the
  assembled rig, generated from `cad/assembly.scad` with the measured dims.
- `viz/final/final.mov` — 1920×1080 8 s loop: CAD rig panning 40°→140°→40°
  side-by-side with the synced top-down court-coverage wedge
  (θ(t) = 90 + 50·sin(2πt/8), the exact motion the firmware produces).
  Regeneration recipe: `viz/EDIT_PLAN.md`.

## 10. Open items

- Print the two parts; dress fits with a file if needed (0.5 mm tolerances).
- Assemble, loosen pan friction, run the sweep test on the real head.
- End-to-end demo rehearsal: tracker → serial → servo → phone follows.
- Pitch: lead with the live demo, then the €20-vs-€2,000 wedge
  (`pitch/pitch.md`, strategy in `IDEA.md`).
