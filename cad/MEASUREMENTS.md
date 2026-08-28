# Measurements — v2 (targeted, based on the photos)

Design locked: servo clamps the **center column** just below the head, shaft up;
printed **fork arm** on the servo horn captures the **handle's metal shaft** and
pushes it left/right. So we only need the numbers below.

**No calipers? Paper-strip trick for tubes:** wrap a paper strip around the tube,
mark where it overlaps, measure the strip length with a ruler, divide by 3.14.
That's the diameter to ~0.5mm.

## The 7 numbers (mm everywhere)

1. **Center column diameter** — the vertical tube directly under the head
   (between the head base and where the leg braces/strap attach). Paper-strip it.
   → `column_d`
2. **Free column height** — vertical clearance on that column from the underside
   of the rotating head base down to the first obstruction (brace hook, strap
   ring, crown). → `column_free_h`
3. **Handle shaft diameter** — the chromed metal rod between the head and the
   plastic grip. Paper-strip or ruler. → `shaft_d`
4. **Arm reach** — horizontal distance from the center of the column (pan axis)
   straight out to the middle of the chromed shaft section. Hold a ruler
   horizontally; eyeballing to ±5mm is fine. → `arm_reach`
5. **Fork drop** — with the head level, vertical distance from the top of the
   column-clamp zone (just below the head base) DOWN to the handle shaft. The
   handle angles downward, so the shaft sits below the head. ±5mm fine. → `fork_drop`
6. **Handle butt hole** — is the hole at the fat end of the grip a real
   through-hole? If yes, its diameter. (Bonus attachment option, not required.)
7. **Servo horn** — once you know which servo the makerspace has: distance from
   horn center to its outermost hole, and horn center-screw size. (Skip if
   unknown tonight; MG996R defaults are in the SCAD.)

## The 3 function checks (do these tonight, 2 minutes)

A. **Find the pan friction control.** Loosen it until the head pans with two
   fingers on the handle — smooth, no stick-slip jerks. If it always feels
   gritty/sticky, note that (we compensate in software with slower slew).
B. **Lock tilt HARD.** Twist the handle grip tight so the phone doesn't droop
   when the fork pushes sideways. Confirm sideways pushes don't change tilt.
C. **Sweep range.** Pan the handle through the arc a court would need (~±45°).
   Confirm nothing collides (legs, column crank, phone mount) through the sweep.

## Video (requested)

Film 15–30s: (1) pan the handle left-right slowly through full sweep, (2) show
the two-finger looseness, (3) slow close pass over the head from all sides.
Drop it in `tripod-photos/` — Claude extracts frames and verifies the mechanism.
