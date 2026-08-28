---
task: 001
toolchain: python
inputs:
  - clips/clip_01.mp4
output: fx/task-001/court_panel.mov
specs:
  width: 960
  height: 1080
  duration: 8.0
  fps: 30
  alpha: false
---
# Goal
Generate a fully synthetic 8.0s animation panel (no input footage needed —
clip_01 is only a style reference for the matching left panel): a top-down
basketball-court diagram showing how a servo-panned phone camera sweeps and
records the whole court from the scorer's-table spot.

Render 240 frames with PIL (or matplotlib Agg), assemble with ffmpeg
(libx264, crf 14, yuv420p, 30 fps, exactly 240 frames → 8.0s).

## Scene (portrait 960x1080)
- Background #fafafa. Margin ~70px all around.
- Full basketball court, top-down, LONG axis vertical, centered right of the
  tripod: draw court floor in a very light warm tint (#f3ead9), boundary and
  markings (center line, center circle, both keys + arcs, three-point arcs)
  as 3px lines in #b9ad97. Court aspect 28:15 (length:width), width fits
  panel: court spans roughly x 150→890, y 90→990 (rotate markings to match
  vertical long axis; baskets at top and bottom ends).
- Tripod/camera: filled dark dot (#19191c, r=14) with orange ring, at the
  LEFT sideline midpoint (x≈150, y≈540). Small label "FollowCam" left of it.

## Motion (must sync EXACTLY)
- Servo angle θ(t) = 90 + 50*sin(2*pi*t/8.0) degrees, t = frame/30.
- Aim direction: θ=90 → straight across the court (+x, horizontal);
  θ>90 rotates the aim toward the TOP basket, θ<90 toward the BOTTOM
  (aim angle from +x axis = θ−90 degrees, counterclockwise positive).
- FOV wedge: 66° total, centered on the aim, originating at the tripod dot,
  radius ~820px (clip to court/panel). Fill #2661e6 at ~18% opacity, edges
  2px solid #2661e6. A thin 1px center ray, dashed, #2661e6.
- Ball: orange circle (#eb6414, r=13, subtle darker outline) positioned ON
  the aim ray at depth d(t) = 380 + 190*sin(2*pi*t/4.0 + 1.1) px from the
  tripod, plus a small perpendicular wobble 18*sin(2*pi*t/1.6). The wedge
  therefore always contains the ball — that is the whole story.
- Trailing path: fading polyline of the ball's last ~1.2s of positions,
  orange at decreasing opacity.

## HUD
- Top-left: "court coverage — top view" in Helvetica bold ~40px, #19191c.
- Top-right: live readout "servo 123°" (rounded int of θ), Helvetica ~40px
  monospace-ish alignment, #19191c on a white chip.
- Around the tripod dot: thin arc gauge from 40° to 140° (r≈60), #b9ad97,
  with a 3px orange needle at current θ and tick labels "40°" / "140°".
- Bottom-left: pulsing red REC dot (opacity 0.35+0.65*|sin(pi*t)|) + "REC"
  + "TRACKING" chip in #19191c text.
- Use /System/Library/Fonts/Helvetica.ttc; degree signs must render.

## Quality bar
- Anti-aliased drawing (draw at 2x = 1920x2160 and LANCZOS-downscale each
  frame to 960x1080 before encoding).
- No flicker: all geometry computed per-frame from t, no randomness.
- Frame 0 and frame 239 must be near-identical to frame 0 of the next loop
  (θ(0)=θ(8)=90) so the clip loops seamlessly.

## Feedback
