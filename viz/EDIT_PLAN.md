# EDIT PLAN — FollowCam: how it rotates & records the court
Output: 1920x1080, 30 fps, target 8.0s seamless loop, final.mov (h264) + mp4 preview

## Timeline
| # | Source | In | Out | Notes |
|---|--------|----|-----|-------|
| 1 | source/rig_sweep.mp4 | 00:00:00.0 | 00:00:08.0 | CAD rig panning 40°→140°→40°, left panel (960x1080) |

## Effects
| # | Where | Effect | Owner | Toolchain |
|---|-------|--------|-------|-----------|
| 1 | right panel, 0–8s | top-down court coverage animation, synced to servo angle | codex | python |
| 2 | full frame, 0–8s | bottom caption bar | claude/ffmpeg | – |

## Delegation notes
- Sync is exact: servo angle θ(t) = 90 + 50·sin(2π·t/8) degrees, t in seconds,
  240 frames at 30 fps. The rig panel was rendered with this same formula.
- Court panel: portrait 960x1080. Full basketball court top-down, long axis
  vertical. Tripod/camera at the LEFT sideline midpoint (scorer's table spot).
  Aim direction = θ−90 away from perpendicular; θ=90 aims straight across the
  court, θ=40/140 aim at the two baskets. FOV wedge ~66°, soft blue fill.
  Orange ball rides the aim ray at varying depth. HUD: live "servo 123°"
  readout + small 40–140° arc gauge + pulsing REC dot.
- Palette matches the poster: bg #fafafa, dark #19191c, orange #eb6414,
  servo blue #2661e6. Font: Helvetica.
