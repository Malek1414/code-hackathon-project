# FollowCam — the €20 robot cameraman (working title; team renames in 5 min tomorrow)

**One-liner:** A 3D-printed robotic actuator that grips the pan handle of any ordinary tripod and steers a phone to automatically follow the ball — turning the tripod you already own into an auto-tracking sports camera for ~€20, instead of a €2,000+ Veo/Pixellot unit.

## Why this wins a one-day hackathon

- **Live physical demo**: someone walks across the room with a ball, the camera visibly follows. No slideware can compete with a robot moving on stage.
- **Hardware + AI + product story** in one build — exactly the "design & build, prototype if you're feeling it" brief.
- **Real market wedge**: 90%+ of youth/amateur games are never filmed or analyzed because auto-tracking cameras cost thousands. Retrofitting existing tripods collapses the entry price to a printed part + hobby servo + the phone already in your pocket.

## The demo (what must work at 16:00 tomorrow)

Phone on tripod → streams to laptop (Continuity Camera / webcam) → OpenCV tracks the ball → laptop sends pan angle over USB to a microcontroller → servo, mounted to the tripod leg via printed bracket, pushes a rod connected to a printed clamp on the pan handle → camera pans to keep the ball centered. Horizontal only. That's the whole demo.

## The story on top (pitch, not build)

The actuator is the cheapest possible entry point into a **position-data layer** for amateur sports:

1. **Broadcasting (B2C)** — auto-followed footage + live overlays (names, score, stats above players). Accessibility angle: easier for kids and viewers with disabilities to follow the game.
2. **Team-side (B2B)** — movement/tactical analysis, formation-based prediction of opponent strategy.
3. **Roadmap** — the camera that follows the ball is the same camera that can keep the score: auto-with-veto scorekeeping for the leagues that run on one harried volunteer (see `reference/camera-score-tracker-vision.md`).

One tracking engine, three revenue surfaces. The hackathon build proves the physical capture layer.

## Fallback ladder (decide fast, don't sink)

- **A (full)**: printed clamp + servo, live ball following.
- **B (servo works, print late/failed)**: servo + rod zip-tied directly to handle. Ugly but moves.
- **C (no servo control)**: "digital pan" — wide static shot, software crops and follows the ball in the output window; printed clamp shown as the hardware roadmap piece.
- **D (tracking flaky)**: neon-colored ball or AprilTag sticker on the ball; tune HSV live. (Sticker = actually part of the product thesis — tags anyone can apply.)
