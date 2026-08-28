# Stage run sheet, analytics side (live demo after the rig demo)

Facts checked in docs/RISKS.md (12:50): only `.venv/bin/python -m vision.live.live`
works (the `vision/live/live.py` form dies with ModuleNotFoundError); camera 0 is
the FaceTime camera and works, the iPhone appears as index 1 only while
Continuity Camera delivers frames; camera permission is granted to Terminal.app
only; two processes may share camera 0.

1. 15:45 Phone unlocked, landscape on the rig, Do Not Disturb on, same Wi-Fi and
   Bluetooth as the Mac. Laptop on power, screen mirrored at 1920x1080.
2. 15:50 Kill every model job so the GPU is free for the demo:
   `ps -axo pid,command | grep .venv/bin/python`, then `kill` each vision/track,
   vision/label, run_all or smoke_test process (monitor, numbers.watch and
   qa.watch are CPU only and may stay). The rig tracker runs on HSV
   (`software/ball_tracker.py`) or `--device cpu` during the live demo, never a
   second MPS model.
3. 15:55 In Terminal.app (not another terminal, camera permission), repo root:
   `.venv/bin/python -m vision.live.live --list-sources` and pick the index
   that reports frames. On this Mac the phone is `--source 1` (measured in the
   14:10 live test: 1920x1080 at 30 fps in, 13 fps rendered under heavy load);
   0 is the laptop camera, and an automatic choice would take the Mac camera.
4. 15:59 Start live mode, window on the projector screen:
   `.venv/bin/python -m vision.live.live --source 1 --minimap panel`
   First action once the window is up, before the demo: press 1 once and see
   "Team A +2" on the bar, then z to undo it. That proves focus and hotkeys.
   Camera dead: `--source 0` (laptop camera, verified). No camera at all:
   `.venv/bin/python -m vision.live.live --source data/clips/dev60.mp4 --realtime --replay out/dev60_v2/tracks.jsonl --minimap panel`
   (out/dev60_v2/tracks.jsonl matches dev60; out/tracks.jsonl is whatever TRACK ran last).
5. Optional stream: `FOLLOWCAM_RTMP_URL` in `.env` (never on screen); MJPEG at
   http://127.0.0.1:8501/stream is loopback only, it does not reach a phone.
6. Hotkeys work only with the OpenCV window focused (click it once): 1 = +2 team A,
   2 = +2 team B, 3 = +3 team A, 4 = +3 team B, z = undo, q = quit. The laptop
   holder presses, not the speaker.
7. Choreography: after the rig follows the ball (Malek), one player shoots at the
   mini hoop; the score bar flashes. Right: "the camera just kept the score".
   Wrong or nothing (the ball is 20 to 40 px at 5 m, expect misses): press the key
   and say "the system calls, a human corrects with one key; auto with veto is
   the product".
8. One line on the numbers, no more: "300 frames auto-labeled, players found at
   0.96, 25 percent fewer id switches after the fine-tune, all on this laptop today."
9. Kill switch: detection stalls for more than 5 s or `det x fps` on the bar drops
   below 5: q, then the video `out/pitch/analytics_side_by_side.mp4` already open in
   QuickTime, full screen. Dashboard only from this laptop (`open out/dashboard.html`,
   it needs overlay.mp4 and minimap.mp4 next to it), click play once before going on.
10. Leaving the stage: q in the live window (the RTMP push stops with the process),
    lock the phone.

## Anticipated Q&A, analytics side

Privacy, filming minors: The pipeline works on the device that films; nothing
is uploaded, no RTMP unless a URL is set on purpose. Faces are not used by any
model: identity comes from jersey number and color, not from faces. The
export step blurs head regions before a video leaves the laptop
(`vision/privacy/blur.py`, from the tracking boxes alone, no extra model;
`out/overlay_blurred.mp4` is the 60 s proof, 11,579 heads in 2 minutes on
CPU), and raw clips are deleted after a set number of hours
(`vision/privacy/retention.py`). If asked about the referee or the bench:
today's blur covers the tracked players; referees and spectators need their
boxes written by the tracker too, that is a one-line contract change, not a
model. Youth games: the club asks consent as it does for team photos, and the
export is the blurred overlay, not the raw video.

Accuracy: Measured today on held-out frames of a Berlin Landesliga game:
players 0.96 mAP50, hoop 0.97, referee 0.76, ball 0.21 with the fine-tuned
model, which is why the ball uses its own detector and shots are cross-checked
with hoop geometry. Tracking holds 10.6 players per frame. Shot detection is
the least mature piece; it ran on one 60 s clip today and found 1 attempt.

What is automatic and what is human-confirmed: Boxes, tracks, team colors,
jersey numbers, court projection and shot candidates are automatic. Scores are
automatic with a veto: the volunteer at the table presses one key to correct.
The FG table is derived from those confirmed events, so a wrong auto call never
reaches the stats without a human having had 1.5 s to veto it.

Why not Veo or Pixellot: They sell the camera for 2000 euros and up plus a
subscription; we retrofit the tripod and the phone people already own, and the
same footage feeds the analytics. Our analytics ran on a MacBook, not a cloud.

What breaks: an edited broadcast with cuts and close-ups (our test footage was
one) breaks camera tracking; a single phone on a tripod, which is exactly what
the rig produces, does not have that problem.

Does it work for other sports: The label loop is sport-agnostic (Grounding
DINO takes a text prompt), the court model and the shot logic are basketball;
football is the next court spec.
