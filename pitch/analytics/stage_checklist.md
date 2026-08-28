# Stage run sheet, analytics side (live demo after the rig demo)

1. 15:50 Rig on the table, phone in the mount, phone unlocked, Continuity Camera
   allowed, laptop on power, screen mirrored to the projector at 1920x1080.
2. Terminal in the repo, venv active, probe the camera:
   `.venv/bin/python -m vision.live.live --list-sources` and note the index of the phone.
3. Start live mode 60 s before we go on, window focused on the projector screen:
   `.venv/bin/python -m vision.live.live --source <index> --minimap panel`
   (fallback if the camera fails: `--source data/clips/dev60.mp4 --realtime --replay out/tracks.jsonl`).
4. Optional stream: `FOLLOWCAM_RTMP_URL` in `.env` (never on screen); MJPEG preview
   at http://127.0.0.1:8501/stream on a second device if the projector cable dies.
5. Hotkeys, window focused: 1 = +2 team A, 2 = +2 team B, 3 = +3 team A, 4 = +3 team B,
   z = undo the last score, q = quit. Whoever holds the laptop presses, not the speaker.
6. Choreography: after the rig follows the ball (Malek), one player takes a shot
   at the mini hoop; the score bar flashes. If the flash is right, the speaker says
   "the camera just kept the score". If it is wrong, press z and say "and a human
   vetoes with one key; auto with veto is the product".
7. Then one line on the numbers, no more: "300 frames auto-labeled, players found
   at 0.96, 10 players tracked per frame, all on this laptop today."
8. Kill switch: if detection stalls for more than 5 s, press q, switch to the
   video `out/pitch/analytics_block.mp4` already open in QuickTime, full screen.
9. Before leaving the stage: q in the live window, stop the RTMP push (it stops
   with the process), lock the phone.
10. Files to have open before we start: live window, QuickTime with the analytics
    block, browser with `out/dashboard.html`, the deck.

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
