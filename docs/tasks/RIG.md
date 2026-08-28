# Role RIG (assigned 12:10 by ORCH, session samimagdouli-61)

Read `README.md`, `docs/FINDINGS.md` and `software/ball_tracker.py` first. You own
`software/ball_tracker_yolo.py` only (new file, Malek's `ball_tracker.py` stays
untouched as the fallback).

Goal: the physical rig's ball following gets a real detector instead of HSV color.

1. `software/ball_tracker_yolo.py`: same CLI and serial protocol as `ball_tracker.py`
   (`--cam`, `--video`, `--serial`, `A<angle>\n` at 115200, 40 to 140 degrees, KP,
   deadband, EMA), but detection via ultralytics with `models/ball_hoop_avishah.pt`
   (class 0 = ball) at imgsz 640 on `device="mps"` if free else cpu, conf 0.35, keep
   the ball nearest the previous position, fall back to the HSV mask when the model
   finds nothing for 15 frames. Target 20+ fps on the laptop webcam at 960x540.
2. Respect the GPU schedule in `docs/ORCHESTRATION.md`: a nano model at 640 is light,
   but check `ps -axo command | grep .venv/bin/python` before long runs and keep test
   runs under 2 minutes while other model jobs are running.
3. Test on `data/clips/dev60.mp4` (`--video`) and on the laptop webcam (`--cam 0`),
   report fps and how often the ball is found; a short screen recording or 3 saved
   frames with the detection box into `out/rig/`.
4. Report to ORCH via `SendMessage(to: "samimagdouli-61", ...)`, first line "RIG ...".
