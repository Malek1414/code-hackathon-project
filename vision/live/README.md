# LIVE mode – stage runbook

All commands from the repo root, models need the GPU slot (ORCH schedule).

1. Without GPU (any time): replay the tracked clip like a live camera
   `.venv/bin/python -m vision.live.live --source data/clips/dev60.mp4 --realtime --replay out/dev60_v2/tracks.jsonl`
2. With models, from file: `.venv/bin/python -m vision.live.live --source data/clips/dev60.mp4 --realtime`
3. Phone (Continuity Camera): wake the iPhone, then
   `.venv/bin/python -m vision.live.live --list-sources` → the iPhone is the highest index (1 on 28.08, "no-frame" in the
   2 s probe is normal, it wakes up within ~10 s); the Mac's own camera is 0.
   `.venv/bin/python -m vision.live.live --source 1`   (or `--source auto` = highest camera that delivers)
4. Stream: put the ingest URL in `.env` as `FOLLOWCAM_RTMP_URL=rtmp://.../<key>` (never in code, never in git);
   the push starts automatically. Local check: `ffmpeg -listen 1 -f flv -i rtmp://127.0.0.1:1935/live/test -f null -`
5. Browser / OBS source: http://127.0.0.1:8501/stream

Window (needs focus): `1`/`2` = +2 for team A/B, `3`/`4` = +3, `z` = undo (auto calls too), `q` = quit.
Auto calls: a confirmed basket with a known team adds +2 and flashes green; an unconfirmed one
(ball lost at the rim) flashes "looked in, press 1 or 2" and adds nothing.
Panel on the right: 2D court from `out/court_calib_<clip>.json`; "uncalibrated" when missing.
Score bar shows "Kamera: kein Bild" while the source is silent; the app keeps running and reopens the device.
At exit: `out/live_events.json` (shots, score, frames).

## Servo pan (rig)
Find the Arduino port: `ls /dev/cu.usb*` (e.g. `/dev/cu.usbserial-1420` or `/dev/cu.usbmodem14201`), then
`.venv/bin/python -m vision.live.live --source 1 --serial /dev/cu.usbserial-XXXX` (add `--invert-pan` if the camera
runs away from the ball). Protocol `A<angle>\n` at 115200, 40-140 deg, centre 90 (software/servo_pan/servo_pan.ino);
law KP 0.06 deg/px, deadband 25 px, EMA 0.35 (software/ball_tracker.py); <= 20 commands/s, only on >= 1 deg change;
ball lost -> hold, after 3 s drift back to 90. Score bar shows `pan 97 deg`. Test without hardware: `--dry-serial`
prints the commands.
