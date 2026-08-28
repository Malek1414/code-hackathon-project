# LIVE mode – stage runbook

All commands from the repo root, models need the GPU slot (ORCH schedule).

1. Without GPU (any time): replay the tracked clip like a live camera
   `.venv/bin/python -m vision.live.live --source data/clips/dev60.mp4 --realtime --replay out/dev60_v2/tracks.jsonl`
2. With models, from file: `.venv/bin/python -m vision.live.live --source data/clips/dev60.mp4 --realtime`
3. Phone (Continuity Camera): wake the iPhone, then
   `.venv/bin/python -m vision.live.live --list-sources` → pick the index that says `ok`, or `--source auto`
   `.venv/bin/python -m vision.live.live --source auto`
4. Stream: put the ingest URL in `.env` as `FOLLOWCAM_RTMP_URL=rtmp://.../<key>` (never in code, never in git);
   the push starts automatically. Local check: `ffmpeg -listen 1 -f flv -i rtmp://127.0.0.1:1935/live/test -f null -`
5. Browser / OBS source: http://127.0.0.1:8501/stream

Window (needs focus): `1`/`2` = +2 for team A/B, `3`/`4` = +3, `z` = undo (auto calls too), `q` = quit.
Auto calls: a confirmed basket with a known team adds +2 and flashes green; an unconfirmed one
(ball lost at the rim) flashes "looked in, press 1 or 2" and adds nothing.
Panel on the right: 2D court from `out/court_calib_<clip>.json`; "uncalibrated" when missing.
Score bar shows "Kamera: kein Bild" while the source is silent; the app keeps running and reopens the device.
At exit: `out/live_events.json` (shots, score, frames).
