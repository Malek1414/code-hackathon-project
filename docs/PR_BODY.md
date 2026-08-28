## What this adds

The analytics half of FollowCam: a clip in, labeled players / ball / hoop, ByteTrack tracks with jersey numbers, a 2D court model, per-player shot stats, a coach dashboard, a live score overlay and a privacy blur. Built on Aug 28 in parallel Claude Code sessions; roles, contracts and GPU schedule in `docs/ORCHESTRATION.md`, per-stage commands in `docs/VISION.md`, measurements in `docs/RESULTS.md`, data handling in `docs/PRIVACY.md`.

| Stage | Where | State |
|---|---|---|
| Frame extraction, auto-labeling (Grounding DINO people, ball/hoop detector), false-ball cleaner, YOLO11n fine-tune | `vision/label/` | 300 frames labeled, 3395 boxes; best.pt val mAP50 0.72 (player 0.96, hoop 0.97, referee 0.76, ball 0.21) |
| Detection, ByteTrack, team colors, overlay video | `vision/track/` | 0.08 s/frame alone on the M3; best.pt persons give 25 % fewer id switches than COCO |
| Jersey numbers (EasyOCR, vote per track, merge ids into players) | `vision/numbers/` | runs; 1 of 75 tracks numbered on the last dev60 run |
| Court calibration, camera propagation across pans and cuts, minimap | `vision/court/` | click tool and propagation work; no calibration file for the demo clips yet |
| Possession, shot events, per-player FG stats | `vision/stats/` | 46 unit tests; 1 attempt found on the 60 s dev clip |
| Coach dashboard (one HTML) | `vision/dashboard/` | renders real data in 0.23 s |
| Live mode: camera in, score bar, hotkey veto, 2D panel, MJPEG, RTMP | `vision/live/` | replay and RTMP tested locally; laptop camera works, iPhone gave no frames at 12:50 |
| Head blur from tracks, retention dry run | `vision/privacy/` | 11,579 heads blurred on the 60 s overlay |
| One-command pipeline, smoke test, status board, QA sheets | `vision/run_all.py`, `vision/smoke_test.py`, `vision/monitor/`, `vision/qa/` | smoke test PASS in 132 s |

## How to run

```
uv venv --python 3.12 .venv && uv pip install ultralytics supervision transformers opencv-python imageio-ffmpeg easyocr scikit-learn lap
.venv/bin/python -m vision.run_all --clip data/clips/dev60.mp4          # or: make demo CLIP=...
.venv/bin/python -m vision.live.live --source 0 --minimap panel
```
Modules that import `vision.*` need the `-m` form. Weights, clips and `out/` are gitignored; the stream key lives only in `.env` (`.env.example` provided).

## Measured

Val on 45 held-out frames: mAP50 0.723 / mAP50-95 0.588, per class above. TRACK on 526 dev60 frames: 9.3 players per frame and 101 track ids with best.pt persons vs 12.5 and 134 with COCO yolo11s; ball in 33 % of frames at conf 0.45, 43 % at 0.30. Labeling 3.2 s/frame, training 25.4 min for 16 epochs, blur 125 s for 60 s of video.

## Unverified

Numbers on `game10.mp4` (10 min): TRACK's full run was scheduled after 14:15. Stats sanity check against a hand count. Live mode from the phone on the rig (only the laptop camera delivered frames). Court calibration on the demo clips (the tool works, nobody clicked the landmarks yet).

## Known limits

Test footage is an edited broadcast with 76 cuts in 10 min (cut detection handles it, single-phone footage is the real input). Ball recall is the weak point: 85 ball labels, fine-tuned ball AP50 0.21, so the ball stays with the separate detector and shots need the ball near the rim. Track ids still switch on fast pans; jersey numbers rarely read at 1080p distance. Referees and bench are not in `tracks.jsonl` since the player class switch, so the blur skips them until TRACK writes the `others` list.

Nothing touches `main`'s rig, CAD, firmware or pitch files except the appended "Analytics pipeline" section in `README.md`.
