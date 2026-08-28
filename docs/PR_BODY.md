## What this adds

The analytics half of FollowCam: a clip in, labeled players / ball / hoop, ByteTrack tracks with jersey numbers, a 2D court model, per-player shot stats, a coach dashboard, a live score overlay and a privacy blur. Built on Aug 28 in parallel Claude Code sessions; roles, contracts and GPU schedule in `docs/ORCHESTRATION.md`, per-stage commands in `docs/VISION.md`, measurements in `docs/RESULTS.md`, data handling in `docs/PRIVACY.md`.

| Stage | Where | State |
|---|---|---|
| Frame extraction, auto-labeling (Grounding DINO people, ball/hoop detector), false-ball cleaner, YOLO11n fine-tune, ball fine-tune on 80 coach-labeled frames | `vision/label/` | 300 frames labeled, 3395 boxes; best.pt val mAP50 0.72 (player 0.96, hoop 0.97, referee 0.76, ball 0.21); ball_hoop_v2.pt: false balls 36 to 8 per 40 held-out frames, recall 43 to 53 % |
| Detection, ByteTrack, team colors, overlay video | `vision/track/` | 0.08 s/frame alone on the M3; best.pt persons give 25 % fewer id switches than COCO |
| Jersey numbers (EasyOCR, vote per track, merge ids into players) | `vision/numbers/` | runs; 1 of 75 tracks numbered on the last dev60 run |
| Court calibration, camera propagation across pans and cuts, minimap | `vision/court/` | game10 calibrated, court lines stay on the floor through zoom and pan (`out/pitch/court_lines_10s.mp4`) |
| Possession, shot events, per-player FG stats | `vision/stats/` | 52 unit tests pass; game10 (10 min): 24 attempts, 10 made, Team A 4 of 12, Team B 6 of 12 (human-verified) |
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

## Measured on the full 10 minutes (game10, published 13:34)

15,001 frames in 2088 s (0.139 s/frame next to other jobs), 7.8 players per frame, ball in 34 % of frames, hoop in 80 %, 2850 track ids; 24 shot attempts, 10 made, 235 possessions, 616 players with distance, 48 camera cuts handled.

Human check by Sami of every called shot (`out/qa/stats_eval_game10.json`, `out/qa/verdicts_game10.json`): attempt precision 0.96 (22 of 23 real, 0 uncalled), made/miss 0.73 (16 of 22; 4 errors are rim-outs called makes), shooter team 0.77 (17 of 22), shooter person 0.40 (6 of 15 judged; 7 attributions changed after a fix, not re-judged). Live mode: phone 1080p30 in, 14 fps overlay with detection under full load. Ball v2 inside TRACK on the full game10 (QA vs the 120 hand labels): ball recall 0.31 to 0.53, precision 0.62 to 0.96, false balls 17 to 2; the shot detector then sees 31 attempts, 10 of them not yet human-checked; on the verified set the made/miss verdicts did not improve (70 % vs 73 %), so the frozen, human-verified v1 list stays the reference.

## Unverified

Jersey-number recall on game10 (the 40 % shooter identity above is the closest measure). The phone camera on the rig on stage (tested on the desk only). Hotkeys on a real keyboard during a live game. The 10 additional shot attempts found with ball v2 (not yet human-checked). The rig tracker on a free GPU (20 fps target, CPU measured at 6 fps under load).

## Known limits

Test footage is an edited broadcast with 76 cuts in 10 min (cut detection handles it, single-phone footage is the real input). Ball recall is the weak point: the fine-tuned player model does not detect the ball (AP50 0.21), so the ball stays with the separate detector; 80 coach labels and 12 minutes of training cut its false positives by 4x but recall is still about half of the frames, mostly the ball in a player's hands. Track ids still switch on fast pans; jersey numbers rarely read at 1080p distance. Referees and bench are not in `tracks.jsonl` since the player class switch, so the blur skips them until TRACK writes the `others` list.

Nothing touches `main`'s rig, CAD, firmware or pitch files except the appended "Analytics pipeline" section in `README.md`.
