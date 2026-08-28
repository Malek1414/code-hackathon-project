# Vision pipeline — orchestration (Aug 28, 2026, 11:40 → freeze 14:30)

Sami's track at the hackathon: **video → labeled players/ball/hoop → 2D court
model → per-player stats (shots, FG%)**. This is the "analytics platform" half
of the pitch; the servo rig is the capture half (Malek).

Nine Claude Code sessions work in parallel on this directory. Each session
owns ONE role below, works only inside its owned paths, and reports to the
orchestrator session (`samimagdouli-61`) via SendMessage when a milestone is
done or it is blocked. Sami can talk to any session directly in its terminal.

| Role | Owns | Deliverable |
|---|---|---|
| **ORCH** (Sami's main terminal) | `docs/`, integration, git, pitch assets | everything merged + demo runs |
| **LABEL** | `vision/label/`, `data/frames/`, `data/dataset/`, `models/` | auto-labeled YOLO dataset + fine-tuned `models/best.pt` |
| **TRACK** | `vision/track/`, `out/tracks.jsonl`, `out/overlay.mp4` | detection + ByteTrack + team colors + annotated video |
| **STATS** | `vision/stats/`, `out/events.json`, `out/stats.json` | ball possession, shot events (made/miss), per-player FG stats from `tracks.jsonl` |
| **COURT** | `vision/court/`, `vision/dashboard/`, `out/court_calib.json`, `out/minimap.mp4`, `out/dashboard.html` | homography → 2D minimap + coach dashboard |
| **LIVE** (added 11:50, owned by the STATS session) | `vision/live/` | live pipeline: camera/phone → detect+track → running score + stats overlay → local preview + optional RTMP push (Twitch/YouTube) with ~5 s delay |
| **MONITOR** (added 11:57) | `vision/monitor/` | local status board at 127.0.0.1:8600: labeling progress + latest labeled frame, training curve, tracking progress, shots, calibration state, log tails |
| **FRONTEND** (added 11:57, takes `vision/dashboard/` from COURT) | `vision/dashboard/`, `out/dashboard.html` | coach-facing analytics page for the judges: score, timeline, shot chart, player table, overlay + minimap side by side |
| **QA** (added 11:57) | `vision/qa/`, `out/qa/` | human verification sheets: per-shot frame strips with verdict checkboxes, ball recall sample, team assignment sample |
| **FOOTAGE**, **PITCH**, **PIPELINE**, **DOCS**, **PRIVACY**, **RIG**, **DECK**, **RISK** (added 12:10) | see `docs/tasks/<ROLE>.md` | continuous single-camera footage; pitch content; one-command pipeline; docs + PR body; face blur + retention; YOLO ball tracker for the rig; HTML deck; adversarial demo review |
| **NUMBERS** (added 12:05) | `vision/numbers/`, `out/identities.json` | jersey-number recognition per track, vote over time, merge track ids into real players (team + number) so stats are per player, not per track id |

Hard rules:
1. Python env: `.venv/bin/python` in this repo (torch 2.13 + MPS, ultralytics
   8.4, supervision 0.29, transformers, opencv, lap, scikit-learn). Never
   `pip install` into system Python. ffmpeg: `python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"`.
2. Test material: `data/clips/moabit_full_1080p.mp4` (BC Lions Moabit vs
   Weddinger Wiesel, Landesliga Berlin, 1080p50, 56 min) and the 10‑minute cut
   `data/clips/game10.mp4` (12:00–22:00 of it). Develop on the 10‑min cut,
   iterate on a 60 s sub‑clip while coding. **Never on the full file.**
3. Reuse, don't rewrite: `~/Desktop/APP/courtside/engine/` already has
   `court/homography.py` (solve_homography, track_camera with player mask),
   `court/geometry.py` (basketball court spec + landmarks), `detect/players.py`
   (YOLO11 at full res; imgsz 640 loses the ball), `analytics/ball.py`
   (clean_ball_track). Copy the functions you need into `vision/`, credit the
   source in a comment.
4. Interfaces below are the contract. Change them only through ORCH.
5. Commit small, on branch `sammy/vision`. Do not commit `data/`, `out/`,
   `*.pt`, `.venv/` (gitignored).
6. Timebox: a milestone that isn't green by its deadline gets the fallback,
   not another hour.

## Interfaces (the contract)

### Classes (YOLO ids)
`0 player`, `1 ball`, `2 hoop`, `3 referee`

### `out/tracks.jsonl` — one line per processed frame (TRACK writes, COURT reads)
```json
{"frame": 1250, "t": 25.0,
 "players": [{"id": 7, "bbox": [x1,y1,x2,y2], "foot": [x,y], "team": 0, "conf": 0.91}],
 "ball": {"bbox": [x1,y1,x2,y2], "center": [x,y], "conf": 0.6} ,
 "hoops": [{"bbox": [x1,y1,x2,y2]}]}
```
`foot` = bottom-center of bbox in pixels (the point COURT projects). `team` is
0/1 by jersey color, `-1` unknown. `ball` may be `null`.

### `out/events.json` — shot events (STATS writes, COURT/dashboard reads)
```json
{"fps": 50, "clip": "data/clips/game10.mp4",
 "shots": [{"t": 83.4, "frame": 4170, "player_id": 7, "team": 0,
            "made": true, "shooter_foot": [x,y], "hoop_bbox": [x1,y1,x2,y2]}]}
```

### `out/stats.json` — per-player table (STATS writes, dashboard reads)
```json
{"players": [{"id": 7, "team": 0, "fga": 5, "fgm": 2, "fg_pct": 0.4,
              "possession_s": 41.2, "distance_m": null}],
 "teams": [{"team": 0, "fga": 21, "fgm": 9}]}
```
`distance_m` is filled by COURT once `court_calib.json` exists (STATS calls
the same projection helper); `null` until then.

### `out/identities.json` — track id → real player (NUMBERS writes, STATS/FRONTEND/TRACK read)
```json
{"clip": "data/clips/dev60.mp4",
 "tracks": {"7": {"team": 0, "number": 12, "conf": 0.83, "votes": {"12": 9, "17": 1}, "reads": 10}},
 "players": [{"key": "A12", "team": 0, "number": 12, "track_ids": [7, 41, 88], "first_t": 0.0, "last_t": 58.2}]}
```
`key` = team letter (A = team 0, B = team 1) + number, e.g. `A12`. A track without a
confident number keeps its own key `A?7` (team letter, `?`, track id). STATS
aggregates `stats.json` per `key` (adds `"number"` and `"key"` to each player row)
and events carry `"player_key"`; FRONTEND shows `#12` instead of track ids.

### `out/court_calib.json` — homography (COURT writes, everyone reads via `vision.court.project.load_calibration`)
```json
{"clip": "data/clips/dev60.mp4", "frame": 1500,
 "court_m": {"length": 28.0, "width": 15.0},
 "points": [{"id": "corner_bl", "px": [x,y], "m": [0,0]}, ...],
 "H_px_to_m": [[...],[...],[...]], "reproj_err_px": 1.3,
 "frames": {"1500": {"points": [...], "H_px_to_m": [...], "H_m_to_px": [...], "reproj_err_px": 1.1}},
 "per_frame": "out/court_H_dev60.npz", "propagation": {"cuts": [...]}}
```
Top-level `frame`/`H_px_to_m` = the first hand keyframe. `frames` = one dict per hand
keyframe. Per-frame matrices (camera pans, cuts) live in the `.npz` named by
`per_frame`; frames outside a calibrated segment are NaN = uncalibrated. Nobody
reads matrices by hand: `cal = load_calibration("out/court_calib.json");
cal.project(frame, [[x, y], ...]) -> metres`, `cal.player_distances(tracks_path)`.
Per-clip files: `out/court_calib_<clip>.json`, `out/court_H_<clip>.npz`,
`out/cuts_<clip>.json`; `out/court_calib.json` = the last calibrated clip.

### `data/dataset/` — YOLO format (LABEL writes)
`images/{train,val}/*.jpg`, `labels/{train,val}/*.txt`, `data.yaml` with the
class list above. `models/best.pt` = fine-tuned detector. Until it exists,
TRACK uses `yolo11m.pt` (COCO: person=0 → player, sports ball=32 → ball) plus
a one-time Grounding DINO hoop box.

### LIVE mode (added 11:50)
`vision/live/live.py --source <cam index | video path> [--realtime]`:
- reuses TRACK's per-frame API (`vision.track.run.Tracker().step(frame) -> tracks line`)
  and STATS's incremental possession/shot logic; processes at ~10 fps, renders the
  output at source fps from the last known boxes (5 s ring buffer is the allowed delay).
- overlay: score bar (team A : team B, running clock), shot flash on a made basket,
  per-team FGA/FGM after a configurable interval; hotkeys `1`/`2` = +2 for team A/B,
  `3`/`4` = +3, `z` = undo (auto-with-veto: the system calls baskets, the human corrects).
- outputs: OpenCV preview window + MJPEG at `http://127.0.0.1:8501/stream` (for OBS or
  a phone), and if env `FOLLOWCAM_RTMP_URL` is set (from `.env`, never in code or git)
  an ffmpeg subprocess (`imageio_ffmpeg`) pushes the rendered frames as H.264/FLV to
  that RTMP URL. `--source data/clips/dev60.mp4 --realtime` simulates a live camera
  for the stage demo.

## GPU schedule (one M3, 16 GB, shared MPS; added 12:02)

Measured: TRACK alone 0.08 s/frame, under three concurrent model jobs 1.7 s/frame.
So model jobs run one at a time, in this order; a slot owner may start only when
the previous slot reports done to ORCH:

| Slot | Owner | Job |
|---|---|---|
| → ~12:10 | TRACK | dev60 tracking finishes (LABEL autolabel done 12:06) |
| 12:10 → ~12:35 | LABEL | YOLO11n fine-tune (exclusive MPS) |
| 12:35 → 12:40 | TRACK | compare best.pt vs yolo11s+avishah on dev60 frames 900+ |
| 12:40 → ~13:00 | TRACK | the 10-minute clip (Moabit game10 or Säckingen, ORCH decides by 12:35), once, with the better weights |
| 13:00 → | STATS/LIVE | live-mode runs from file and phone |

CPU-only jobs (optical-flow propagation, minimap, dashboard, QA sheets, monitor)
may run any time but at most one heavy one at a time. Before starting any job that
loads a model: `ps -axo pid,command | grep .venv/bin/python` and if another model
job is running, wait or ask ORCH.

## Milestones and deadlines

| Time | LABEL | TRACK | STATS | COURT |
|---|---|---|---|---|
| 12:00 | frames extracted 1 fps from `game10.mp4` → `data/frames/` (ORCH delivers the clips) | dev clip runs through YOLO11m + ByteTrack, boxes drawn, `tracks.jsonl` written | reader for `tracks.jsonl` + synthetic fixture; possession = nearest player to ball (foot distance), unit-tested | calibration tool: click ≥6 court points on frame 0, writes `court_calib.json`, shows reprojected court overlay |
| 12:45 | Grounding DINO labels for all frames (player, basketball, hoop, referee) → YOLO dataset, contact sheet `out/label_preview.jpg` | team assignment by jersey color (k‑means on torso crops) in `tracks.jsonl` | shot candidate = ball enters hoop zone from above; made = ball seen below rim inside hoop x-range within 0.5 s; `events.json` on the dev clip | `minimap.mp4`: 2D court with team-colored dots + ball from `tracks.jsonl` |
| 13:30 | YOLO11n/s fine-tune on MPS (imgsz 960, ≤15 epochs, timebox 30 min) → `models/best.pt`; mAP on val printed | `overlay.mp4` with ids, teams, ball trail; shot flashes read from `events.json` if present | `stats.json` per player + team; sanity-checked against a hand count on 2 minutes of video | dashboard: shot chart on court, per-player table from `stats.json`, minimap embedded |
| 14:15 | swap `best.pt` into TRACK, compare ball recall vs COCO | full `game10.mp4` processed | events + stats on full `game10.mp4` | dashboard reads final `events.json`/`stats.json` |
| 14:30 | **FREEZE.** ORCH assembles: overlay + minimap side by side for the pitch video | | |

LIVE milestones (STATS session, after its 12:45 events milestone): 13:15 preview
window with live boxes + score bar from a file in `--realtime`; 13:45 hotkeys, MJPEG,
RTMP push tested against a local ffmpeg listener (no real key needed); 14:15 run from
the phone camera (Continuity Camera index) on the tripod rig.

Fallbacks: no `best.pt` by 13:30 → COCO weights stay. Ball too unreliable for
shots → shots from hoop-zone + player proximity only, flagged "unconfirmed".
Panning camera → calibrate on 2–3 keyframes and interpolate H by time.

## Reporting
On every milestone, and on any blocker >10 min:
`SendMessage(to: "samimagdouli-61", message: "<ROLE> <milestone> done|blocked: <one line> — <path to artifact>")`.
