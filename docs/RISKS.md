# Demo risks, 16:00 stage path (RISK role, pass 2 at 13:45; pass 1 was 12:55)

Scope: live.py from the phone on the rig, score overlay, dashboard on the projector,
video pitch playback, and the 14:30 freeze command. Ranked by damage. Every fact was
checked on this laptop at 13:40 to 13:45, read-only for other roles' files (the only
code change is in vision/run_all.py, PIPELINE's own file, see item 1).

## Checks run and what they showed (pass 2, changes vs pass 1 marked)

| Check | Result at 13:42 |
|---|---|
| live.py command forms | fixed: `.venv/bin/python vision/live/live.py --help` and `-m vision.live.live --help` both work (sys.path shim, line 37); docs/ORCHESTRATION.md LIVE section now shows the `-m` form |
| cv2 indices 0..3, one frame each | 0 = FaceTime HD 1920x1080@30 ok. 1 = iPhone: opened, first read returned no frame in my probe, but `live.py --list-sources` (2 s read timeout) got 1920x1080@30 from it seconds later: the phone delivers intermittently, depending on whether it is awake. 2, 3 do not exist |
| live.py on a silent camera | changed: reads run in a thread with timeout, the score bar shows "Kamera: kein Bild", the device is reopened, `--source auto` picks the first index that delivers. At start it retries 20 x 0.25 s, then exits "no frames from source" |
| replay fallback | pitch/analytics/stage_checklist.md line 23 still says `--replay out/dev60/tracks.jsonl`; that file no longer exists (TRACK archives to out/dev60_v2, _v3, _v4). FileNotFoundError verified. vision/live/README.md uses out/dev60_v2/tracks.jsonl, which exists; `--replay out/dev60_v4/tracks.jsonl` verified working |
| live render rate, replay, CPU only, 6 s realtime | `--minimap off`: 144 frames rendered (24 fps). `--minimap panel` with the per-frame calibration: 57 frames (9.5 fps), frames are dropped to keep pace |
| `out/dashboard.html` | 312 kB, game10 (events 24 shots, 23 active players), only http string is the SVG namespace, references `overlay.mp4` next to it (471 MB, 10 min, H.264 avc1, controls, muted, no autoplay). No minimap: the builder leaves it out because out/minimap.mp4 is 60 s (dev60) and the overlay is 10 min. Jersey numbers: 0, because out/identities.json is still the dev60 file (builder logs "identities.json is for dev60, events are game10, jersey numbers ignored") |
| `vision/dashboard/build.py` duration | 0.57 s to a scratch path |
| ball not seen for 10 s | unchanged: possession None, no shots, score freezes, no crash. New: an unconfirmed basket flashes "looked in, press 1 or 2" and adds nothing |
| GPU jobs at 13:42 | `vision/label/train_ball.py` (LABEL) on the GPU, NUMBERS watch on game10 (CPU OCR, 563 crops to go at 13:02), pytest of STATS. RIG's ball_tracker_yolo.py now defaults to `--device cpu` |
| freeze command `make demo CLIP=data/clips/game10.mp4` | would have failed: TRACK's run.py writes archives only since 13:2x and touches the contract paths only with `--publish`, so run_all's TRACK step would end with "Ausgabe fehlt"; and a clean run would recompute TRACK on 10 min of video plus OCR. Fixed in vision/run_all.py at 13:44 (see item 1). Dry run now: TRACK, COURT, STATS, FRONTEND adopted, NUMBERS runs (identities is dev60), QA runs |
| pitch assets | pitch/deck/index.html exists (13:20), references ../../out/pitch/analytics_side_by_side.mp4 (20 s, dev60, H.264) and ../../viz/followcam_assembled_poster.png by relative path. No game10 side-by-side yet (freeze table: 14:00 to 14:15). Malek's pitch video not in the repo |
| laptop | AC, 77 % charging, disk 17 GB free (was 18 at 12:50; a game10 overlay is 471 MB, each dev60 archive ~150 MB), camera permission Terminal.app only, 4 commits not pushed, 2 files modified by other roles |
| MJPEG / RTMP | unchanged: 127.0.0.1:8501 only, no .env so RTMP off |

## Ranked risks

### 1. The 14:30 freeze command and the projector dashboard do not match the pitch material
What breaks: `make demo CLIP=data/clips/game10.mp4` must pass from a clean state at 14:30. Until 13:44 it would have failed on TRACK (archive-only default, no `--publish`) and then recomputed everything on the GPU for 30+ min. run_all now passes `--publish`, adopts outputs that already exist for the clip and checks the `clip` field of every json output. What is still open: NUMBERS on game10 is not finished, so the dashboard shows track ids instead of jersey numbers, and it has no minimap (60 s dev60 minimap versus 10 min overlay).
Likelihood: high for the numbers, medium for the minimap (COURT is on minimap_game10 in the freeze table).
How we notice: `make plan CLIP=data/clips/game10.mp4` lists what would still run; dashboard header says which clip each input comes from.
30 s fallback: ship the dashboard without numbers (it already ignores the mismatched identities.json) and show the 2D court from out/pitch/analytics_side_by_side.mp4 in QuickTime. Do not start NUMBERS on game10 after 15:00, it will not finish.

### 2. The phone camera is asleep when live.py starts
What breaks: index 1 opens and delivers nothing until the iPhone wakes up; at start live.py gives up after 5 s ("no frames from source"). Mid-run silence is handled now (score bar "Kamera: kein Bild", reopen).
Likelihood: medium (was high; the code is robust now, the phone is not).
How we notice: `--list-sources` says `no-frame` for the phone; the start message.
30 s fallback: wake the phone, `--source auto` (first camera that delivers), else `--source 0` (laptop camera, verified), else the file replay in item 4. Do Not Disturb on the phone, landscape on the rig, Continuity toggle on.

### 3. The live window stutters with the court panel
What breaks: with `--minimap panel` and a per-frame calibration the render loop manages 9.5 fps in replay on CPU; with the models running in the worker thread on MPS the main thread gets less, so the projector shows a slideshow and the boxes lag.
Likelihood: medium to high.
How we notice: `det x fps` in the score bar is fine but the video itself is choppy.
30 s fallback: restart with `--minimap off` (24 fps in replay) or `--minimap window` and drag the small window onto the projector only when talking about the court.

### 4. The documented replay fallback points to a deleted file
What breaks: stage_checklist.md line 23 uses `--replay out/dev60/tracks.jsonl`, which TRACK's archive change removed; the command dies with FileNotFoundError exactly when it is needed.
Likelihood: high if the checklist is followed verbatim.
How we notice: traceback at start.
30 s fallback: `--replay out/dev60_v2/tracks.jsonl` (vision/live/README.md) or out/dev60_v4/tracks.jsonl (newest, verified). PITCH should update the line; TRACK should not delete out/dev60_v2 to v4 before 17:00.

### 5. A model job is still on the GPU at 16:00
What breaks: LABEL's train_ball.py was on the GPU at 13:42 and NUMBERS OCR eats CPU; anything left over at 16:00 drops live.py from 10 to 1 detection fps. Good news: RIG's yolo tracker defaults to cpu now.
Likelihood: medium.
How we notice: `ps -axo pid,command | grep .venv/bin/python`; `det` fps below 5.
30 s fallback: at 15:50 kill every vision/track, vision/label, vision/numbers, run_all process (monitor and qa.watch are harmless), then start live.py alone.

### 6. Dashboard or deck opened away from the repo
What breaks: dashboard.html needs overlay.mp4 (471 MB) beside it, the deck needs ../../out/pitch/ and ../../viz/. A copy of the html alone (AirDrop, USB, other laptop) shows empty boxes; the overlay needs a click (no autoplay).
Likelihood: medium.
How we notice: black video frames.
30 s fallback: `open out/dashboard.html` and `open pitch/deck/index.html` from this laptop, click play once before going on; QuickTime with out/pitch/analytics_side_by_side.mp4 as the backup picture.

### 7. Hotkeys land in the wrong window
Unchanged: 1 to 4 and z only work while the OpenCV window is focused; after clicking the deck, arrow keys flip slides and the score does not move. Fallback: click the live window title bar, press again; one person owns the keyboard.

### 8. The ball is not detected on stage
Unchanged: nothing crashes, possession and shots stay empty, the score never moves by itself. An unconfirmed basket now flashes "looked in, press 1 or 2". Fallback: keys 1/2 (+2), 3/4 (+3), bright ball, walk it closer to the phone, and say "the system calls, the human corrects".

### 9. Wrong terminal, permission dialog, disk
Unchanged plus disk: camera access is granted to Terminal.app only (VS Code, iTerm, Warp would prompt on stage). Disk went 18 to 17 GB in 50 min; every game10 TRACK run adds 471 MB, every dev60 archive 150 MB. Fallback: Terminal.app; delete out/smoke_pipeline and out/dev60_v3 if space runs out (regenerable, v2 and v4 are the replay files).

### 10. Promised outputs that do not exist
Unchanged: MJPEG reachable only on this laptop, RTMP off without .env, Malek's pitch video not in the repo, no game10 side-by-side yet. Fallback: do not promise a phone view; keep the pitch video as a local file in the repo by 15:15.
