# Demo risks, 16:00 stage path (RISK role, pass 1 at 12:55, pass 2 due 13:40)

Scope: live.py from the phone on the rig, score overlay, dashboard on the projector,
video pitch playback. Ranked by damage. Every fact below was checked on this laptop
at 12:50, read-only (nothing in out/ was written by the checks).

## Checks run and what they showed

| Check | Result |
|---|---|
| `.venv/bin/python -m vision.live.live --help` | works. `python vision/live/live.py --help` fails with ModuleNotFoundError: No module named 'vision' (the form written in docs/ORCHERSTRATION.md LIVE section) |
| `cv2.VideoCapture(i)` for i in 0..3, one frame each | 0 = FaceTime HD 1920x1080 @ 30 fps, frame ok. 1 = the iPhone (Continuity Camera, "iPhone von Sami", listed by system_profiler) opens but returns no frame, fps 1.0. 2 and 3 do not exist. `live.py --list-sources` therefore lists only `--source 0` |
| two processes on camera 0 at once | both get frames (macOS shares the camera), so rig tracker plus live.py on the same camera is fine |
| `out/dashboard.html` self-contained | yes except two relative files next to it: `overlay.mp4` (43 MB) and `minimap.mp4`; the only http string is the SVG namespace. Videos have controls, muted, preload=metadata, no autoplay. Both mp4 are H.264 (avc1, yuv420p), so Safari and Chrome play them |
| `vision/dashboard/build.py` duration | 0.23 s (built to a scratch path) |
| ball not seen for 10 s | no crash: possession returns None (possession.py:77), shots need a ball chain within 0.5 s (shots.py), score simply freezes; hotkeys keep working |
| live render path without models | `--source data/clips/dev60.mp4 --realtime --replay out/dev60/tracks.jsonl --max-seconds 6`: 174 frames rendered in 6 s, events file written |
| live.py on a camera that stops delivering | `cap.read()` false ends the loop (live.py, `if not ok: break`), the window closes and the process exits with "done: rendered N" |
| MJPEG / RTMP | MJPEG binds 127.0.0.1:8501 (no host flag), so a phone or second laptop cannot reach it. No `.env`, so RTMP is off |
| GPU sharing | ORCHESTRATION measurement: TRACK 0.08 s/frame alone, 1.7 s/frame with three model jobs. Seen today: the smoke TRACK step slowed to 0.28 s/frame next to one other run.py |
| laptop | AC power, 60 % charging, sleep blocked by caffeinate, disk 18 GB free (91 % used), camera permission granted to Terminal.app only (TCC), 3 commits not yet pushed |
| pitch assets | out/pitch/analytics_side_by_side.mp4 (11 MB), overlay_12s.mp4, results_sheet_10s.mp4 exist; pitch/deck/index.html not yet (DECK in progress); Malek's 60 to 90 s pitch video is filmed 14:30 to 15:15 and is not in the repo |

## Ranked risks

### 1. The phone camera delivers no frames and live.py quits
What breaks: `--source 1` opens but yields no frame right now. live.py then exits after the first failed read (or at start with "cannot open source"), the live window disappears on stage.
Likelihood: high. Continuity Camera drops when the phone locks, is too far from the Mac, gets a call or notification, or Wi-Fi/Bluetooth toggles.
How we notice: `--list-sources` shows only `--source 0`; the terminal prints "done: rendered 0" within a second.
30 s fallback: `.venv/bin/python -m vision.live.live --source 0 --minimap panel` (laptop camera, verified) or the file replay `--source data/clips/dev60.mp4 --realtime --replay out/dev60/tracks.jsonl --minimap panel`. Pre-stage: unlock the phone, landscape on the rig, Do Not Disturb on, run `--list-sources` 5 min before and only then choose the index.

### 2. Two model jobs on the GPU during the demo, both crawl
What breaks: live.py runs two YOLO models at 10 fps target. If the rig's `ball_tracker_yolo.py` (RIG brief: device mps) or a leftover TRACK job runs at the same time, detection drops toward 1 fps, the boxes lag seconds behind the players and shots are missed.
Likelihood: medium to high, the rig demo and the analytics demo are back to back or simultaneous.
How we notice: the score bar prints `det x fps`; below 5 the demo is lagging.
30 s fallback: rig on HSV (`software/ball_tracker.py`, no model) or `--device cpu`; live.py with the single nano model `--weights models/best.pt`; and before 15:50 `ps -axo command | grep .venv/bin/python`, kill every track/label/run_all process (monitor, numbers.watch, qa.watch are CPU only).

### 3. Wrong command form under stress
What breaks: docs/ORCHESTRATION.md shows `vision/live/live.py --source ...`; typed that way it dies with ModuleNotFoundError. Only `.venv/bin/python -m vision.live.live` works. Same for numbers, stats, qa modules.
Likelihood: medium.
How we notice: traceback immediately.
30 s fallback: pitch/analytics/stage_checklist.md has the correct `-m` form, read the command from there or from shell history. STATS could add the two-line sys.path shim run_all.py uses; that is their file.

### 4. The documented replay fallback mixes two videos
What breaks: stage_checklist.md says `--source data/clips/dev60.mp4 --realtime --replay out/tracks.jsonl`. out/tracks.jsonl is whatever TRACK ran last (game10 is scheduled), so boxes of another game get drawn on dev60.
Likelihood: high if the fallback is used as written.
How we notice: boxes float beside the players from the first second.
30 s fallback: `--replay out/dev60/tracks.jsonl` (exists since 12:08, matches dev60). PITCH should change the line.

### 5. Dashboard or deck opened without their neighbours
What breaks: dashboard.html needs overlay.mp4 and minimap.mp4 in the same folder; the deck will reference out/pitch/... by relative path. A copy of the html alone (AirDrop, USB, other laptop, or opening out/smoke_pipeline/dashboard.html) shows empty video boxes. The videos also wait for a click (controls, no autoplay) and minimap.mp4 is 20 s next to a 60 s overlay.
Likelihood: medium.
How we notice: black video frames on the projector.
30 s fallback: open from this laptop, `open out/dashboard.html`, click play once before going on; if a video stays black play out/pitch/analytics_side_by_side.mp4 in QuickTime. Rebuild costs 0.23 s (`vision/dashboard/build.py`).

### 6. The ball is not detected on stage
What breaks: phone at 5 m, room light, a ball of 20 to 40 px: the ball model misses it for seconds. Nothing crashes, but possession and shots stay empty and the score never moves by itself.
Likelihood: high (the models were trained on 1080p gym footage, not a hall with a phone).
How we notice: no ball box and no possession ring in the live window.
30 s fallback: the auto-with-veto story: keys 1/2 (+2) and 3/4 (+3) put the score on the bar, say "the system calls, the human corrects". Use a bright ball, walk the ball closer to the phone.

### 7. Hotkeys land in the wrong window
What breaks: OpenCV only sees keys while the live window is focused. After clicking the deck or the browser, 1 to 4 and z do nothing, arrow keys flip slides instead.
Likelihood: medium.
How we notice: score does not change after the key.
30 s fallback: click the live window title bar, press again. One person owns the keyboard, the other talks.

### 8. Camera permission dialog or black frames from another terminal
What breaks: macOS grants camera access per app; today only Terminal.app is allowed. Starting live.py from VS Code, iTerm or Warp triggers a permission dialog on stage or returns black frames.
Likelihood: low to medium.
How we notice: dialog, or `--list-sources` finds nothing.
30 s fallback: run from Terminal.app; if the dialog appears, click OK and restart the command.

### 9. Leftover jobs, disk and sleep at 16:00
What breaks: background loops from the day (numbers.watch, qa.watch, monitor, any TRACK run) keep CPU and GPU busy; the disk has 18 GB left while every overlay render is 40 to 100 MB; caffeinate processes hold the Mac awake now but not on stage.
Likelihood: medium.
How we notice: fans, `det` fps low, "no space left" in a log.
30 s fallback: at 15:50 kill the watchers and any TRACK process, `caffeinate -d &`, plug in AC, close the monitor tab; delete out/smoke_pipeline if space is needed (regenerable).

### 10. Promised outputs that do not exist yet
What breaks: MJPEG is reachable only on this laptop (127.0.0.1), RTMP is off without .env, the pitch video and the deck are not in the repo at 12:55.
Likelihood: certain for MJPEG-on-phone, medium for the rest.
How we notice: a phone browser cannot open http://<mac-ip>:8501.
30 s fallback: do not promise a phone view; mirror the live window to the projector. Keep the pitch video as a local file in the repo (pitch/ or out/pitch/) by 15:15, not only on a phone.
