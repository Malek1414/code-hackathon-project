# Demo risks, 16:00 stage path (RISK role, pass 3 at 15:25; passes 1 and 2 were 12:55 and 13:45)

Scope: live.py from the phone on the rig with the servo link, score overlay and widgets, dashboard on
the projector, video pitch playback, the freeze command. Ranked by damage. Every fact was checked on
this laptop between 15:20 and 15:25, read-only for other roles' files.

## Checks run and what they showed (pass 3)

| Check | Result |
|---|---|
| servo link, `--dry-serial` on a 6 s replay (dev60_v4) | 12 commands in 6 s (limit 20/s), angles 67 to 90 (limit 40 to 140), ball lost: holds, then back to `A90`. `ls /dev/cu.usb*`: no Arduino connected to this Mac at 15:20, the real serial path is unverified |
| widgets `broadcast/assets/` | all seven ids present as 1920x1080 PNG: score_bug, made_flash, player_card, team_overview, lower_third, end_summary with alpha; heat_map.png has no alpha (full frame, acceptable). Sources and `layout.json` in broadcast/widgets/ |
| `broadcast/config.json` | does not exist. live.py falls back to "Team A" and "Team B" (verified in --help and code). No key-like content anywhere in broadcast/ |
| `live.py --help` | has `--bind`, `--team-a/--team-b/--color-a/--color-b`, `--serial/--dry-serial`, `--panel-every`, `--loop` |
| `/state.json` from a running replay (`--bind 127.0.0.1`, MJPEG on) | answers in 1 s: `{"schema": 1, "brand": "Big Ball Baller", ..., "teams": [Lions #2f6fdb, Wiesel #c8102e], "camera": "ok", "pan_deg": null}`, Content-Type application/json, Cache-Control no-cache (spec says no-store, cosmetic); `out/live_state.json` written; `/stream` answers 200 |
| cv2 indices 0..3, `--list-sources` | unchanged: 0 = FaceTime ok; 1 = iPhone opens, no frame in my 2 s probe, ok in live.py's probe seconds later; runbook now says the phone wakes within ~10 s and is index 1 |
| `out/dashboard.html` (15:12) | game10, 24 shots, 23 active players, 20 with jersey numbers (identities.json is game10 now), overlay.mp4 (462 MB, 10 min, H.264) and minimap.mp4 (10 min game10_v2, H.264) referenced next to it, controls and muted, no autoplay, only http string is the SVG namespace |
| pitch and deck | pitch/video/: 01_prototype_10s, 02_ingame_20s, 03_combined_30s, analytics_segment_37s (Malek's video is in the repo). pitch/deck/index.html: 5 slides, no placeholders left, references ../../out/pitch/analytics_side_by_side.mp4 (the 12:44 dev60 cut, 20 s) and ../../viz/followcam_assembled_poster.png; newer out/pitch/analytics_segment.mp4 (15:00) is not used by the deck |
| `make plan CLIP=data/clips/game10.mp4` | TRACK, NUMBERS, COURT, STATS, FRONTEND adopted, only QA would run (CPU sheets); run_all unchanged since pass 2 |
| GPU and CPU jobs at 15:20 | vision/track/overlay.py on game10_v2, vision/track/run.py on data/clips/whatsapp_1515.mp4 (new clip from 15:15), numbers.watch on out/tracks.jsonl, vision.live.celebrate render, monitor, qa.watch |
| laptop | **on battery, 59 %, discharging** (was on AC at 13:40), disk 15 GB free (17 at 13:40), camera permission Terminal.app only, 0 commits unpushed, untracked vision/live/celebrate.py |
| replay fallback files | out/dev60_v2 to v5 exist; checklist and runbook now use out/dev60_v5/tracks.jsonl (14:59, best.pt + ball_hoop_v2), verified present |

## Ranked risks

### 1. The laptop is on battery with four model jobs running, 35 min before the demo
What breaks: MPS on battery throttles, the battery drops fast under load (59 % at 15:20), and a dead or
sleeping Mac ends the demo. Two TRACK jobs, the OCR watcher and a celebration render are alive right now.
Likelihood: high if nothing changes.
How we notice: `pmset -g batt` says discharging; fans; `det x fps` in the score bar below 5.
30 s fallback: plug in the charger now, not at 15:50. At 15:50 kill everything except monitor and qa.watch
(`ps -axo pid,command | grep .venv/bin/python`), `caffeinate -d &`, then start live.py alone.

### 2. The score bug says "Team A" and "Team B"
What breaks: broadcast/config.json does not exist and the stage command in the checklist
(`--source 1 --minimap panel`) carries no `--team-a/--team-b/--color-a/--color-b`, so the widgets show
the placeholders and default colors on the projector.
Likelihood: high as written.
How we notice: the first frame of the live window.
30 s fallback: add `--team-a "<name>" --team-b "<name>" --color-a "#2f6fdb" --color-b "#c8102e"` to the
command, or write broadcast/config.json with team_a, team_b, color_a, color_b (FRONTEND owns the menu).
Names are never guessed, so somebody has to type them before 15:55.

### 3. The phone camera is asleep when live.py starts
Unchanged from pass 2: index 1 delivers only once the iPhone is awake; live.py waits 15 s for a first
frame, then exits. Fallback: wake the phone, `--source 1` or `--source auto`, else `--source 0`, else the
replay `--source data/clips/dev60.mp4 --realtime --replay out/dev60_v5/tracks.jsonl` (file verified).

### 4. Rig tracker and live.py on the same GPU
What breaks: the checklist starts `ball_tracker_yolo.py --device mps` next to live.py's two models;
measured earlier, one neighbour job triples the frame time. The checklist has the cpu fallback but nobody
has measured the rig tracker's fps on either device yet.
Likelihood: medium.
How we notice: rig tracker window fps below 10, or `det` in the score bar below 5.
30 s fallback: rig tracker `--device cpu` or Malek's HSV ball_tracker.py; live.py `--weights models/best.pt`.

### 5. The serial path has never run on hardware from live.py
What breaks: no Arduino is connected to this Mac at 15:20, so `--serial <port>` (opening the port,
115200, the slew limit on real hardware, `--invert-pan` direction) is untested; only `--dry-serial` is.
Likelihood: medium.
How we notice: `ls /dev/cu.usb*` empty, or the rig turns away from the ball.
30 s fallback: `--invert-pan`; if the port is missing, run live.py without `--serial` and let
software/ball_tracker.py (Malek's path) drive the rig from the laptop camera.

### 6. Court panel render rate
Pass 2 measured 9.5 fps with the panel; since then LIVE renders the panel every 3rd frame
(`--panel-every`). Not remeasured with the models running. Fallback stays `--minimap off` or `window`.

### 7. Dashboard or deck opened away from the repo
Unchanged: dashboard.html needs overlay.mp4 (462 MB) and minimap.mp4 beside it, the deck needs
../../out/pitch/ and ../../viz/. Note the deck still shows the 12:44 dev60 side-by-side, not the 15:00
game10 segment. Fallback: open both from this laptop, click play once before going on, QuickTime with
pitch/video/03_combined_30s.mp4 as the backup picture.

### 8. Hotkeys land in the wrong window
Unchanged: 1 to 4, z, t, b, e only work while the OpenCV window is focused. Fallback: click the live
window title bar first; one person owns the keyboard.

### 9. Ball not detected on stage
Unchanged: nothing crashes, possession and shots stay empty, the score moves only by hotkeys. The servo
holds and returns to centre after 3 s (verified in the dry run). Fallback: bright ball, walk it closer,
keys 1/2 and 3/4, say "the system calls, the human corrects".

### 10. Small things that bite late
Camera permission is granted to Terminal.app only (VS Code, iTerm, Warp would prompt on stage).
vision/live/celebrate.py is untracked, so a PR merge or a checkout loses it. heat_map.png has no alpha
and /state.json sends Cache-Control no-cache instead of no-store, both harmless today. Disk 15 GB.
