# Role FOOTAGE (assigned 12:05 by ORCH, session samimagdouli-61)

Read `docs/ORCHESTRATION.md` first (roles, GPU schedule). Work only inside
`vision/footage/` and `data/clips/` (never delete existing clips).

Problem: our current test video (`data/clips/moabit_full_1080p.mp4`, TAMPA 2 FILMS)
is an EDITED production with close-ups, scoreboard inserts and dissolves, which breaks
camera tracking and starves shot detection. The product's real input is one phone on
a tripod at the sideline, so we need continuous single-camera footage.

Tasks, in order:

1. The Landesliga BW game "TV Bad Säckingen beim USC Freiburg 3" (YouTube id
   EzBvE0ew5yM, uploader stammix, 1080p60) is being downloaded by ORCH to
   `data/clips/saeck_full_1080p.mp4` (1.24 GB, done ~12:12). If it is missing, run:
   `export PATH=$HOME/.deno/bin:$PATH; yt-dlp --js-runtimes deno --ignore-config -f 299 -N 4 --http-chunk-size 10M -o "data/clips/saeck_full_1080p.%(ext)s" "https://youtu.be/EzBvE0ew5yM"`
2. Check continuity of BOTH full videos with ffmpeg scene detection (binary:
   `.venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"`;
   use `-vf "scale=480:-1,select='gt(scene,0.35)',showinfo" -f null -`) and count cuts
   per 10 minutes. Also extract a 1 fps contact sheet (6x6 tiles) of a 3-minute window
   of each and LOOK at it (Read the jpg) to confirm: single wide camera, half court
   visible, hoop in frame.
3. If the Säckingen game is continuous single-camera: cut `data/clips/saeck10.mp4` =
   a 10-minute window of live play (skip warm-up) and `data/clips/saeck60.mp4` = 60 s
   containing at least two shot attempts. Find shots cheaply on CPU only:
   `models/ball_hoop_avishah.pt` via ultralytics `predict(device="cpu", imgsz=960)` on
   1 fps frames of the candidate window; a shot candidate = ball box within 1.5
   hoop-widths of a hoop box. Write `vision/footage/shots_candidates.json` with
   timestamps. Never use MPS (`device="cpu"` only), the GPU is scheduled for other jobs.
4. Report to ORCH via `SendMessage(to: "samimagdouli-61", ...)`, first line
   "FOOTAGE ...": cuts per 10 min for both videos, verdict, clip paths, shot
   timestamps. Target: verdict by 12:30.

## Verdict by ORCH, 12:20 (task closed, do not start it)

Säckingen (`data/clips/saeck_full_1080p.mp4`, downloaded): one wide camera from high
up at the far side, no editing, but players are ~50 px tall and the ball ~8 px at
1080p, hoops tiny at the far wall. 25 scene-change triggers per 10 min are zooms and
fast pans by the operator, not cuts. Unusable for ball and shot detection today.
Moabit `game10.mp4`: 0 hard cuts at scene threshold 0.35, dissolves and close-ups are
handled by COURT's cut detection. **Decision: Moabit stays the material for all runs.**
Säckingen remains as the "what a phone on a balcony looks like" example for the pitch.
