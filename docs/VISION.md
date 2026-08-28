# Analytics pipeline (vision/)

Video in, per-player stats out. Every command below is taken from the module's
own `--help` on Aug 28, 2026; every number is a measurement from a log or a
session report (`docs/RESULTS.md`, `docs/ORCHESTRATION.md`, `out/*.log`).
Modules that import `vision.*` must be run as modules (`-m`); the file form
`vision/x/y.py` fails for those with ModuleNotFoundError. Contracts (file
formats, class ids) are in `docs/ORCHESTRATION.md`.

Environment: `.venv/bin/python` (torch 2.13 with MPS, ultralytics 8.4,
supervision 0.29, transformers 5.16, opencv 5.0, EasyOCR, imageio-ffmpeg).
ffmpeg binary: `.venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"`.
Weights in `models/`, clips in `data/clips/`, outputs in `out/`; all three
are gitignored.

## One command

```
.venv/bin/python -m vision.run_all --clip data/clips/dev60.mp4 [--weights models/best.pt] [--calib out/court_calib_dev60.json] [--skip numbers,qa] [--force] [--dry-run]
make demo CLIP=data/clips/dev60.mp4      # same thing
make smoke                               # 10 s cut of dev60 through the pipeline, contract check (PASS in 132 s on 12:46)
```
`run_all` runs TRACK, NUMBERS, COURT (only with a calibration), STATS, QA,
FRONTEND in that order, skips a step whose outputs are newer than its inputs
(stamps in `out/.run_all/`), and `--wait-for-gpu SEK` waits for other model
jobs first.

## Labeling (`vision/label/`, LABEL)

```
.venv/bin/python vision/label/extract_frames.py data/clips/game10.mp4 --fps 1 --out data/frames
.venv/bin/python vision/label/autolabel.py --frames data/frames --dataset data/dataset --every 2 --device mps
.venv/bin/python vision/label/clean_balls.py --dataset data/dataset
.venv/bin/python vision/label/preview.py            # out/label_preview.jpg
```
In: a clip. Out: `data/frames/f_%05d.jpg` (1 fps, 1920x1080), a YOLO dataset
`data/dataset/{images,labels}/{train,val}` plus `data.yaml`, classes
0 player, 1 ball, 2 hoop, 3 referee. People come from Grounding DINO
(`grounding-dino-tiny`, prompt `basketball player. referee.`, threshold 0.2),
ball and hoop from `models/ball_hoop_avishah.pt` (imgsz 1280, ball conf 0.45,
one ball per frame). `clean_balls.py` deletes ball boxes that keep the same
offset to the hoop across 4 or more frames (wall fixtures). Measured: 600
frames extracted, 300 labeled in 958 s (3.2 s per frame with GPU contention),
3395 boxes after 72 false balls were removed.

## Training (`vision/label/train.py`, LABEL)

```
.venv/bin/python vision/label/train.py --epochs 10 --imgsz 960 --batch 4 --device mps --time 0.4167
```
In: `data/dataset/data.yaml`. Out: `runs/label_yolo11n/`, best weights copied
to `models/best.pt`, val mAP per class printed. `--time` (hours) overrides the
epoch count in ultralytics. Measured: 16 epochs in 25.4 min, val mAP50 0.723
(player 0.957, hoop 0.973, referee 0.755, ball 0.207), 5.5 MB.

## Tracking (`vision/track/run.py`, TRACK)

```
.venv/bin/python vision/track/run.py --video data/clips/dev60.mp4 --stride 2 --person-weights models/best.pt --conf-ball 0.35 --out out/tracks.jsonl --overlay out/overlay.mp4
```
In: a clip, optional `--events`, `--identities`, `--calib`, `--cuts`. Out:
`out/tracks.jsonl` (one line per processed frame: players with bbox, foot,
team, id; ball; hoops), `out/tracks_meta.json`, `out/overlay.mp4`; the run
writes to `out/<clip>/` and publishes the contract paths atomically at the
end. Two models: persons (`--person-weights`, COCO yolo11s or `best.pt`) and
ball/hoop (`ball_hoop_avishah.pt`); `--weights` switches to one contract model.
ByteTrack (or `--tracker botsort`), team by jersey color (`--team-mode
rules|kmeans`). Measured: 0.08 s per frame alone on the M3, 0.26 to 0.45 with
other model jobs (`out/track_dev60.log`); on 526 dev60 frames `best.pt`
persons give 9.3 players per frame and 101 track ids against 12.5 and 134
with COCO yolo11s (25 percent fewer id switches); ball in 33 percent of
frames at conf 0.45, 43 percent at 0.30.

## Jersey numbers (`vision/numbers/`, NUMBERS)

```
.venv/bin/python -m vision.numbers.read [--tracks out/tracks.jsonl] [--min-s 10] [--max-crops 12]
.venv/bin/python -m vision.numbers.merge [--reads out/numbers_reads.json] [--out out/identities.json]
.venv/bin/python -m vision.numbers.watch     # both, again whenever tracks.jsonl changes
```
In: `out/tracks.jsonl` and the clip from `tracks_meta.json`. Out:
`out/numbers_reads.json`, `out/numbers_preview.jpg`, `out/identities.json`
(track id to team and number, tracks merged into players). EasyOCR on CPU,
digit allowlist, torso crops (rows 15 to 60 percent of the box) at 256 px,
0.8 s per crop; a number needs 2 reads and 60 percent of the vote mass.
Measured on the 12:54 dev60 run: 1 of 75 tracks longer than 10 s got a number.

## Court model (`vision/court/`, COURT)

```
.venv/bin/python vision/court/calibrate.py data/clips/dev60.mp4            # click landmarks per keyframe, writes out/court_calib_dev60.json + out/court_calib.json
.venv/bin/python vision/court/propagate.py data/clips/dev60.mp4 --tracks out/tracks.jsonl [--preview]
.venv/bin/python vision/court/minimap.py --tracks out/tracks.jsonl --calib out/court_calib.json --out out/minimap.mp4
.venv/bin/python vision/court/project.py --calib out/court_calib.json --tracks out/tracks.jsonl   # distance_m per player
```
In: clip, clicks, tracks. Out: `out/court_calib.json` (`H_px_to_m` per
keyframe), `out/court_H_<clip>.npz` (one homography per frame from sparse
optical flow chained between keyframes, players masked out; camera code from
Courtside), `out/minimap.mp4` (28 x 15 m court, 40 px per m, team colors,
ball trail), `out/cuts_<clip>.json`. Measured: 76 cuts or dissolves detected
in game10 (`out/court_chain_game10.log`). No `out/court_calib.json` existed
at 13:00, so the minimap renders the court without projected players and
the dashboard has no shot positions in metres.

## Stats (`vision/stats/`, STATS)

```
.venv/bin/python -m vision.stats.build --tracks out/tracks.jsonl --clip data/clips/dev60.mp4 [--calib out/court_calib.json] [--identities out/identities.json]
.venv/bin/python -m vision.stats.build --fixture made      # synthetic smoke run
```
In: tracks, optional calibration, identities, cut list. Out: `out/events.json`
(shots with made/miss, shooter, hoop box; possessions) and `out/stats.json`
(per player FGA, FGM, FG%, possession seconds, distance in m when calibrated).
Possession = nearest foot to the ball, shot = ball enters the hoop zone from
above, made = ball seen below the rim within 0.5 s; bench players and
spectators filtered unless `--no-court-filter`. Measured: 46 unit tests; 1
shot attempt, 0 made, on the 60 s dev clip.

## Dashboard (`vision/dashboard/build.py`, FRONTEND)

```
.venv/bin/python vision/dashboard/build.py [--events out/events.json] [--stats out/stats.json] [--team-a "BC Lions Moabit"] [--team-b "Weddinger Wiesel"]
```
Out: `out/dashboard.html`, self-contained, references `overlay.mp4` and
`minimap.mp4` by relative path (keep the three files together). Builds in
0.23 s; every missing input becomes a note on the page.

## Live mode (`vision/live/live.py`, LIVE)

```
.venv/bin/python -m vision.live.live --list-sources
.venv/bin/python -m vision.live.live --source 0 --minimap panel
.venv/bin/python -m vision.live.live --source data/clips/dev60.mp4 --realtime --replay out/dev60/tracks.jsonl --minimap panel
```
Camera or file in; detection in a worker thread (about 10 fps target on MPS,
`--process-fps`), every frame rendered with the last boxes; StatsEngine calls
made/missed shots, the score bar adds +2 for the shooter's team, hotkeys
1/2 (+2), 3/4 (+3), z (undo), q (quit) veto. Out: preview window, MJPEG on
`127.0.0.1:8501/stream`, RTMP push only when `FOLLOWCAM_RTMP_URL` is set in
`.env`, `--events-out`. Measured: replay of 6 s renders 174 frames in 6 s;
the iPhone via Continuity Camera opened but delivered no frames at 12:50,
the laptop camera did.

## QA and monitor (`vision/qa/`, `vision/monitor/`)

```
.venv/bin/python -m vision.qa.watch --once      # out/qa/: ball recall sheet, shot sheets, number crops, index.html
.venv/bin/python -m vision.monitor.serve         # status board on http://127.0.0.1:8600
```

## Privacy (`vision/privacy/`, PRIVACY)

```
.venv/bin/python vision/privacy/blur.py --video out/overlay.mp4 --tracks out/tracks.jsonl --out out/overlay_blurred.mp4
.venv/bin/python vision/privacy/retention.py --hours 24 [--apply]
```
Heads blurred from the tracking boxes alone (no model, CPU); off-court boxes
in full once a calibration exists. Measured: 11,579 heads in 125 s on the 60 s
overlay. Referees and bench are only blurred once TRACK writes them into the
optional `others` list. See `docs/PRIVACY.md`.
