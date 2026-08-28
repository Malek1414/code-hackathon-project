# Privacy (facts from the code, Aug 28, 2026)

**Where processing happens.** Every model in this repository runs on the
laptop that holds the footage (`device=mps` or `cpu` in `vision/`). No module
under `vision/` imports an HTTP client; the only network code is the live
mode (`vision/live/stream.py`), which serves an MJPEG preview on
`127.0.0.1:8501` (loopback only) and pushes RTMP solely when
`FOLLOWCAM_RTMP_URL` is set in `.env` (`vision/live/env.py`; the key is
masked in logs). Model weights are downloaded once from Hugging Face and
Ultralytics on first use; no footage or derived data is sent anywhere.

**What is stored.** Raw clips in `data/clips/`, extracted frames in
`data/frames/` (1 fps JPEGs), the auto-labeled training set in
`data/dataset/` (frames plus box coordinates), and derived files in `out/`:
`tracks.jsonl` (per frame: box, foot point, team color index, track id,
confidence; no names), `identities.json` (jersey number per track id),
`events.json` and `stats.json` (shots and per-player counts keyed by track id
or jersey number), `overlay.mp4` and `minimap.mp4`. No face crops, face
embeddings or names exist anywhere in the pipeline; identity is jersey
number plus team color.

**What is blurred.** `vision/privacy/blur.py` reads only `out/tracks.jsonl`
(no model) and pixelates, in every frame, the top 22 percent of every
person box widened by 20 percent on each side (mosaic 12 px plus Gaussian
blur), for players, referees and anyone else TRACK boxed. When
`out/court_calib.json` exists (`--calib`), a person whose foot point projects
more than 1 m outside the 28 x 15 m court is blurred in full (bench, table,
spectators). Output is H.264 via the bundled ffmpeg
(`-c:v libx264 -pix_fmt yuv420p -movflags +faststart`), for the overlay or
the raw clip (`--stride` from `out/tracks_meta.json`). The blurred file is
the one that leaves the laptop; raw clips do not. Measured on
`out/overlay.mp4` (60 s, 1501 frames): 11,579 head regions blurred in 125 s
on CPU, output `out/overlay_blurred.mp4`. Known gap (12:55): since TRACK
switched to the fine-tuned player class, referees and bench spectators are
no longer in `tracks.jsonl` and therefore not blurred; fixing it needs an
`others` box list in `tracks.jsonl` (referee and off-court detections) from
TRACK, which `blur.py` would then treat like players.

**How long.** `vision/privacy/retention.py` lists and, with `--apply`,
deletes `data/clips/*.mp4` and `data/frames/*.jpg` older than `--hours`
(default 24). It is a dry run by default and prints every file with age and
size before anything is deleted. It does not touch `data/dataset/`,
`models/` or `out/`, which contain no raw footage.

**Consent.** Footage of minors in German gyms is filmed with the club's
consent process, as for team photos; the product exports the blurred overlay
and the stats, not the raw video.
