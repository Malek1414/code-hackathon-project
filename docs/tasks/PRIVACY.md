# Role PRIVACY (assigned 12:10 by ORCH, session samimagdouli-61)

Read `docs/ORCHESTRATION.md` and `reference/camera-score-tracker-vision.md` (privacy
decisions: German gyms, minors, short retention). You own `vision/privacy/`.

1. `vision/privacy/blur.py --video out/overlay.mp4 --tracks out/tracks.jsonl --out out/overlay_blurred.mp4`:
   blurs the head region (top 22 percent of every player bbox, widened by 20 percent)
   and every person-like box outside the court when a calibration exists, frame by
   frame, using only `out/tracks.jsonl` (no model loading; the GPU is scheduled for
   other jobs). Works on any video with the same frame indexing (overlay, or the raw
   clip with `--stride` from `out/tracks_meta.json`). Output H.264 via the
   imageio_ffmpeg binary (`-c:v libx264 -pix_fmt yuv420p -movflags +faststart`).
2. `vision/privacy/retention.py`: deletes raw clips and frames older than N hours from
   `data/` (dry run by default, `--apply` to delete), prints what it would delete.
   Never run `--apply` today.
3. `docs/PRIVACY.md`: half a page, in English: what is stored, for how long, what is
   blurred, what is processed on device, what is sent nowhere. Facts from the code only.
4. Report to ORCH via `SendMessage(to: "samimagdouli-61", ...)`, first line "PRIVACY ...".
