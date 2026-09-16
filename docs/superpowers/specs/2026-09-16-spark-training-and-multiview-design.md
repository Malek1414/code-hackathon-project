# FollowCam: Spark training pipeline and multi-view synced recording

Date: 2026-09-16. Status: draft for review.

## 1. Summary

Two pieces of work, one spec, because the second feeds the first.

**Part A: training pipeline on the DGX Spark.** A repeatable pipeline that
trains a four-class YOLO11 detector (player, ball, hoop, referee) on the
NVIDIA DGX Spark and measures it, with the ball as the class that has to get
better. Output: weights, a results table, a Core ML export handed to the iOS
work that follows. The app does not change in this part.

**Part B: multi-view synced recording and fusion.** Several phones record the
same game with a shared clock, the clips are uploaded after the game, and
the analysis pipeline aligns the views and fuses the ball across them so the
ball is never out of sight for long. This is the software-only product:
teams need phones and the app, not the servo pod. Every synced session is
also training data from our exact domain, which is why it sits next to Part A.

### Goals

- Ball recall good enough that auto-follow does not hunt: measured offline on
  a rig clip as the longest gap without a ball detection, target under 2 s,
  and validated in a live court test.
- A training run anyone on the team can repeat on the Spark with one command.
- A recording flow where a team with two or three phones gets a fused,
  multi-angle analysis back without buying hardware.

### Non-goals

- iOS on-device inference and the auto-follow state machine. Separate plan.
- Live streaming or real-time multi-view fusion. Recording first, upload after.
- 3D ball triangulation. Fusion happens in time and, where a court
  calibration exists, on the court plane. No stereo geometry.
- Pricing, accounts, billing. The upload is one shared secret per team for now.
- Panning the servo from multiple views.

## 2. Context

- The hackathon detector was YOLO11n at imgsz 960, trained 25 minutes on an M3.
  Players and hoop learned fine (mAP50 0.957 and 0.973), the ball did not
  (recall 0.077; 53 percent after a 120-frame hand-labeled fine-tune).
- From the rig position the ball is about 25 px in a 1080p frame at the far
  end of the court. Resolution and data volume are the two levers.
- The hackathon data (`game10.mp4`, `models/best.pt`, the hand labels) is not
  on this Mac and is not assumed. Training data is public datasets plus the
  repo's bootstrap set, then our own recordings as they appear.
- The Spark is a GB10 Grace Blackwell with 128 GB unified memory on ARM64
  Linux (DGX OS). Training runs in NVIDIA's PyTorch container. Core ML export
  needs macOS, so export runs on the Mac.
- Existing single-view pipeline: `vision/` (track, numbers, court, stats,
  dashboard). Contracts in `docs/ORCHESTRATION.md`: `tracks.jsonl` one line
  per frame with players, ball, hoops; `events.json`; `stats.json`.

---

## Part A: Spark training pipeline

### A1. Layout

```
ml/spark/
  SETUP.md              container tag, ssh, rsync, first-run commands
  classes.yaml          the canonical class map (0 player, 1 ball, 2 hoop, 3 referee)
  fetch_public.py       download public datasets, remap classes, write manifest
  merge_dataset.py      merge sources into data/spark/dataset, split by source video
  train.py              run one config on the Spark
  configs/              one yaml per run (s_1280.yaml, m_1280.yaml, ...)
  pseudolabel.py        label new frames with a trained model, write review sheet
  eval_ball.py          ball recall, FP per frame, longest gap on a clip
  export_coreml.py      runs on the Mac: .pt -> .mlpackage at 640 and 960
Makefile targets: spark-data, spark-train CONFIG=..., spark-eval, spark-export
```

Data and runs live on the Spark under `~/followcam/` (`data/spark/`,
`runs/spark/`), gitignored, synced with `rsync`. Only configs, scripts,
manifests and result JSON are committed.

### A2. Spark environment (`SETUP.md`)

- SSH over the LAN; hostname and user recorded in `SETUP.md`, no keys in git.
- `docker run` of `nvcr.io/nvidia/pytorch:<tag>` with the project folder
  mounted, `pip install ultralytics roboflow` inside; the exact tag is
  pinned once it is verified to see the GPU (`torch.cuda.is_available()`).
- A `make spark-smoke` target trains YOLO11n for one epoch on 50 images and
  proves the loop end to end before anything bigger runs.

### A3. Data

**Class map.** Fixed in `classes.yaml`: 0 player, 1 ball, 2 hoop, 3 referee.
Every source is remapped to this; sources missing a class contribute
negatives for it, which is fine for YOLO.

**Sources, round one.**

| Source | What it gives | Handling |
|---|---|---|
| Public Roboflow Universe basketball sets (2 to 3, chosen for ball + hoop + referee labels and a mix of broadcast and gym angles) | Volume, all four classes | `fetch_public.py` downloads via the Roboflow API key in `.env`, remaps class names to ids, writes `manifest.json` with per-source counts |
| Repo bootstrap set (`ml/data/dataset`, 1,115 frames, person + ball from two YouTube clips) | Gym-camera angle | Remapped 0 player, 1 ball. Hoop and referee absent: the round-one model pseudo-labels hoops on these frames, reviewed, then included in round two |
| Rig or multi-view recordings (Part B) | Exact domain | Pseudo-label + review loop (A6). Not required for round one |

**Split.** `merge_dataset.py` splits by source video or source dataset, never
by random frame, so consecutive frames do not leak across train and val.
Val is about 10 percent of images. Any clip from our own camera is held out
entirely and becomes the primary eval; it is never trained on until a newer
clip replaces it as eval.

**Manifest.** `data/spark/dataset/manifest.json` records, per source, image
and box counts per class and the split. `docs/RESULTS.md` quotes it.

### A4. Training recipe

- Models: YOLO11s and YOLO11m from COCO weights. Two runs per round.
- imgsz 1280. Batch chosen by `batch=-1` (ultralytics auto) on the first run
  and then pinned in the config.
- 100 epochs, `patience=20`, AdamW with the ultralytics auto lr, cosine
  schedule. Mosaic on, `scale=0.5`, `copy_paste=0.3` to multiply the ball,
  `close_mosaic=10`.
- Each config is a yaml in `configs/`; `train.py` takes `--config` and
  nothing else, so a run is fully described by a committed file. Run name
  equals config name; `runs/spark/<name>/` holds weights and ultralytics
  logs.
- After a run, `train.py` copies `best.pt` to `models/followcam_<name>.pt`
  and prints the per-class val table.

### A5. Evaluation

Two layers.

1. **Dataset val.** Ultralytics per-class precision, recall, mAP50,
   mAP50-95 on the held-out split, recorded per run.
2. **Ball report (`eval_ball.py`).** On a labeled set or on a clip:
   - recall at IoU 0.3 for conf in {0.25, 0.35, 0.45}
   - false positives per frame
   - on a clip: fraction of frames with a ball detection, and the
     **longest run of consecutive frames without one**, in seconds. This
     is the offline proxy for the field bar (under 2 s).
   Output `out/spark/eval_<model>_<clip>.json` and a contact sheet of the
   worst gaps so a human can see why the ball was lost.

**Acceptance for Part A.** The chosen model, on the newest held-out rig clip:
longest gap under 2 s at the chosen confidence, and then a live court test
in which the ball is not lost for more than 2 s. The court test is the final
word; the offline gate exists so court time is not wasted on a model that
cannot pass it.

### A6. Pseudo-label loop (`pseudolabel.py`)

For new footage (rig clips, Part B sessions):

1. Extract frames at 2 fps (5 fps in a window around detected shots).
2. Run the best current model, keep boxes above conf 0.5 for player, hoop,
   referee and above 0.35 for ball.
3. Write YOLO labels plus a review sheet (`out/spark/review_<clip>/`,
   contact sheets with boxes drawn, one page per 16 frames) and a
   `review.json` listing frames flagged for a human: ball conf between
   0.35 and 0.6, or no ball where a neighbouring frame had one.
4. A human edits or approves; the approved frames are added to the dataset
   as a new source with `origin: pseudo+review` in the manifest.

Cross-view rule from Part B: when view A has a ball at conf above 0.7 at
time t and view B has none at t, B's frame is put at the top of B's review
queue. Coordinates are not transferred between views.

### A7. Export handoff (`export_coreml.py`, Mac only)

`yolo export format=coreml imgsz=640` and `imgsz=960`, `nms=True`, half
precision, from the chosen `.pt`. Output `models/followcam_<name>_{640,960}.mlpackage`.
The script asserts it runs on macOS and prints the file sizes. Nothing
consumes these files in this spec.

---

## Part B: multi-view synced recording and fusion

### B1. User flow

1. One phone opens **New session**. The app shows a QR code (session id,
   host name, team name).
2. Other phones scan it. They join over Multipeer Connectivity (peer to peer,
   no Wi-Fi infrastructure or laptop needed). Each joiner measures its clock
   offset to the host with a few round trips, NTP style, and keeps the
   median.
3. Host taps **Record**. Every phone starts recording within a second of each
   other; exact alignment does not matter because every phone stamps its
   own recording start in host time.
4. Host taps **Stop**. Each phone writes a sidecar JSON next to its clip.
5. When on Wi-Fi, each phone uploads its clip and sidecar to the ingest
   service under the session id. The host phone also uploads the session
   manifest. Upload runs in the background and resumes after interruption.
6. The pipeline runs on the Spark when all expected views have arrived, or
   when someone runs it manually with whatever has arrived. The team gets a
   link to the dashboard.

A session with one phone is valid and is just today's single-view pipeline.

### B2. Sidecar and manifest

`<clip>.followcam.json` per phone:

```json
{"schema": 1, "session": "<uuid>", "view": "<uuid>", "device": "iPhone16,1",
 "host_offset_ms": 12.4, "host_offset_err_ms": 3.1,
 "record_start_host_ms": 1758030000123, "fps": 30, "width": 1920, "height": 1080,
 "orientation": "landscape", "label": "baseline left"}
```

`session.json` from the host: session id, team, date, list of expected
view ids with labels, which view is **primary** (defaults to the host).

### B3. App changes (iOS)

Kept to what the flow needs.

- `SessionCoordinator`: Multipeer host/join, QR generation and scanning,
  clock offset estimation, record and stop commands.
- `CameraManager`: record start captures host time into the sidecar; writes
  the sidecar next to the clip in the app's documents folder as well as
  saving the clip to Photos as today.
- `Uploader`: background `URLSession` upload of clip plus sidecar to the
  ingest service with the team's token, resumable, with a per-session
  progress list on screen.
- A **Sessions** screen: current session, joined phones and their offsets,
  upload state per clip.
- The rig link, HR strap and tap-to-track stay as they are in this spec.

### B4. Ingest service

`ingest/`, a small FastAPI app running on the Spark, reachable from outside
through Tailscale (the Spark joins the tailnet; phones on the tailnet or an
exposed funnel URL). Endpoints:

- `POST /sessions/{id}/views/{view}` multipart or chunked upload of the clip,
  `PUT .../sidecar`, `PUT /sessions/{id}/manifest`.
- `GET /sessions/{id}` status: which views arrived, pipeline state, dashboard
  URL.
- One bearer token per team in `ingest/teams.yaml` (gitignored, example
  committed). No accounts.

Files land in `data/sessions/<id>/<view>.mp4` plus sidecars. A watcher
starts `vision.multiview.run` when the manifest's expected views are all
present.

### B5. Alignment (`vision/multiview/align.py`)

1. Coarse offset per view from the sidecar: `record_start_host_ms` and
   `host_offset_ms`. Expected accuracy tens of milliseconds.
2. Refinement by audio: cross-correlate each view's audio envelope against
   the primary view within a window of plus or minus 500 ms around the
   coarse offset. The peak gives the final offset; the peak height is a
   confidence. A view whose audio refinement disagrees with the sidecar by
   more than 300 ms keeps the sidecar offset and is flagged.
3. Output `out/<session>/align.json`: per view, offset to the primary in ms,
   method, confidence. All later steps address time as primary-view
   milliseconds.

### B6. Per-view detection and tracking

Each view runs the existing `vision/track/run.py` unchanged with the Part A
weights, producing its own `tracks.jsonl`, and the existing court
calibration flow if someone has clicked landmarks for that view. Views are
processed sequentially on the Spark's GPU.

### B7. Fusion (`vision/multiview/fuse.py`)

Fusion works in time, not pixels.

- **Ball timeline.** For every 100 ms slot in primary time, collect the ball
  detections from every view (frame nearest the slot). The slot is *covered*
  if any view has a ball above conf 0.35. The slot's *best view* is the one
  with the highest ball confidence, with hysteresis: a switch needs the new
  view to win by 0.1 for 3 consecutive slots, so the timeline does not
  flap.
- **Coverage metric.** Fraction of covered slots, and longest uncovered gap
  in seconds. This is the number that says "never lose the ball" and it is
  reported per session next to the single-view figure for the primary view.
- **Events.** Each view's `vision.stats` run produces shot events. Fused
  events are the union, deduplicated within 1.5 s; for a duplicate, the
  made/miss verdict comes from the view whose hoop box is largest (closest
  look at the rim). A shot seen in only one view is kept and marked
  `views: [id]`.
- **Court plane, when calibrated.** If two or more views have a court
  calibration, ball floor positions are projected to court metres per view
  and averaged per slot, weighted by confidence. This feeds the minimap. It
  is optional and is skipped without calibration.
- **Player stats** stay per view and are reported from the primary view.
  Identity merging across views is out of scope.

Output `out/<session>/fused_events.json`, `fused_ball.json`, and
`coverage.json`. The dashboard gains a coverage row and a view selector
that plays the best view for each shot.

### B8. Training tie-in

After fusion, `ml/spark/pseudolabel.py --session <id>` runs A6 on every
view, with the cross-view rule ordering the review queue. Sessions are the
intended way rig-domain data accumulates.

---

## 3. Milestones and order

1. **A-smoke.** Spark reachable, container pinned, one-epoch smoke run.
2. **A-round1.** Public sets plus bootstrap, s and m at 1280, ball report on
   the dataset val. Pick the round-one model.
3. **B-record.** Session create, join, synced record, sidecar. Verified with
   two phones and a clap: sidecar offsets within 50 ms of the audio peak.
4. **B-ingest.** Upload, storage, status. Verified with the two clips above.
5. **B-align-fuse.** Align, per-view track, fuse, coverage. Verified on the
   two-phone clip: coverage above the primary view's single-view figure.
6. **A-round2.** Pseudo-label the sessions, review, retrain. Rig-clip gap
   under 2 s, then the live court test.
7. **A-export.** Core ML export of the chosen model. End of this spec.

A-round1 and B-record can run in parallel; they touch different code.

## 4. Testing

- `ml/spark`: unit tests for the class remap, the source-level split (no
  frame of a source in both splits), the ball report on a synthetic labeled
  set with known recall, and the gap metric.
- `vision/multiview`: unit tests for alignment on two synthetic audio tracks
  with a known offset, the hysteresis in best-view selection, and event
  deduplication. One fixture session with two short clips runs end to end.
- `ingest`: request tests for upload, sidecar, manifest, status, and token
  rejection.
- iOS: the clock-offset estimator and sidecar encoding are pure and get unit
  tests in `FollowCamTests`, like the pan control law today. Multipeer and
  camera are checked by the two-phone clap test in milestone 3.

## 5. Decisions taken and assumptions

- Training data for round one is public plus the repo's bootstrap set. The
  hackathon data is not assumed. If it turns up, it is added as one more
  source.
- Fusion is temporal and event-level, with an optional court-plane layer.
  Pixel-level transfer between views is not attempted.
- The ingest service runs on the Spark behind Tailscale. If the Spark cannot
  be reached from outside the building, the fallback is a cloud bucket plus
  a sync job; the API stays the same.
- The team token model is deliberately minimal; accounts come with pricing.
- Phones are iPhone 15 Pro or newer for the rig; recording-only phones can
  be any iPhone that runs iOS 17.
