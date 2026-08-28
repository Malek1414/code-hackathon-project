# Vision pipeline — measured results (Aug 28, 2026)

Numbers as measured on the hackathon machine (MacBook, Apple M3, 16 GB, MPS).
Source video: `data/clips/game10.mp4` (BC Lions Moabit vs Weddinger Wiesel,
Landesliga Berlin, 10 min cut, 1080p50, panning sideline camera).

## LABEL — auto-labeling and detector fine-tune

### Frames
| | |
|---|---|
| Extracted | 600 frames, 1 fps, 1920x1080 (`vision/label/extract_frames.py`, ffmpeg from imageio_ffmpeg, ~20 s) |
| Labeled | 300 frames (every 2nd), split 255 train / 45 val (seeded, 85/15) |

### Auto-labeling (`vision/label/autolabel.py`)
Two detectors per frame, no human labels:
- players + referees: Grounding DINO (`IDEA-Research/grounding-dino-tiny`), prompt `basketball player. referee.`, box/text threshold 0.2
- ball + hoop: `models/ball_hoop_avishah.pt` (ultralytics), imgsz 1280, ball conf 0.45 (best ball per frame), hoop conf 0.5

Measured while choosing the prompts (4 dev frames): the combined 4-class
prompt at threshold 0.3 found 13 players, `basketball player.` alone found 21
at 0.3 and 32 at 0.2 (about 8 per frame, which matches the court). The
multi-phrase ball prompt put "balls" on orange wall fixtures next to the
backboard, so the ball moved to the dedicated model.

| Class | Boxes raw | Boxes after cleaning |
|---|---|---|
| player | 2307 | 2307 |
| ball | 157 | 85 |
| hoop | 379 | 379 |
| referee | 624 | 624 |
| **total** | **3467** | **3395** |

Cleaning (`vision/label/clean_balls.py`): the ball model scored the fire
alarm and the orange box next to the backboard at 0.48-0.69 (a real ball
scores ~0.86). Those keep the same pixel offset to the hoop in every frame,
a ball does not, so ball boxes whose offset to the hoop recurs in 4+ frames
are removed: **72 static false balls removed**, 74 of 146 balls next to a
hoop kept, 85 ball boxes in 85 frames remain (28% of labeled frames).
Referee boxes include coaches and spectators in black on the sideline
(accepted noise, least important class).

| Timing | |
|---|---|
| Labeling total | 958 s for 300 frames (MPS, GPU shared with TRACK inference part of the time) |
| Per frame | 3.2 s (2.0 s/frame measured with the GPU alone) |

Preview: `out/label_preview.jpg` (4x4 contact sheet with drawn labels).

### Fine-tune (`vision/label/train.py`)
| | |
|---|---|
| Model | YOLO11n from `yolo11n.pt` (COCO pretrained, 2.59 M parameters, 6.5 GFLOPs) |
| Config | imgsz 960, batch 4, workers 0, cache off, AdamW lr 0.00125 (auto), mosaic on, device mps |
| Timebox | 25 min; ultralytics used it to schedule the epochs |
| Epochs run | 16 (nominal 10) in 25.4 min, ~95 s per epoch |
| Best epoch | 13 (mAP50-95 0.588) |
| Weights | `models/best.pt`, 5.5 MB |
| Slot | 12:08 to 12:35, GPU exclusive (TRACK paused) |

Validation on the 45 held-out frames (503 boxes), `best.pt` at imgsz 960:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 45 | 503 | 0.757 | 0.607 | **0.723** | **0.588** |
| player | 45 | 340 | 0.954 | 0.795 | 0.957 | 0.825 |
| hoop | 33 | 54 | 0.926 | 0.944 | 0.973 | 0.784 |
| referee | 45 | 96 | 0.819 | 0.612 | 0.755 | 0.628 |
| ball | 13 | 13 | 0.328 | 0.077 | 0.207 | 0.115 |

Learning curve (val mAP50 / mAP50-95 per epoch): 0.23/0.16, 0.52/0.37,
0.60/0.43, 0.65/0.48, 0.65/0.49, 0.68/0.52, 0.69/0.54, 0.69/0.55, 0.70/0.55,
0.70/0.56, 0.71/0.58, 0.71/0.57, **0.72/0.59**, 0.72/0.59, 0.71/0.58, 0.71/0.59.

Reading: players and hoop are learned from 300 auto-labeled frames in 25
minutes. The ball is not: 85 training examples of a ~25 px object are too few
(13 val instances, recall 0.08). For the ball, `models/ball_hoop_avishah.pt`
stays the better detector until the ball labels are multiplied (label all
600 frames, or label at 5 fps around shots).

Contact sheet for the pitch: `out/results_labels.jpg` (3 val frames, left the
auto-labels the model learned from, right `best.pt` predictions on the same
frames, which the model never saw in training). Built with
`vision/label/results_sheet.py`, CPU.

### Ball fine-tune on 80 coach-labeled frames, 12 min on the laptop GPU
Sami hand-labeled the ball on 120 game10 frames (`out/qa/ball_labels.json`,
90 with ball, 30 without, radius 12 to 17.5 px). `vision/label/build_ball_dataset.py`
made `data/dataset_ball/` (classes of `ball_hoop_avishah.pt`): train 122
images (60 ball + 20 none hand-labeled, plus 42 cleaned auto ball frames from
before the first val frame), val 40 images (30 ball + 10 none), split by
frame index. `vision/label/train_ball.py`: from `ball_hoop_avishah.pt`, imgsz
1280, batch 4, lr0 0.002 AdamW, no frozen layers, 12 epochs in 12.1 min on
MPS, output `models/ball_hoop_v2.pt` (6.3 MB), 13:39 to 13:52.

Held-out val, `vision/label/eval_ball.py` (40 frames, 30 with ball, IoU 0.3
vs the hand boxes, imgsz 1280, CPU):

| Model | conf | Ball recall | False positives / 40 frames | Precision |
|---|---|---|---|---|
| avishah | 0.35 | 13/30 = 43 % | 36 | 27 % |
| **v2** | 0.35 | **16/30 = 53 %** | **8** | **67 %** |
| avishah | 0.45 | 10/30 = 33 % | 29 | 26 % |
| v2 | 0.45 | 13/30 = 43 % | 4 | 76 % |
| v2 | 0.30 | 16/30 = 53 % | 9 | 64 % |

Ultralytics AP50 on the same val: Basketball 0.153 (v2) vs 0.021 (avishah),
Hoop 0.982 vs 0.994 (IoU 0.5 on 30 px boxes is strict for both; the ratio is
the signal). TRACK's measurement on all 120 labeled frames (which include
the 80 training frames, so recall there is optimistic): recall 0.59 vs 0.46 at
conf 0.25, frames with a false ball 12 % vs 67 %; at conf 0.35 recall 0.52
vs 0.41, false 8 % vs 57 %. Reading: 80 labels from the coach and 12
minutes on a laptop cut false balls on wall objects by a factor of 4 to 5
and lifted recall by 10 points; the remaining misses are the ball in a
player's hands.

### 14:15 comparison (measured by TRACK, `best.pt` vs COCO weights)
Setup (TRACK): `data/clips/dev60.mp4`, frames 900 to 3121, stride 4, 526
frames, 40 sample frames per config in `out/compare/`.

| Config | Ball in frames | Players / frame | Track ids | Hoop in frames |
|---|---|---|---|---|
| A: COCO yolo11s persons + `ball_hoop_avishah.pt` (ball conf 0.45) | 33 % | 12.5 | 134 | 100 % |
| A at ball conf 0.35 / 0.30 | 38 % / 43 % | | | |
| B: `best.pt` alone | 0.2 % | 11.1 | 142 | 98 % |
| C: `best.pt` persons + avishah ball/hoop | 33 % | 9.3 | 101 | 100 % |

Reading (TRACK): B confirms that `best.pt` is useless as a ball detector
(AP50 0.21). C is the production config since 13:00: the fine-tuned player
class stops counting spectators and referees as players (12.5 to 9.3 per
frame) and gives 101 instead of 134 track ids on the same footage, 25 % fewer
id switches than COCO yolo11s. In the one real shot window of the clip (57.0
to 57.8 s) the ball is carried onto the rim in 7 of 10 frames in every A and
C config, and the static wall-fixture false ball never wins any more.

## Court model (COURT, 28 Aug 13:50)

Pipeline: hand-clicked landmarks on stills (`out/court_click.html`, browser,
no server) → homography per keyframe (`vision/court/from_points.py`, RANSAC in
pixels) → per-frame homographies by optical-flow camera chaining between
direct SIFT anchors (`vision/court/propagate.py`) → minimap, overlay lines,
projection helper (`vision/court/project.py`, used by TRACK, STATS, FRONTEND).

Footage facts that shaped it: the video is an edited production, not one
continuous pan. dev60 has 7 cuts/dissolves, game10 48 (a 12-cut visual sample
was 12 real cuts), detected as frame vs. frame-one-second-earlier difference
after aligning by the tracked camera motion (threshold 25 on 0..255 grey).
Camera chaining never crosses a cut. dev60 is game10 from frame 3000 on
(pixel-identical), so keyframes are shared with a +3000 offset. The game is
played cross-court, the game basket is the ceiling-hung hoop on the far long
wall; the far half of the court is compressed to about 50 px of image height,
which bounds depth accuracy near the basket.

Keyframes (Sami's clicks, 6 points each; halfway-line clicks were wrong in
two frames and dropped):

| Clip / frame | Reprojection error | Inliers |
|---|---|---|
| dev60 1000 (= game10 4000) | 1.0 px / 0.07 m | 6/6 |
| dev60 1500 (= game10 4500) | 1.6 px / 0.03 m | 6/6 |
| game10 8500 | 2.1 px / 0.08 m | 6/6 |
| game10 12000 | 11.6 px / 0.09 m | 4/6 |
| game10 26000 | 4.3 px / 0.08 m | 5/6 |

Propagation (game10, 30001 frames): 203 SIFT auto anchors (one every 100
frames per segment, 80+ inliers and 0.6+ inlier ratio required, matched on
the static hall background), 27821 of 30001 frames calibrated (92.7 %), 9
segments uncalibrated by design (close-ups and scoreboard, NaN for consumers).
Chain drift measured between consecutive anchors on in-image court points:
mostly 10 to 60 px; 8 stretches where anchors failed for 100 to 500 frames
(fast action, zoom) exceed 150 px and are published as `uncertain_frames`
(2373 frames = 47 s of 600; `Calibration.is_uncertain(frame)`), where FRONTEND
marks shots as uncertain and distances are not summed. dev60: 21 anchors,
2148 of 3001 frames calibrated, no uncertain stretch.

Accuracy statement: width across the court is good (a few px at the
keyframes, tens of px between anchors); depth along the court near the far
basket is about ±0.5 m at the keyframes because 5.8 m of paint span ~30 px of
image, worse mid-stretch. Good enough for minimap and shot chart, not for
metre-precise distances. Stride > 1 in the camera chain is unusable (stride 2
diverges by thousands of px on a real pan), measured, so chains run at stride
1 and half resolution (~25 fps CPU, cached per clip).

## Rig tracker (`software/ball_tracker_yolo.py`, RIG)

YOLO ball detector for the physical rig, same CLI and serial protocol as
Malek's HSV `ball_tracker.py` (`A<angle>\n` at 115200, 40 to 140 degrees,
KP 0.06, deadband 25 px, EMA 0.35). Ball via `models/ball_hoop_avishah.pt`
class 0 at imgsz 640, conf 0.35; once the ball is known the model runs on a
320 px crop around it at native resolution (4x cheaper, small balls keep
their pixels), full frame at 640 when lost; HSV fallback (same mask plus a
roundness check) after 15 lost frames. Default device cpu so the GPU stays
free for the live demo.

Measured 14:20 on CPU, 960x540, next to TRACK's game10 run (machine loaded):

| Source | fps | Ball by model | HSV fallback fired |
|---|---|---|---|
| `data/clips/dev60.mp4` (wide gym shot, ball 8 to 12 px at 640) | 6.9 | 24/208 frames = 12 % | 9 frames = 4 % |
| `--cam 0` (laptop camera, no ball in view) | 5.9 | 0/179 = 0 % | 99 frames = 55 % |

Reading: on the CPU under load the tracker runs at 6 to 7 fps, below the 20
fps target; the model's ROI path is what makes it usable at all. The HSV
fallback fires on skin tones when no ball is in view (55 % on the laptop
camera), so on stage the ball must be the only orange object near the lens or
the fallback should be disabled. The dev60 12 percent is the wide-shot case;
the stage case (ball 40 px or more at 5 m from the phone) was not measured.
MPS was not tested because a `vision/track` job held the GPU during the
measurement slot. Three frames with the detection box: `out/rig/`.
