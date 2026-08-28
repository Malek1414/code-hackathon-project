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

### 14:15 comparison (TRACK, `best.pt` vs COCO weights)
Pending, numbers from TRACK go here.
