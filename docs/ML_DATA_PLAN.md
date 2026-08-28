# FollowCam — ML data & training plan

Target capabilities, in build order: **player/ball tracking → points → rebounds
→ assists**. Events are read off tracks; nothing downstream works until
tracking is solid on OUR camera angle (single fixed phone, scorer's table).

## Data tiers (train on all three)

| Tier | What | Gets us | Labels |
|------|------|---------|--------|
| 1. NBA | Broadcast games + play-by-play from stats.nba.com (`ml/fetch_playbyplay.py`) | Label density: thousands of timestamped points/rebounds/assists per season, free | Auto: align video clock ↔ play-by-play (weak supervision) |
| 2. League tiers below | Regional/semi-pro/youth single-camera streams (club YouTube, FIBA youth) | Domain match: fixed sideline camera = our product's view | Scoreboard OCR + selective hand labels |
| 3. Our recordings | Phone-on-rig footage from the court tests | Exact domain: our height, sweep, rolling shutter, servo motion | Hand labels; primary eval set |

Rule of thumb: **train the tracker on tiers 2+3, train the event heads on
tier 1, fine-tune everything on tier 3.** NBA footage teaches events, not
viewpoint; amateur footage teaches viewpoint, not events.

Licensing note: broadcast footage is copyrighted — fine for local research/
prototyping, not redistributable. The repo stores only derived artifacts
(tracks, labels, timestamps), never the source video.

## Stack (proven repos — use their mechanisms, don't reinvent)

| Repo | Stars | What we take |
|------|-------|--------------|
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 61k | YOLO detection + built-in ByteTrack/BoT-SORT tracking (`model.track()`) |
| [roboflow/supervision](https://github.com/roboflow/supervision) | 50k | Annotation, zones, detection post-processing |
| [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot) | 8.3k | Pluggable SOTA trackers if ByteTrack falls short (OC-SORT, StrongSORT) |
| [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack) | 6.7k | The tracking-by-association paper implementation (reference) |
| [roboflow/sports](https://github.com/roboflow/sports) | 5.3k | Sports-specific: court keypoint homography, team clustering, ball path |
| [roboflow/trackers](https://github.com/roboflow/trackers) | 3.7k | Clean Apache-2.0 tracker re-implementations |
| [swar/nba_api](https://github.com/swar/nba_api) | 3.8k | Play-by-play + box scores for tier-1 labels |
| [chonyy/AI-basketball-analysis](https://github.com/chonyy/AI-basketball-analysis) | 1.3k | Shot detection via pose + ball trajectory (points head reference) |

Basketball detection weights to start from: Roboflow Universe basketball
datasets (ball/player/hoop classes) → fine-tune YOLO11n for speed.

## Event heads (dependency chain — build in this order)

1. **Points**: ball trajectory enters hoop zone downward + scoreboard OCR
   agreement. Easiest, highest-signal. Ship first.
2. **Rebounds**: missed-shot event followed by possession change within ~3s.
   Needs reliable player tracks + ball-possession assignment (nearest-track
   heuristic first, learned assignment later).
3. **Assists**: completed pass → possession ≤1 dribble/2s → made shot.
   Hardest; needs possession attribution to be trustworthy. Ship last.

## Pipeline in this repo

- `ml/fetch_playbyplay.py` — pull tier-1 event labels to CSV (nba_api).
- `ml/analyze_video.py` — YOLO11 + ByteTrack over any mp4/mov → annotated
  video + `tracks.jsonl` (per-frame boxes/ids). Run it on every court
  recording the same day it's shot.
- `ml/.venv/` — local env (gitignored); `ml/requirements.txt` pins the stack.

## Today (hackathon) vs after

- **Today**: no training. HSV tracker drives the demo. Court recordings from
  the pan test are dataset seed — shoot per the checklist in FINDINGS.md §10.
  Run `analyze_video.py` on them; the annotated output doubles as a pitch
  visual ("position-data layer" slide, live proof).
- **Week 1 after**: fetch one NBA season of play-by-play; fine-tune YOLO11n
  on a Roboflow basketball dataset; get tracking clean on tier-3 clips.
- **Week 2+**: points head with scoreboard-OCR weak labels; then rebounds;
  then assists. Evaluate only on tier 3.
