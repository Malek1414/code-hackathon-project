# Analytics half of the deck: "the same camera keeps the stats"

Slide 4 (Platform) in Malek's structure: demo, problem, solution, platform, why us.
One slide of content. Numbers are from today's runs on Landesliga Berlin footage
(BC Lions Moabit vs Weddinger Wiesel, 1080p50, panning sideline camera) and are
documented in `docs/RESULTS.md`. Values in curly braces were filled by PITCH from
`docs/RESULTS.md` and `out/`; ORCH replaces the shot count with the game10 number
at 14:30.

## Headline

The camera that follows the ball is the camera that keeps the stats.

## What we built today (on the slide, 6 lines maximum)

1. Auto-labeled 300 frames of a real Landesliga game with Grounding DINO and a ball
   detector, 3395 boxes, no human drew a box. ({N_FRAMES} = 300; 600 frames
   extracted, every 2nd labeled)
2. Fine-tuned YOLO11n on those labels in 25 minutes on a laptop: players 0.96,
   hoop 0.97, referee 0.76 mAP50 on held-out frames. Ball 0.21 ({MAP_BALL} = 0.21):
   85 ball examples are not enough yet, the ball keeps its dedicated detector.
   Payoff in tracking (measured by TRACK on 526 frames): 25 % fewer id switches
   than the COCO model (101 instead of 134 track ids), spectators and referees
   no longer counted as players (9.3 instead of 12.5 per frame).
   Second payoff: false balls on wall objects from 67 to 12 percent of frames
   after 12 minutes of training on the coach's own labels (80 frames the coach
   labeled on the laptop; held-out check: false positives 36 to 8 in 40
   frames, recall 43 to 53 percent).
3. Tracked every player with ByteTrack: 10.6 players per frame, 258 track ids over
   60 s, team by jersey color, jersey numbers read and merged into 
   player identities (`out/identities.json`).
4. Projected feet onto a 2D court (homography, one click per court landmark) and
   rendered a minimap next to the video.
5. Shot events from ball and hoop geometry, per-player FGA / FGM / FG% table.
   {SHOTS_FOUND} = 1 shot attempt found in the 60 s dev clip (game10 number at 14:30).
6. Live mode: same models on the phone stream at about 10 fps, running score bar,
   auto +2 with human veto by hotkey, MJPEG and RTMP push out.

## Speaker notes (30 to 45 s)

Everything on this slide ran on one MacBook today, on footage of a Berlin
Landesliga game nobody films professionally. The point is not the mAP numbers,
the point is the loop: film with the rig, label automatically, fine-tune, track,
and the stats fall out. The ball number is honest, 0.21, because 85 examples are
too few; that is a data problem, and the rig is the data machine. Every game the
rig films adds labeled frames.

Three surfaces on the same data: broadcast overlays for viewers, tactical
analytics for coaches, and the scoreboard for the volunteer at the table.

## Numbers ready for the slide (copy as needed)

| Item | Value | Source |
|---|---|---|
| Frames extracted | 600 (1 fps, 1920x1080) | `docs/RESULTS.md` |
| Frames auto-labeled | 300 | `data/dataset/label_summary.json` |
| Boxes | 3395 (player 2307, ball 85, hoop 379, referee 624) | `docs/RESULTS.md` |
| False balls removed by geometry | 72 | `vision/label/clean_balls.py` |
| Fine-tune | YOLO11n, 16 epochs, 25.4 min, 5.5 MB weights | `runs/label_yolo11n/results.csv` |
| Val mAP50 player / hoop / referee / ball | 0.957 / 0.973 / 0.755 / 0.207 | `docs/RESULTS.md` |
| Players per frame tracked | 10.6 (COCO run), 9.3 with best.pt persons | `out/track_summary.json`, `out/compare/` |
| Track ids on 526 frames, COCO vs best.pt persons | 134 vs 101 (25 % fewer switches) | `out/compare/` (TRACK) |
| Ball fine-tune on 80 coach-labeled frames, 12 min | false balls 67 % to 12 % of frames (TRACK, 120 frames); held-out: FP 36 to 8, recall 43 % to 53 % | `docs/RESULTS.md` |
| Ball seen in frames | 41% | `out/track_summary.json` |
| Tracking speed | 0.56 s per frame at 1080p on MPS (yolo11s + ball model) | `out/track_summary.json` |
| Live detection rate | about 10 fps | `vision/live/live.py` |
| Shots found (dev 60 s) | 1 attempt, 0 made | `out/events.json` |

## Proposed change to `pitch/pitch.md` (for ORCH to pass to Malek)

Platform bullet (a) says "names and stats above players". Today's overlay shows
track ids, jersey numbers where read, team color and the ball; "stats above
players" is roadmap. Suggest: "names, numbers and live score on the video".
