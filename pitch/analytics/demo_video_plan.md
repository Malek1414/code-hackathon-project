# Analytics segment of the video pitch (60 to 90 s inside the 90 s cut, or its own 60 s block)

Sources under `out/`, all produced today. First versions exist now for the 60 s
dev clip (`data/clips/dev60.mp4`); ORCH swaps in the game10 outputs after 14:15
when they exist under the same names.

| Source | What it is | State at 13:05 |
|---|---|---|
| `out/overlay.mp4` | 1920x1080, 25 fps, 60 s, boxes with track ids, team colors, ball trail | exists |
| `out/minimap.mp4` | 1200x680, 25 fps, 20 s, 2D court with team dots | exists, says "uncalibrated" until COURT's calibration lands |
| `out/dashboard.html` | shot chart, per-player table, minimap embed | exists, screen-record it |
| `out/results_labels.jpg` | 2x3 sheet, auto-labels vs fine-tuned predictions | exists |
| `out/label_preview.jpg` | 4x4 sheet of auto-labels | exists |
| live mode | `vision/live/live.py` window with score bar | screen-record during the rehearsal |

## Shot list

| # | Seconds | Shot | Source | Voice over |
|---|---|---|---|---|
| 1 | 0 to 8 | Side by side: overlay video left, minimap right, wide court play | `out/pitch/analytics_side_by_side.mp4` | "The rig films. The software watches." |
| 2 | 8 to 18 | Auto-labels, then the fine-tuned model on frames it never saw (Ken Burns over the sheet, left column then right) | `out/results_labels.jpg` | "300 frames labeled by AI, no human drew a box. 25 minutes later the model finds every player and the hoop." |
| 3 | 18 to 30 | Overlay video full frame, a possession with ids and jersey numbers visible | `out/overlay.mp4`, pick the window with the wide court | "Every player tracked, numbers read off the jerseys." |
| 4 | 30 to 42 | Dashboard screen recording: shot chart, then the per-player table | `out/dashboard.html` in a browser, 1920x1080 window | "Shots, made and missed, per player. The table a coach never had for a Landesliga game." |
| 5 | 42 to 55 | Live mode: score bar, a made shot flashes, a hotkey veto | screen recording of `vision/live/live.py` | "Live, the score keeps itself. One key to veto." |
| 6 | 55 to 60 | Back to the rig following the ball (Malek's footage) | phone footage | "One tripod. One phone. The whole game, filmed, tracked, scored." |

Text on screen, bottom left, in every analytics shot: the number that shot
proves (300 frames, 0.96 mAP players, 10.6 players per frame, 1 shot found).

## Assembly commands

ffmpeg binary from the venv, never the system one:

```
FF=$(.venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p out/pitch
```

Side by side, overlay 1280x720 left, minimap 640 wide right, on a 1920x1080
canvas, 20 s from second 40 of the sources (tested 13:00, 74 s render time on
the M3; output `out/pitch/analytics_side_by_side.mp4`, 1920x1080, 25 fps):

```
$FF -hide_banner -y -ss 40 -t 20 -i out/overlay.mp4 -ss 0 -t 20 -i out/minimap.mp4 \
  -filter_complex "[0:v]scale=1280:720:flags=lanczos[a];[1:v]scale=640:-2:flags=lanczos,pad=640:720:0:(oh-ih)/2:color=black[b];[a][b]hstack=inputs=2,pad=1920:1080:0:(oh-ih)/2:color=black,format=yuv420p[v]" \
  -map "[v]" -c:v libx264 -preset fast -crf 20 -r 25 -movflags +faststart out/pitch/analytics_side_by_side.mp4
```

Notes from the test: `-shortest` does not stop the output at the shorter input
when a filter graph is used, so trim both inputs with `-t`. Pick the `-ss` of the
overlay so that the window shows the wide court, not the bench (the dev clip has
edited close-ups; check with a frame grab before rendering):

```
$FF -hide_banner -y -ss 8 -i out/pitch/analytics_side_by_side.mp4 -frames:v 1 out/pitch/check.jpg
```

Ken Burns over the results sheet (10 s, slow zoom from the left column to the
right, 1920x1080):

```
$FF -hide_banner -y -loop 1 -i out/results_labels.jpg -t 10 \
  -vf "scale=3840:-2,zoompan=z='1.0+0.15*on/250':x='iw/2-(iw/zoom/2)+(on/250)*iw*0.12':y='ih/2-(ih/zoom/2)':d=250:s=1920x1080:fps=25,format=yuv420p" \
  -c:v libx264 -preset fast -crf 20 -movflags +faststart out/pitch/results_sheet_10s.mp4
```

Overlay window, 12 s from the wide-court segment, with a caption:

```
$FF -hide_banner -y -ss 40 -t 12 -i out/overlay.mp4 \
  -vf "drawtext=text='10.6 players per frame, 258 tracks in 60 s':fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=44:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=16:x=40:y=h-110,format=yuv420p" \
  -c:v libx264 -preset fast -crf 20 -movflags +faststart out/pitch/overlay_12s.mp4
```

Concatenate the analytics block (all parts 1920x1080, 25 fps, no audio; the
voice over is recorded on the phone and laid under in CapCut):

```
printf "file 'analytics_side_by_side.mp4'\nfile 'results_sheet_10s.mp4'\nfile 'overlay_12s.mp4'\nfile 'dashboard_12s.mp4'\nfile 'live_13s.mp4'\n" > out/pitch/concat.txt
$FF -hide_banner -y -f concat -safe 0 -i out/pitch/concat.txt -c:v libx264 -preset fast -crf 20 -r 25 -movflags +faststart out/pitch/analytics_block.mp4
```

Screen recordings (dashboard and live mode) come from macOS: Shift+Cmd+5,
record a 1920x1080 browser window, then trim with
`$FF -ss 3 -t 12 -i recording.mov -vf scale=1920:1080,format=yuv420p -r 25 -an out/pitch/dashboard_12s.mp4`.

## Fallbacks

1. No calibration by 14:15: minimap stays "uncalibrated" with the court only.
   Drop the side by side, use the overlay full frame for shot 1 and keep the
   minimap as a still from `out/minimap_preview.png` with the caption "2D court
   projection, calibrated on the next game".
2. Live mode unstable on stage footage: record it on `data/clips/dev60.mp4`
   with `--realtime --replay out/tracks.jsonl`, which renders from the saved tracks.
3. game10 outputs not ready by 14:30: the dev60 versions are already cut and
   the numbers on screen say "60 s clip".
