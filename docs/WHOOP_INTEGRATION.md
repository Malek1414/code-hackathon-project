# WHOOP heart-rate integration — live HR on the recording, correlated with mistakes

Question this answers for the player: *when my heart rate went out of bounds,
did my mistake rate go up?*

## How the data flows

```
WHOOP strap (HR Broadcast mode, BLE 0x180D)
        │ live bpm, ~1 Hz
        ▼
FollowCam app (HeartRateMonitor.swift)
  - shows live bpm in the HUD
  - during recording: logs (unix_ts, bpm) -> Documents/hr_<stamp>.csv
        ▼ after the session
ml/analyze_video.py  -> tracks.jsonl        (what happened on court)
ml/events.py         -> events.json         (shot attempts; misses once the
                                             hoop fine-tune lands)
ml/correlate_hr.py   -> correlation report  (mistake rate per HR zone,
                                             in-bounds vs out-of-bounds)
```

Two key design points:

1. **Live HR needs no WHOOP API.** WHOOP's "Heart Rate Broadcast" advertises
   the standard Bluetooth Heart Rate service, so the app reads it directly
   (same code works with Polar/Garmin straps). Enable: WHOOP app → device
   settings → Heart Rate Broadcast.
2. **The WHOOP REST API is the post-game enrichment**, not the live path:
   OAuth2 at developer.whoop.com, pull the workout's strain/recovery/HR-zone
   summary and attach it to the session report. Needs an app registration —
   Sammy can do this in parallel (scopes: `read:workout`, `read:recovery`).

## Alignment

The HR log starts at the same instant as the recording, so
`video_t = unix_ts - first_row_ts`. Events are frames at a known fps.
`correlate_hr.py` joins the two timelines and buckets by HR zone
(percent of max HR, default max 190, `--max-hr` to set).

## What "mistake" means, in stages

- **v0 (now)**: shot attempts from events.json; report attempts + rate per
  zone. Optionally pass `--mistakes labels.csv` (frame ranges Sammy tags by
  hand) for true mistakes: turnovers, bad passes, misses.
- **v1**: missed shots, automatic — needs the hoop-class fine-tune
  (ML_DATA_PLAN §event heads).
- **v2**: turnovers from possession-change patterns in events.json.

## Output

`correlation.json` + printed table: minutes and events per HR zone, mistake
rate in-bounds vs above `--hr-limit`, and the delta. That delta is the
product claim ("your turnovers double above 165 bpm") — the pitch's
B2B analytics slide, grounded in the player's own footage.
