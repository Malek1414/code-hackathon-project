# Big Ball Baller, broadcast spec for the Swift app (28.08.2026, MONITOR/RISK)

Sources: docs/ORCHESTRATION.md "Broadcast package", vision/live/live.py (--help, 14:40), app/ (Malek's
FollowCam iOS app: CameraManager, PanController, SubjectTracker, Keychain helper in WhoopSync.swift).
Owners: FRONTEND widgets and start menu, COURT heat map and summary renderers, STATS/LIVE live_state.json
and compositing in live.py, this file MONITOR/RISK. Interfaces change only through ORCH.

## 1. Architecture

```
phone camera (app, back wide, 1920x1080 30 fps)            phone camera as Continuity Camera
        |  on-device model (later: CoreML)                          |  cv2 index, --source auto
        v                                                           v
   Swift app: ball box, pan law  --ws://mac:8765 "A95"-->  live.py on the Mac: YOLO + ByteTrack, StatsEngine,
        |                         software/pan_bridge.py     ScoreBoard, PanController --serial "A95\n" 115200
        |                                                           |
        |<-- GET http://<mac>:8501/state.json every 1 s -----------+-- writes out/live_state.json every 1 s
        |    (live.py --bind 0.0.0.0, default 127.0.0.1)             |
        v                                                           |
   overlay widgets (PNG with alpha, broadcast/assets/)             +-- MJPEG http://<mac>:8501/stream
        v                                                           +-- RTMP push (ffmpeg) when FOLLOWCAM_RTMP_URL is set
   RTMP from the phone (stream key from Keychain)
```
Two producers of the same state: the Mac (live.py, today) or the phone alone (on-device model, later).
The Swift app never computes stats itself; it renders whatever state it receives. One source of truth
per game: either the Mac path or the on-device path, chosen in the start menu, never both.

## 2. State contract `live_state.json`

Written by LIVE atomically (tmp file, then rename) every 1.0 s to `out/live_state.json` and served at
`http://<mac>:8501/state.json` (`Content-Type: application/json`, `Cache-Control: no-store`) by the same
server as the MJPEG stream. The server binds `--bind` (default `127.0.0.1`; `0.0.0.0` for the phone on the
same Wi-Fi, same `--mjpeg-port`). The app polls every 1 s and treats a response older than 5 s (`t`
unchanged) as "signal lost". All fields are required unless marked nullable; numbers are JSON numbers,
colors are `#rrggbb` strings. `schema` is the first field; the app refuses any other value than 1.

| field | type | meaning |
|---|---|---|
| `schema` | int | always `1` for this contract; a breaking change bumps it |
| `brand` | string | always `"Big Ball Baller"` |
| `clip_or_source` | string | camera index, `auto`, or the video path that feeds LIVE |
| `t` | number, s | seconds since LIVE started (source time for files) |
| `period` | int, from 1 | current period as counted by the start menu period length |
| `clock` | string `mm:ss` | time left in the period |
| `teams[2]` | object | `id` 0 or 1, `name`, `color`, `score`, `fga`, `fgm`, `fg_pct` (nullable, null when `fga` is 0), `possessions` |
| `players[]` | object | `key` (A5, B12, A?7 for a track without a number), `number` (nullable int), `team` 0 or 1, `pts`, `fga`, `fgm`, `fg_pct` (nullable), `possession_s`, `distance_m` (nullable, null without calibration); sorted by `pts` descending |
| `last_event` | object, nullable | `t`, `type` one of `made`, `miss`, `manual`, `team`, `player_key` (nullable), `points` |
| `pan_deg` | int, nullable | last servo angle 40 to 140, null without `--serial` |
| `camera` | string | `ok` or `no-frame` (LIVE keeps running and reopens the device) |

Team names and colors are copied from the start menu: live.py flags `--team-a`, `--team-b`, `--color-a`,
`--color-b`, or the file form `broadcast/config.json` with the same four keys (`team_a`, `team_b`,
`color_a`, `color_b`; flags win over the file). Never derived from jersey colors. The app must not add
fields it invents.

## 3. Widgets

All widgets are PNG with alpha on a 1920x1080 canvas, exported from `broadcast/widgets/<id>.html` or
`.svg` to `broadcast/assets/<id>.png`; text that changes per game is drawn by the app on top of the
asset, so the asset holds the glass panel, the wordmark and the fixed labels only. Origin is the top
left corner of the 1080p frame. Minimum type size 28 px. One team accent color per team, nothing else.
Only one of `player_card` and `team_overview` is visible at a time (`team_overview` wins);
`end_summary` and `heat_map` hide every other widget.

| id | purpose | size, position | appears | disappears | data fields |
|---|---|---|---|---|---|
| `score_bug` | score, clock, period | 520x88 at x 700, y 24 | always | never | `teams[].name`, `color`, `score`, `clock`, `period` |
| `made_flash` | confirms a basket | same box as score_bug, drawn over it | a `last_event` with type `made` or `manual` and points > 0 | 1.5 s later | `last_event.team`, `points` |
| `player_card` | who scored | 420x160 at x 48, y 872 | 3 s after a made basket by that player; and the top scorer of each team every 3 min | after 3 s | `players[i].number`, `key`, `pts`, `fgm`, `fga`, `fg_pct` |
| `team_overview` | both teams at a glance | 900x420 at x 510, y 330 | every 5 min for 6 s, and on hotkey `t` (timeout) | after 6 s or next `t` | `teams[]` all fields, top scorer per team |
| `lower_third` | brand and game title | 1920x120 at x 0, y 960 | first 10 s of the stream and on hotkey `b` | after 10 s | `brand`, game title from the start menu |
| `end_summary` | efficiency table | full frame | hotkey `e` or end of file | on `e` again, or swipe to page 2 | per player and team: `pts`, `fga`, `fgm`, `fg_pct`, possession share (`possession_s` / sum) |
| `heat_map` | where each team played, shot chart | full frame, or right panel 640x1080 at x 1280 | page 2 of the summary | on `e` | position density per team from tracks and calibration, shots made and missed |

Timing is measured on the state clock `t`, not on wall time, so a replayed file behaves like a game.
Hotkeys on the Mac (window focused): 1/2 = +2 team A/B, 3/4 = +3, z = undo, t, b, e as above, q = quit.
On the phone the same actions are buttons on the live view; both write `type: manual` events.

## 4. Start menu flow

Fields, in this order, all required before the start button enables: team A name, team A color (default
`#2f6fdb`), team B name, team B color (default `#c8102e`), game title (free text, one line), period length
in minutes (default 10), number of periods (default 2), camera source (phone back camera with the
on-device model, or Mac path with the Mac's address and `--source auto` or an index), stream target
(off, or RTMP). The stream key is entered once and stored in the Keychain under `bbb.rtmpURL` with the
existing Keychain helper (WhoopSync.swift); on the Mac it lives in `.env` as `FOLLOWCAM_RTMP_URL`.
It is never written to `broadcast/config.json`, never shown after entry (masked as `rtmp://host/app/***`),
never logged. Everything else is saved to `broadcast/config.json` (`team_a`, `team_b`, `color_a`,
`color_b`, `title`, `period_min`, `periods`, `source`), which live.py reads as the file form of
`--team-a/--team-b/--color-a/--color-b`; the flags override the file, so both paths show the same names.
Flow: menu, validation (names not empty and different, colors not equal, key present when streaming),
camera preview with the pan link status (`rig` button turns green when ws://mac:8765 answers), start,
live view with score bug and the manual buttons, end button or end of file, summary pages, export.

## 5. End-of-game summary and heat map pages

Page 1 `end_summary`: one table, teams first (score, FGA, FGM, FG%, possessions), then players sorted by
points (key or number, team color chip, PTS, FGA, FGM, FG%, possession share). Page 2 `heat_map`: the
28 x 15 m court, one density layer per team in its color, shot markers (filled = made, hollow = miss),
and a caption with the calibration state ("uncalibrated" when no `out/court_calib_<clip>.json`).
Both pages are rendered by COURT as 1920x1080 PNGs (`out/end_summary.png`, `out/heat_map.png`) on the Mac
path; on the phone path the app draws page 1 itself from the final state and shows page 2 only when the
Mac delivered it. Export: one tap saves both pages and the final `live_state.json` to Photos and Files.

## 6. What the Swift app must not do

Guess or infer team names, club names or colors from the picture; they come from the menu only.
Store, crop, upload or match faces; the only per-person data is a track key, a jersey number and
numbers derived from positions. Log, print, persist in plain text, or send anywhere but the RTMP
endpoint the stream key or any URL that contains it. Upload video or state to a cloud service by
default; recording stays in Photos on the phone. Show free-text player names; keys and numbers only.
Write into `out/` contract files (tracks, events, stats, identities); the app reads state, it never
produces it. Keep streaming when the camera reports `no-frame` for more than 30 s; show the signal
lost card instead.
