# Video pitch script, 60 to 90 s (structure from pitch/pitch.md)

Files: `out/pitch/upload/01_prototype_10s.mp4`, `02_ingame_20s.mp4`, `03_combined_30s.mp4`,
`out/pitch/court_lines_10s.mp4`, `out/pitch/analytics_segment.mp4`, phone clips per the shoot list
in `demo_video_plan.md`. Numbers are the human-checked ones from `docs/RESULTS.md`.
Speaking pace: 2.5 words per second, so a 15 s block holds about 35 words.

## 0 to 10 s, hook: the rig follows the ball, no words

| Time | Visual | Spoken |
|---|---|---|
| 0 to 6 | Phone clip 1 from the shoot list: wide, the tripod head turning with the ball walking past; servo sound up | nothing |
| 6 to 10 | Phone clip 2: close on the fork arm pushing the pan handle; fallback `01_prototype_10s.mp4` if no real clip | nothing |

Speaker notes
1. Let the servo be heard, no music under the first 6 s.
2. The cut to the close-up lands on a direction change of the head.
3. Title card only at second 9: "FollowCam" in one word, white on black.

## 10 to 25 s, one-liner and problem

| Time | Visual | Spoken |
|---|---|---|
| 10 to 16 | Phone clip 4: the two printed parts on the table next to the servo and the Arduino | "A 3D-printed hand that grips the pan handle of any tripod and follows the ball. Twenty euros of parts." |
| 16 to 25 | Wide gym shot from game10 (`out/overlay.mp4` 40 to 49 s without boxes, or the raw frame) | "Millions of amateur and youth games are never filmed and never analysed, because an auto-tracking camera costs two thousand euros and a cameraman costs more." |

Speaker notes
1. The number is 20 euros against 2000; say both.
2. The problem is the games with the least support: kids, volunteers, Landesliga.
3. Do not explain the hardware yet, that is the next block.

## 25 to 50 s, how it works: phone, servo, our tracking

| Time | Visual | Spoken |
|---|---|---|
| 25 to 33 | Phone clip 3: over the shoulder, the phone screen with the tracking box, then the servo moving | "The phone sees the ball. The laptop turns that into an angle. A hobby servo pushes the handle. Horizontal only, that is ninety percent of court sports." |
| 33 to 42 | `02_ingame_20s.mp4` 0 to 9 s: overlay with boxes next to the 2D court, the made basket by #44 | "The same video goes through our tracking: every player, the ball, the hoop, projected onto a 2D court." |
| 42 to 50 | `out/pitch/court_lines_10s.mp4` 0 to 8 s: the camera pans, the drawn court stays on the floor | "When the camera pans, the court model follows." |

Speaker notes
1. Three nouns in order: phone, servo, tracking. Nothing else.
2. "Horizontal only" pre-empts the tilt question.
3. The court lines shot answers "does it work with a moving camera" before anyone asks.

## 50 to 75 s, platform: the same camera keeps the stats

| Time | Visual | Spoken |
|---|---|---|
| 50 to 58 | `02_ingame_20s.mp4` 9.5 to 15.5 s: dashboard, score timeline, then the shot chart | "The camera that follows the ball is the camera that keeps the stats. Ten minutes of a Berlin Landesliga game, no human drew a box: twenty-four shot attempts found, ten made." |
| 58 to 66 | `02_ingame_20s.mp4` 15.5 to 20 s: live mode with the score bar, a key press, the "+2" flash | "Checked by the coach: ninety-six percent of the called attempts were real, made or miss right three times out of four. Live, the score keeps itself, and one key corrects it." |
| 66 to 75 | Three-line card over the dashboard: broadcast overlays, team analytics, scorekeeping for volunteer leagues | "One position-data layer, three products: broadcast overlays that make games easier to follow, team analytics, and the scoreboard for the volunteer at the table." |

Speaker notes
1. Say the checked numbers, not the model numbers: 24 attempts, 10 made, 96 percent, three out of four.
2. "No human drew a box" is the line that explains why this scales.
3. Auto with veto: the human presses one key, the system never overrules a person.

## 75 to 90 s, team and "built in 7 hours at CODE"

| Time | Visual | Spoken |
|---|---|---|
| 75 to 83 | Phone clip 1 again, the rig following the ball, both founders in frame | "Designed, printed, wired and demoed in one day at CODE. Malek built the rig, Sami built the analytics." |
| 83 to 90 | End card: FollowCam, the two names, "built in 7 hours at CODE Berlin", the repo name | "Next: a real game this semester, with the rig on the sideline and the stats on the coach's phone." |

Speaker notes
1. Names first, then the time: seven hours.
2. One concrete next step, one semester, one game.
3. End on the rig moving, not on a slide.

## Timing check
Hook 10, problem 15, how 25, platform 25, team 15: 90 s. To land at 60 s cut the problem block to 8 s (drop the second sentence), the platform block to 15 s (drop the three-products card) and the team block to 8 s.
