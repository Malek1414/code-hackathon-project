# Role RISK (assigned 12:10 by ORCH, session samimagdouli-61)

Read `docs/ORCHESTRATION.md`, `PLAN.md`, `IDEA.md` and skim `vision/` first. You own
`docs/RISKS.md` only. You change no code.

Goal: adversarial review of the stage demo path. At 12:40 and again at 13:40 write
`docs/RISKS.md`: for each step of the 16:00 demo (live.py from the phone on the rig,
score overlay, dashboard on the projector, video pitch playback) list what breaks,
how likely, how we would notice, and the 30-second fallback. Check facts by running
the read-only checks yourself: does `vision/live/live.py --help` work, does the
Continuity Camera appear as a cv2 index (`cv2.VideoCapture(i)` for i in 0..3, one
frame each), is `out/dashboard.html` self-contained (grep for http), how long does
`vision/dashboard/build.py` take, what happens when the ball is not seen for 10 s.
Rank by damage. Ten items maximum. Send the top three to ORCH each time via
`SendMessage(to: "samimagdouli-61", ...)`, first line "RISK ...".
