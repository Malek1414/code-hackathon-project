# Role PIPELINE (assigned 12:10 by ORCH, session samimagdouli-61)

Read `docs/ORCHESTRATION.md` first. You own `vision/run_all.py`, `vision/smoke_test.py`
and `Makefile` targets only. Do not edit other roles' modules; if one of their CLIs
needs a flag, ask that role directly (names in ORCHESTRATION.md).

Goal: ONE command runs the whole analytics pipeline on a clip, so that at the 14:30
freeze and on stage nobody types eight commands.

1. `vision/run_all.py --clip data/clips/dev60.mp4 [--weights ...] [--calib out/court_calib_dev60.json]`
   runs, in order and each step skippable when its output is newer than its inputs:
   TRACK (`vision/track/run.py`), NUMBERS (`vision/numbers/`), COURT propagate + minimap
   (`vision/court/propagate.py`, `vision/court/minimap.py`, only if a calibration
   exists), STATS (`vision/stats/build.py`), QA sheets (`vision/qa/watch.py --once`),
   FRONTEND (`vision/dashboard/build.py`). Read each CLI's `--help` first, do not guess
   flags. Print one status line per step with elapsed seconds, stop on the first
   failure with the failing command shown.
2. `vision/smoke_test.py`: runs run_all on a 10 s cut of dev60 (make it with the
   imageio_ffmpeg binary, `-c copy`) and asserts the contract files exist and parse
   (`out/tracks.jsonl`, `out/events.json`, `out/stats.json`, `out/dashboard.html`).
   Respect the GPU schedule: run the smoke test only when `ps -axo command | grep
   .venv/bin/python` shows no other model job, otherwise wait.
3. `make demo CLIP=...` and `make smoke` targets in a Makefile at repo root.
4. Report to ORCH via `SendMessage(to: "samimagdouli-61", ...)`, first line "PIPELINE ...".
