# Role DOCS (assigned 12:10 by ORCH, session samimagdouli-61)

Read `docs/ORCHESTRATION.md` and `README.md` first. You own `README.md` section
"Analytics pipeline" (append, do not touch Malek's sections), `docs/VISION.md` and
`docs/PR_BODY.md`.

1. `docs/VISION.md`: what the analytics side does, one paragraph per stage (labeling,
   training, tracking, jersey numbers, court model, stats, live mode, QA), each with
   the exact command, inputs and outputs, taken from the real code in `vision/`
   (read the files, do not invent flags). Numbers only where measured (grep the
   session reports in `docs/ORCHESTRATION.md` and the logs in `out/*.log`).
2. `README.md`: add a short "Analytics pipeline" section that links to `docs/VISION.md`
   and shows the one-command run (`vision/run_all.py`, being built by PIPELINE) and
   the live command (`vision/live/live.py`).
3. `docs/PR_BODY.md`: PR description for `sammy/vision` into `main`, for Malek's
   review: what was added, how to run, what is measured, what is unverified, known
   limitations (edited footage, ball recall, id switches). Keep it under 60 lines.
4. Refresh all three at 14:15 from the then-current code and outputs.
   Report to ORCH via `SendMessage(to: "samimagdouli-61", ...)`, first line "DOCS ...".
