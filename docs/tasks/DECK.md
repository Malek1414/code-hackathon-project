# Role DECK (assigned 12:10 by ORCH, session samimagdouli-61)

Read `docs/ORCHESTRATION.md`, `IDEA.md`, `pitch/pitch.md` and, when it exists,
`pitch/analytics/slides.md` (written by PITCH). You own `pitch/deck/` only.

1. `pitch/deck/index.html`: a self-contained 5-slide HTML deck (arrow keys, no
   external requests, no npm, inline CSS/JS) following Malek's structure in
   `pitch/pitch.md`: demo (a title card only, the demo is live), problem, solution,
   platform, why us. Dark, large type, one idea per slide, no emojis, no decorative
   icons, no dashes as bullets. Use `viz/followcam_assembled_poster.png` on the
   solution slide and leave a 16:9 box on the platform slide for the analytics
   demo video (`out/pitch/...` from PITCH) via relative path.
2. Placeholders `{N_FRAMES}`, `{MAP_BALL}`, `{SHOTS_FOUND}` stay as text until ORCH
   fills them at 14:30.
3. Open it in the browser (`open pitch/deck/index.html`), check every slide with a
   screenshot, fix overflow.
4. Report to ORCH via `SendMessage(to: "samimagdouli-61", ...)`, first line "DECK ...".
