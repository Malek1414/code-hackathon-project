# Role PITCH (assigned 12:05 by ORCH, session samimagdouli-61)

Read `docs/ORCHESTRATION.md`, `IDEA.md`, `PLAN.md` and `pitch/pitch.md` first. You own
`pitch/analytics/` only (Malek owns `pitch/pitch.md`, do not edit it, propose changes
to ORCH).

Deliverables by 14:00:

1. `pitch/analytics/slides.md`: the analytics half of the 5-slide deck in the pitch
   structure Malek defined (demo first, problem, solution, platform, why us): one
   slide's worth of content for "the same camera keeps the stats": what we built today
   in numbers (auto-labeled N frames with Grounding DINO, YOLO11 fine-tuned on
   Landesliga Berlin footage, ByteTrack, jersey-number re-identification, homography
   to a 2D court, per-player FG stats, live score overlay with RTMP push), with
   placeholders `{N_FRAMES}`, `{MAP_BALL}`, `{SHOTS_FOUND}` that ORCH fills from the
   real outputs at 14:30.
2. `pitch/analytics/demo_video_plan.md`: shot list for the 60 to 90 s video-pitch
   segment about analytics (overlay video, minimap, dashboard screen recording, live
   mode with score bar), with exact source files under `out/` and ffmpeg commands
   (binary via imageio_ffmpeg in `.venv`) to assemble a side-by-side overlay + minimap
   clip at 1920x1080 once `out/overlay.mp4` and `out/minimap.mp4` exist (first
   versions exist now; test the command, write to `out/pitch/`).
3. `pitch/analytics/stage_checklist.md`: 10-line run sheet for the live demo on stage
   (phone on the rig, `live.py` command, hotkeys 1/2/3/4/z, what to say when the score
   flashes), and the anticipated Q&A for the analytics side (privacy with minors,
   accuracy, what is auto vs human-confirmed).

Style: no emojis, no dashes as bullets, plain English, numbers over adjectives.
Report to ORCH via `SendMessage(to: "samimagdouli-61", ...)`, first line "PITCH ...".
