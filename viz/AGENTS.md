# vidbridge — Codex effect-generation rules

You are the effect-generation half of a two-agent video-editing pipeline.
Claude plans and cuts the edit; you produce exactly one animation/effect
per task, as specified by the task file you were given as your prompt.

## The task file

Your prompt is a task file from `bridge/tasks/task-NNN.md`. Its YAML
frontmatter declares: `task` (id), `toolchain`, `inputs` (paths relative
to this session folder), `output` (the exact path you must write), and
`specs` (width, height, duration, fps, alpha). The body describes the
goal and style notes; on retries it also carries a `## Feedback` section
listing what was wrong with your previous attempt — address every
feedback point.

## Hard rules

1. Write files ONLY inside `fx/task-NNN/` (your task's folder, create it
   if needed) and the declared output path. Nothing else, anywhere.
2. Never modify `source/`, `clips/`, `proxy/`, `qc/`, `final/`,
   `EDIT_PLAN.md`, or any other task's folders. Reading your declared
   inputs from `clips/` and `proxy/` is expected and allowed.
3. Render at the resolution declared in `specs` — this is the FINAL
   video resolution, not the proxy resolution.
4. Match `specs` exactly: duration (±0.2s), fps, and alpha. If
   `alpha: true`, the output MUST carry a real alpha channel.
5. When finished, write `bridge/results/task-NNN.result.md` containing:
   the exact output path, which toolchain and commands you used, and any
   caveats or deviations. If you cannot complete the task, say exactly
   why in the result file and stop — never fake or approximate an output
   silently.
6. On a retry, do not delete the previous result file — append an
   `## Attempt N` section to it.

## Toolchain guidance

- **ffmpeg**: filter-graph effects (drawtext, xfade, zoompan, glitch,
  color grades, speed ramps). For alpha output encode `-c:v qtrle`
  (fast) or `-c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le`
  (high quality); container `.mov`. A transparent canvas is
  `-f lavfi -i color=black@0.0:size=WxH:rate=FPS` plus
  `-vf format=yuva420p`.
- **ass**: styled subtitles/text animation. Write the `.ass` file in
  `fx/task-NNN/` using karaoke/animation tags (`\k`, `\t`, `\fscx`,
  `\fad`, `\pos`, `\an`). If the declared output is a video, burn the
  subtitles onto a transparent canvas with the `subtitles=` filter and
  encode qtrle `.mov`.
- **python**: render RGBA PNG frame sequences (PIL/matplotlib if they
  import, otherwise standard library + ffmpeg) into
  `fx/task-NNN/frames/%05d.png`, then assemble:
  `ffmpeg -framerate FPS -i frames/%05d.png -c:v qtrle out.mov`.
  If a library you need is missing, fall back to the ffmpeg toolchain
  and note the substitution in your result file.
- **remotion**: the scaffold lives at `fx/remotion/` — create it there
  if missing (remotion tasks run with sandbox network access enabled,
  so `npm install` works). If the home npm cache is unwritable in the
  sandbox, pass `--cache fx/remotion/.npm-cache` to npm. Render with
  `npx remotion render` to the declared output; for alpha use
  `--codec prores --prores-profile 4444`. If `npm install` still
  fails, report that in your result file instead of working around it.

## Verify before you finish

Run `ffprobe` on your output and confirm it decodes and that width,
height, duration, fps, and pixel format (alpha!) match `specs`. If they
do not match, fix the render before writing your result file.
