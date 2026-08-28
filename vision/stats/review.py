"""Hand-count helper: list detected shots as mm:ss and cut a short snippet
around each one so a human can confirm made / miss / shooter.

    .venv/bin/python -m vision.stats.review --events out/events.json --clip data/clips/game10.mp4 \
        --from 0:00 --to 2:00 --snippets out/stats_review

Snippets are `<n>_<mmss>_<made|miss>_p<id>.mp4`, 3 s before to 2 s after the
rim entry, re-encoded small (640 px wide) so they open instantly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_ts(s: str) -> float:
    parts = s.split(":")
    return sum(float(p) * 60**i for i, p in enumerate(reversed(parts)))


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def cut(clip: Path, t: float, out: Path, before: float = 3.0, after: float = 2.0) -> None:
    start = max(t - before, 0.0)
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-i", str(clip), "-t", f"{before + after:.2f}",
        "-vf", "scale=640:-2", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default="out/events.json")
    ap.add_argument("--clip", default=None, help="default: the clip named in events.json")
    ap.add_argument("--from", dest="t_from", default="0:00")
    ap.add_argument("--to", dest="t_to", default=None)
    ap.add_argument("--snippets", default=None, help="directory for the video snippets (omit: list only)")
    args = ap.parse_args(argv)

    events = json.loads(Path(args.events).read_text())
    clip = Path(args.clip or events["clip"])
    t0 = parse_ts(args.t_from)
    t1 = parse_ts(args.t_to) if args.t_to else float("inf")
    shots = [s for s in events["shots"] if t0 <= s["t"] <= t1]

    made = sum(1 for s in shots if s["made"])
    print(f"{len(shots)} shots between {mmss(t0)} and {mmss(t1) if t1 != float('inf') else 'end'}: {made} made, {len(shots) - made} missed")
    for n, s in enumerate(shots, 1):
        who = f"p{s['player_id']}" if s["player_id"] is not None else "p?"
        flag = "" if s.get("shooter_confirmed", True) else "  (shooter unconfirmed)"
        print(f"  {n:2d}  {mmss(s['t'])}  {'MADE' if s['made'] else 'miss'}  {who} team {s['team']}{flag}")

    if args.snippets:
        out = Path(args.snippets)
        out.mkdir(parents=True, exist_ok=True)
        for n, s in enumerate(shots, 1):
            who = f"p{s['player_id']}" if s["player_id"] is not None else "px"
            name = f"{n:02d}_{mmss(s['t']).replace(':', 'm').replace('.', 's')}_{'made' if s['made'] else 'miss'}_{who}.mp4"
            cut(clip, s["t"], out / name)
            print(f"  -> {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
