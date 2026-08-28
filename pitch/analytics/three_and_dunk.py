"""Cut 04_three_and_dunk_10s.mp4: 5 s around the three, 5 s around the dunk, portrait on a dark 1080p canvas.

Usage:
    .venv/bin/python pitch/analytics/three_and_dunk.py --src out/whatsapp_1515/celebration.mp4 \
        --three <release time s> --dunk <dunk time s> [--out out/pitch/upload/04_three_and_dunk_10s.mp4]

Windows: three = release - 3 s to release + 2 s, dunk = dunk - 3 s to dunk + 2 s, or
explicit --three-window START DUR / --dunk-window START DUR. Used 15:35 on
whatsapp_1515: three 10.3 + 5.5 (release 12.3 s, swish 14.0 s), dunk 29.0 + 4.5
(hand at the rim 31.7 s), times read off 4 to 5 fps contact sheets. The portrait clip is scaled to 1080 px height and centred on a
0x141414 canvas, BBB wordmark (cropped from broadcast/assets/lower_third.png)
top left, caption bottom left. H.264, 25 fps, threads 2, no audio.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FONT = "/System/Library/Fonts/Helvetica.ttc"
WORDMARK = ROOT / "out" / "pitch" / "bbb_wordmark.png"


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def part(src: Path, start: float, dur: float, caption: str, out: Path) -> None:
    cap = (f"drawtext=fontfile={FONT}:fontsize=52:fontcolor=white:box=1:boxcolor=black@0.65:"
           f"boxborderw=20:x=60:y=h-140:text='{caption}'")
    fc = (f"[0:v]trim={start}:{start + dur},setpts=PTS-STARTPTS,scale=-2:1080:flags=lanczos,"
          f"pad=1920:1080:(ow-iw)/2:0:color=0x141414[v0];"
          f"[v0][1:v]overlay=60:40[v1];[v1]{cap},format=yuv420p[v]")
    cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
           "-i", str(src), "-i", str(WORDMARK), "-filter_complex", fc, "-map", "[v]",
           "-r", "25", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "2", "-an", str(out)]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=ROOT / "out" / "whatsapp_1515" / "celebration.mp4")
    ap.add_argument("--three", type=float, default=None, help="time (s) the ball leaves the hand on the three")
    ap.add_argument("--dunk", type=float, default=None, help="time (s) of the dunk")
    ap.add_argument("--before", type=float, default=3.0)
    ap.add_argument("--after", type=float, default=2.0)
    ap.add_argument("--three-window", type=float, nargs=2, metavar=("START", "DUR"),
                    help="explicit window for the three, overrides --three/--before/--after")
    ap.add_argument("--dunk-window", type=float, nargs=2, metavar=("START", "DUR"),
                    help="explicit window for the dunk")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "pitch" / "upload" / "04_three_and_dunk_10s.mp4")
    args = ap.parse_args()
    dur = args.before + args.after
    if args.three_window:
        t_start, t_dur = args.three_window
    elif args.three is not None:
        t_start, t_dur = max(0.0, args.three - args.before), dur
    else:
        ap.error("--three or --three-window required")
    if args.dunk_window:
        d_start, d_dur = args.dunk_window
    elif args.dunk is not None:
        d_start, d_dur = max(0.0, args.dunk - args.before), dur
    else:
        ap.error("--dunk or --dunk-window required")
    tmp = args.out.parent / "tmp_04"
    tmp.mkdir(parents=True, exist_ok=True)
    a, b = tmp / "three.mp4", tmp / "dunk.mp4"
    part(args.src, t_start, t_dur, "three pointer", a)
    part(args.src, d_start, d_dur, "dunk", b)
    lst = tmp / "list.txt"
    lst.write_text(f"file '{a.resolve()}'\nfile '{b.resolve()}'\n")
    subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-threads", "2", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(args.out)], check=True)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
