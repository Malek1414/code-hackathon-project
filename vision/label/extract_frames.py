"""Extract frames at a fixed rate from a clip for auto-labeling.

Usage:
    .venv/bin/python vision/label/extract_frames.py [clip] [--fps 1] [--out data/frames]

Defaults to data/clips/game10.mp4 -> data/frames/f_%05d.jpg at full resolution
(1920x1080, no scaling: a downscaled ball is not a ball anymore, see
courtside/engine/detect/players.py). Uses the ffmpeg binary bundled with
imageio_ffmpeg, never a system ffmpeg.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIP = ROOT / "data" / "clips" / "game10.mp4"
DEFAULT_OUT = ROOT / "data" / "frames"


def extract(clip: Path, out_dir: Path, fps: float = 1.0, quality: int = 2,
            prefix: str = "f_") -> int:
    """Write JPEGs to out_dir and return how many were written."""
    if not clip.exists():
        raise FileNotFoundError(clip)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{prefix}*.jpg"):
        old.unlink()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(clip),
        "-vf", f"fps={fps}",
        "-q:v", str(quality),  # 2 = near-lossless JPEG, keeps the ball crisp
        str(out_dir / f"{prefix}%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob(f"{prefix}*.jpg")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip", nargs="?", type=Path, default=DEFAULT_CLIP)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--prefix", default="f_")
    args = ap.parse_args()
    n = extract(args.clip, args.out, args.fps, prefix=args.prefix)
    print(f"{n} frames -> {args.out} (from {args.clip.name} at {args.fps} fps)")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
