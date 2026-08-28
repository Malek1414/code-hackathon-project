"""Delete raw clips and extracted frames older than N hours from data/ (dry run by default).

Usage:
    .venv/bin/python vision/privacy/retention.py [--hours 24] [--root data] [--apply]

Without --apply nothing is deleted; every candidate is printed with its age
and size. With --apply the same list is deleted. Scope is deliberately
narrow: data/clips/*.mp4 (raw footage) and data/frames/*.jpg (extracted
frames). The labeled dataset (data/dataset/) and the weights (models/) are
not raw footage and are left alone. Age = file modification time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PATTERNS = ("clips/*.mp4", "clips/*.mp4.part", "clips/*.mov", "frames/*.jpg")


def candidates(root: Path, hours: float, now: float | None = None) -> list[tuple[Path, float, int]]:
    now = time.time() if now is None else now
    out = []
    for pattern in PATTERNS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            age_h = (now - path.stat().st_mtime) / 3600
            if age_h >= hours:
                out.append((path, age_h, path.stat().st_size))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hours", type=float, default=24.0, help="delete files older than this many hours")
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    found = candidates(args.root, args.hours)
    total = sum(size for _, _, size in found)
    mode = "DELETING" if args.apply else "dry run"
    print(f"{mode}: {len(found)} files older than {args.hours:g} h under {args.root}/ ({total / 1e6:.1f} MB)")
    for path, age_h, size in found:
        print(f"  {age_h:7.1f} h  {size / 1e6:8.1f} MB  {path}")
        if args.apply:
            path.unlink()
    if not args.apply and found:
        print("nothing deleted; rerun with --apply to delete these files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
