"""Remove static "ball" labels: orange wall fixtures next to the backboard.

Usage:
    .venv/bin/python vision/label/clean_balls.py [--dataset data/dataset] [--dry-run]

The ball/hoop model scores the fire alarm and the orange box next to the
backboard at 0.48-0.69 (a real ball scores ~0.86), and a per-frame threshold
cannot separate them. What separates them is time: a fixture keeps the same
pixel offset to the hoop box in every frame, a ball does not. So: for each
ball with a hoop in the same frame, compute the offset ball-center minus
hoop-center; offsets that recur within RADIUS px in at least MIN_FRAMES frames
are fixtures and those ball lines are deleted. Balls in frames without a hoop
are left alone. Runs on the label txt files only, so it can run after (or
during) autolabel.py without touching the GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autolabel import DEFAULT_DATASET  # noqa: E402

W, H = 1920, 1080  # frames are extracted at full resolution
RADIUS = 14.0  # px; the same fixture drifts a few px with the hoop box
MIN_FRAMES = 4


def parse(txt: Path):
    rows = []
    for line in txt.read_text().splitlines():
        if line.strip():
            cid, cx, cy, bw, bh = line.split()
            rows.append((int(cid), float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H, line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted((args.dataset / "labels").glob("*/*.txt"))
    # (file, line, offset) for every ball that shares a frame with a hoop
    candidates = []
    for txt in files:
        rows = parse(txt)
        hoops = [r for r in rows if r[0] == 2]
        for r in rows:
            if r[0] != 1 or not hoops:
                continue
            hx, hy = min(hoops, key=lambda h: (h[1] - r[1]) ** 2 + (h[2] - r[2]) ** 2)[1:3]
            candidates.append((txt, r[5], (r[1] - hx, r[2] - hy)))

    static = set()
    for i, (_, _, (dx, dy)) in enumerate(candidates):
        near = [j for j, (_, _, (ex, ey)) in enumerate(candidates)
                if (dx - ex) ** 2 + (dy - ey) ** 2 <= RADIUS ** 2]
        if len(near) >= MIN_FRAMES:
            static.update(near)

    removed = 0
    for i in sorted(static):
        txt, line, off = candidates[i]
        if not args.dry_run:
            lines = [l for l in txt.read_text().splitlines() if l != line]
            txt.write_text("\n".join(lines) + ("\n" if lines else ""))
        removed += 1
    kept = len(candidates) - removed
    print(f"{len(files)} label files, {len(candidates)} balls next to a hoop, "
          f"{removed} static (removed{' [dry run]' if args.dry_run else ''}), {kept} kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
