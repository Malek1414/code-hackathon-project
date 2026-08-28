"""Draw a 4x4 contact sheet from YOLO label files that already exist.

Usage:
    .venv/bin/python vision/label/preview.py [--dataset data/dataset] [--out out/label_preview.jpg]
                                            [--first N | --spread]

Reads labels/{train,val}/*.txt and the matching images, so it works on a
partial dataset while autolabel.py is still running (the 20-frame sanity
check before burning 30 minutes on bad labels).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autolabel import CLASSES, PREVIEW, DEFAULT_DATASET, contact_sheet, draw_boxes  # noqa: E402


def read_labels(txt: Path, w: int, h: int):
    dets = []
    for line in txt.read_text().splitlines():
        if not line.strip():
            continue
        cid, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        dets.append((int(cid), 1.0, [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))
    return dets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=PREVIEW)
    ap.add_argument("--first", type=int, default=0, help="use the first N labeled frames (default: spread over all)")
    args = ap.parse_args()

    labels = sorted((args.dataset / "labels").glob("*/*.txt"))
    if not labels:
        print("no labels yet")
        return 1
    if args.first:
        labels = labels[: args.first]
    step = max(1, len(labels) // 16)
    picked = labels[::step][:16]
    tiles = []
    counts = {c: 0 for c in CLASSES}
    for txt in picked:
        img_path = args.dataset / "images" / txt.parent.name / f"{txt.stem}.jpg"
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        dets = read_labels(txt, w, h)
        for d in dets:
            counts[CLASSES[d[0]]] += 1
        tiles.append(draw_boxes(image, dets, w, h))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(tiles).save(args.out, quality=88)
    print(f"{len(labels)} labeled frames, {len(tiles)} shown, counts on sheet={counts} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
