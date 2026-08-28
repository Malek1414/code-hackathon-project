"""2x3 contact sheet for the pitch: auto-labels (left) vs best.pt predictions (right).

Usage:
    .venv/bin/python vision/label/results_sheet.py [--weights models/best.pt] [--out out/results_labels.jpg]
                                                  [--frames f_00013 f_00201 f_00455] [--device cpu]

Picks three val frames (the model never trained on them) unless --frames is
given. CPU by default so it never competes with TRACK for the GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autolabel import CLASSES, DEFAULT_DATASET, ROOT, contact_sheet, draw_boxes  # noqa: E402
from preview import read_labels  # noqa: E402

PRED_IMGSZ = 960
PRED_CONF = 0.25


def caption(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 44)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([0, 0, 24 + len(text) * 24, 64], fill=(20, 20, 20))
    d.text((16, 8), text, fill=(255, 255, 255), font=font)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "best.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "results_labels.jpg")
    ap.add_argument("--frames", nargs="*", default=None, help="frame stems, default: 3 val frames with a ball")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    val_labels = sorted((args.dataset / "labels" / "val").glob("*.txt"))
    if args.frames:
        picked = [args.dataset / "labels" / "val" / f"{s}.txt" for s in args.frames]
    else:
        with_ball = [t for t in val_labels if any(l.startswith("1 ") for l in t.read_text().splitlines())]
        pool = with_ball if len(with_ball) >= 3 else val_labels
        step = max(1, len(pool) // 3)
        picked = pool[::step][:3]

    tiles = []
    for txt in picked:
        img_path = args.dataset / "images" / txt.parent.name / f"{txt.stem}.jpg"
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        labels = read_labels(txt, w, h)
        r = model.predict(image, imgsz=PRED_IMGSZ, conf=PRED_CONF, device=args.device, verbose=False)[0]
        preds = [(int(c), float(s), [float(v) for v in b])
                 for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        tiles.append(caption(draw_boxes(image, labels, w, h), f"{txt.stem}  auto-label ({len(labels)} boxes)"))
        tiles.append(caption(draw_boxes(image, preds, w, h), f"{txt.stem}  best.pt ({len(preds)} boxes)"))
        print(f"{txt.stem}: {len(labels)} labels, {len(preds)} predictions")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(tiles, cols=2, tile_w=960).save(args.out, quality=90)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
