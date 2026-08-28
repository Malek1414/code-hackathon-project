"""Ball recall and false positives of a ball/hoop model against Sami's hand labels (val split).

Usage:
    .venv/bin/python vision/label/eval_ball.py [--weights models/ball_hoop_avishah.pt] [--conf 0.35]
                                              [--imgsz 1280] [--device cpu] [--split val]

A prediction of class 0 (Basketball) hits when its IoU with the hand box
(square, side 2r) is >= 0.3. Recall = frames with a hit / frames with a ball.
False positives = predictions without a hit (counted over all frames of the
split, including the "none" frames). Prints per-conf numbers so before/after
of the fine-tune are comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "data" / "frames"


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "ball_hoop_avishah.pt")
    ap.add_argument("--labels", type=Path, default=ROOT / "out" / "qa" / "ball_labels.json")
    ap.add_argument("--split-file", type=Path, default=ROOT / "data" / "dataset_ball" / "split.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.35])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()

    from ultralytics import YOLO
    frames = json.loads(args.labels.read_text())["frames"]
    split = json.loads(args.split_file.read_text())["split"]
    frames = [f for f in frames if split.get(f["file"]) == args.split]
    model = YOLO(str(args.weights))
    min_conf = min(args.conf)
    preds = {}
    for f in frames:
        r = model.predict(str(FRAMES / f["file"]), imgsz=args.imgsz, conf=min_conf, classes=[0],
                          device=args.device, verbose=False)[0]
        preds[f["file"]] = list(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()))

    n_ball = sum(1 for f in frames if f["ball"])
    print(f"{args.weights.name}: {len(frames)} {args.split} frames, {n_ball} with a ball, imgsz {args.imgsz}, {args.device}")
    results = {}
    for conf in args.conf:
        hits = fps = 0
        for f in frames:
            gt = None
            if f["ball"]:
                cx, cy, r = f["ball"]
                gt = [cx - r, cy - r, cx + r, cy + r]
            hit = False
            for box, c in preds[f["file"]]:
                if c < conf:
                    continue
                if gt is not None and iou(box, gt) >= args.iou and not hit:
                    hit = True
                else:
                    fps += 1
            hits += hit
        results[conf] = {"recall": hits / max(n_ball, 1), "hits": hits, "false_positives": fps,
                         "precision": hits / max(hits + fps, 1)}
        print(f"  conf {conf:.2f}: recall {hits}/{n_ball} = {hits / max(n_ball, 1):.0%}, "
              f"false positives {fps} in {len(frames)} frames, precision {hits / max(hits + fps, 1):.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
