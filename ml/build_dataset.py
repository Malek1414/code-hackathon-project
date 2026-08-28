#!/usr/bin/env python3
"""Build a YOLO fine-tune dataset from analyzed clips via self-training:
frames are exported and the tracker's own high-confidence detections become
the labels (pseudo-labeling). Classes: 0=person, 1=ball.

Usage:
  ml/.venv/bin/python ml/build_dataset.py ml/data/open_video/*_analysis \
      --out ml/data/dataset --every 15
"""
import argparse
import json
import random
from pathlib import Path

import cv2

CLS = {"person": 0, "sports ball": 1}
MIN_CONF = {"person": 0.55, "sports ball": 0.35}  # ball is rarer; keep more


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dirs", nargs="+",
                    help="*_analysis dirs from analyze_video.py")
    ap.add_argument("--out", default="ml/data/dataset")
    ap.add_argument("--every", type=int, default=15, help="keep every Nth frame")
    ap.add_argument("--val", type=float, default=0.15)
    a = ap.parse_args()

    out = Path(a.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(0)
    kept = 0
    for d in map(Path, a.analysis_dirs):
        video = next((p for p in d.parent.glob(d.name.replace("_analysis", ".*"))
                      if p.suffix.lower() in (".mp4", ".mov", ".mkv")), None)
        if not video or not (d / "tracks.jsonl").exists():
            print(f"skip {d}: missing video or tracks")
            continue
        cap = cv2.VideoCapture(str(video))
        W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        with open(d / "tracks.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                if rec["frame"] % a.every:
                    continue
                labels = []
                for o in rec["objects"]:
                    if o["conf"] < MIN_CONF.get(o["cls"], 1.0):
                        continue
                    x1, y1, x2, y2 = o["xyxy"]
                    labels.append(f"{CLS[o['cls']]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                                  f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
                if not labels:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame"])
                ok, img = cap.read()
                if not ok:
                    continue
                split = "val" if random.random() < a.val else "train"
                stem = f"{video.stem}_{rec['frame']:06d}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / split / f"{stem}.txt").write_text("\n".join(labels))
                kept += 1
        cap.release()

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: person\n  1: ball\n")
    print(f"{kept} labeled frames -> {out}/data.yaml")


if __name__ == "__main__":
    main()
