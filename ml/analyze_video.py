#!/usr/bin/env python3
"""Track players + ball in a game/court video.

YOLO11 (COCO pretrained: person, sports ball) + ByteTrack via ultralytics.
Outputs an annotated video plus tracks.jsonl (per-frame ids/boxes) — the raw
material every event head (points/rebounds/assists) is built on.

Usage:
  ml/.venv/bin/python ml/analyze_video.py path/to/clip.mov
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

PERSON, SPORTS_BALL = 0, 32  # COCO class ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="any ultralytics weight; swap in a basketball fine-tune later")
    ap.add_argument("--out", default=None, help="output dir (default: <video>_analysis)")
    a = ap.parse_args()

    src = Path(a.video)
    outdir = Path(a.out) if a.out else src.parent / f"{src.stem}_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(a.model)
    results = model.track(source=str(src), classes=[PERSON, SPORTS_BALL],
                          tracker="bytetrack.yaml", stream=True, save=True,
                          project=str(outdir.resolve()), name="annotated", exist_ok=True,
                          verbose=False)

    n = 0
    with open(outdir / "tracks.jsonl", "w") as f:
        for i, r in enumerate(results):
            objects = []
            if r.boxes is not None and r.boxes.id is not None:
                for box, tid, cls, conf in zip(r.boxes.xyxy.tolist(),
                                               r.boxes.id.tolist(),
                                               r.boxes.cls.tolist(),
                                               r.boxes.conf.tolist()):
                    objects.append({"id": int(tid), "cls": model.names[int(cls)],
                                    "conf": round(conf, 3),
                                    "xyxy": [round(v, 1) for v in box]})
            f.write(json.dumps({"frame": i, "objects": objects}) + "\n")
            n = i + 1
    print(f"{n} frames -> {outdir}/tracks.jsonl + annotated video in {outdir}/annotated/")


if __name__ == "__main__":
    main()
