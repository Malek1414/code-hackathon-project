#!/usr/bin/env python3
"""Fine-tune YOLO11n on the self-labeled dataset from build_dataset.py.

Small on purpose: proves the full loop (footage -> pseudo-labels -> train ->
better court model) on a laptop. Scale epochs/data once real hand labels and
a hoop class exist (docs/ML_DATA_PLAN.md).

Usage:
  ml/.venv/bin/python ml/train_finetune.py --epochs 10
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data/dataset/data.yaml")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default="followcam_v0")
    a = ap.parse_args()

    model = YOLO("yolo11n.pt")
    results = model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz,
                          project="ml/runs", name=a.name, exist_ok=True,
                          patience=5, verbose=False)
    print(f"best weights: ml/runs/{a.name}/weights/best.pt")
    print("use with: ml/analyze_video.py <clip> --model ml/runs/"
          f"{a.name}/weights/best.pt")


if __name__ == "__main__":
    main()
