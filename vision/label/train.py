"""Fine-tune YOLO11n on the auto-labeled dataset and export models/best.pt.

Usage:
    .venv/bin/python vision/label/train.py [--epochs 10] [--imgsz 960] [--batch 4]
                                          [--device mps] [--time 0.4167] [--weights yolo11n.pt]

Defaults agreed with ORCH (Aug 28): batch 4, workers 0, cache off, because
16 GB RAM are shared with TRACK inference.
Timebox: `--time` is in hours (default 25 min). Note that ultralytics treats
`time` as an override for `epochs`: it measures the first epoch and then
schedules as many epochs as fit into the timebox (the Aug 28 run did 16 in
25.4 min instead of the nominal 10), and still writes best.pt. Pass
`--time 0` to train exactly `--epochs`. imgsz 960 rather than 640 because the
ball is ~7 px at 1080p and disappears when downscaled further
(courtside/engine/detect/players.py). Prints val mAP at the end.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "data" / "dataset" / "data.yaml"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATA_YAML)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--time", type=float, default=25 / 60, help="timebox in hours (default 25 min)")
    ap.add_argument("--workers", type=int, default=0, help="dataloader workers; 0 keeps RAM low next to TRACK")
    ap.add_argument("--name", default="label_yolo11n")
    ap.add_argument("--out", type=Path, default=MODELS / "best.pt", help="where best weights are copied")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"{args.data} missing, run autolabel.py first")
        return 1

    import torch
    from ultralytics import YOLO

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using cpu")
        device = "cpu"

    model = YOLO(args.weights)
    try:
        results = model.train(
            data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=device, time=args.time or None, project=str(RUNS), name=args.name, exist_ok=True,
            workers=args.workers, cache=False, patience=8, plots=True, verbose=True,
        )
    except RuntimeError as e:
        if device == "mps":
            print(f"MPS training failed ({str(e)[:100]}), retrying on cpu")
            device = "cpu"
            model = YOLO(args.weights)
            results = model.train(
                data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, time=args.time or None, project=str(RUNS), name=args.name, exist_ok=True,
                workers=args.workers, cache=False, patience=8, plots=True, verbose=True,
            )
        else:
            raise

    run_dir = Path(results.save_dir) if results is not None else RUNS / args.name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        best = run_dir / "weights" / "last.pt"
    if not best.exists():
        print(f"no weights in {run_dir / 'weights'}")
        return 1
    target = args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)

    metrics = YOLO(str(target)).val(data=str(args.data), imgsz=args.imgsz, device=device,
                                    batch=args.batch, plots=False, verbose=False)
    names = metrics.names
    print(f"\nval mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")
    for i, ap50 in zip(metrics.box.ap_class_index, metrics.box.ap50):
        print(f"  {names[int(i)]:<8} AP50={ap50:.3f}")
    print(f"weights -> {target} (from {best})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
