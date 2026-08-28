"""Fine-tune the ball/hoop detector on Sami's hand ball labels -> models/ball_hoop_v2.pt.

Usage:
    .venv/bin/python vision/label/train_ball.py [--epochs 12] [--imgsz 1280] [--batch 4] [--lr0 0.002]
                                               [--device mps] [--time 0.2] [--out models/ball_hoop_v2.pt]

Starts from models/ball_hoop_avishah.pt (same two classes, so the head is
kept), no frozen layers, on data/dataset_ball/data.yaml. `--time` (hours,
default 12 min) is the hard stop; ultralytics reschedules the epoch count
to fill it, so `--epochs` is the nominal value (pass --time 0 to train exactly
--epochs). Prints val AP50 per class, Basketball first. Do not start while
another model job holds the GPU (docs/ORCHESTRATION.md GPU schedule).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "dataset_ball" / "data.yaml"
RUNS = ROOT / "runs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "ball_hoop_avishah.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--time", type=float, default=0.2, help="timebox in hours (0 = train exactly --epochs)")
    ap.add_argument("--name", default="ball_v2")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "ball_hoop_v2.pt")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"{args.data} missing, run build_ball_dataset.py first")
        return 1
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    results = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        lr0=args.lr0, optimizer="AdamW", freeze=None, device=args.device,
        time=args.time or None, project=str(RUNS), name=args.name, exist_ok=True,
        workers=0, cache=False, patience=20, plots=True, verbose=True,
        mosaic=1.0, scale=0.3, fliplr=0.5,
    )
    run_dir = Path(results.save_dir) if results is not None else RUNS / args.name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        best = run_dir / "weights" / "last.pt"
    if not best.exists():
        print(f"no weights in {run_dir / 'weights'}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.out)

    m = YOLO(str(args.out)).val(data=str(args.data), imgsz=args.imgsz, device=args.device,
                                batch=args.batch, plots=False, verbose=False)
    print(f"\nval mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")
    for i, ap50 in zip(m.box.ap_class_index, m.box.ap50):
        print(f"  {m.names[int(i)]:<16} AP50={ap50:.3f}")
    print(f"weights -> {args.out} (from {best})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
