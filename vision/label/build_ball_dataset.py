"""Build data/dataset_ball/ (YOLO, classes of ball_hoop_avishah.pt) from Sami's hand ball labels.

Usage:
    .venv/bin/python vision/label/build_ball_dataset.py [--labels out/qa/ball_labels.json] [--device cpu]

Classes: 0 Basketball, 1 Basketball Hoop (the classes of models/ball_hoop_avishah.pt,
so the fine-tune starts from its head).
Images: the 120 hand-labeled game10 frames (data/frames/f_<i>.jpg, every 5th
second). Ball box = square of side 2r around Sami's (cx, cy, r). Hoop boxes:
from data/dataset/labels (LABEL's auto-labels, class 2 -> 1) when the frame
is in that set (odd i), else from the avishah model at conf 0.5, imgsz 1280.
Split by frame index, no leakage: ball frames sorted by i, first 60 train,
last 30 val; "none" frames (30, kept as background images) first 20 train,
last 10 val. Extra train material: LABEL's 85 cleaned auto ball labels from
frames that are not among the 120 and lie before the first val frame in
time (so nothing within 2 s of a val frame enters training).
Writes data/dataset_ball/{images,labels}/{train,val}, data.yaml and
split.json (which frame went where, for eval_ball.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "data" / "frames"
AUTO = ROOT / "data" / "dataset"
OUT = ROOT / "data" / "dataset_ball"
W, H = 1920, 1080
NAMES = {0: "Basketball", 1: "Basketball Hoop"}
HOOP_CONF = 0.5


def auto_labels(stem: str) -> list[str] | None:
    for split in ("train", "val"):
        p = AUTO / "labels" / split / f"{stem}.txt"
        if p.exists():
            return [l for l in p.read_text().splitlines() if l.strip()]
    return None


def hoops_from_auto(lines: list[str]) -> list[str]:
    out = []
    for l in lines:
        c, cx, cy, bw, bh = l.split()
        if c == "2":
            out.append(f"1 {cx} {cy} {bw} {bh}")
    return out


def ball_line(cx: float, cy: float, r: float) -> str:
    side = 2 * r
    return f"0 {cx / W:.6f} {cy / H:.6f} {side / W:.6f} {side / H:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, default=ROOT / "out" / "qa" / "ball_labels.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-train-ball", type=int, default=60)
    ap.add_argument("--n-train-none", type=int, default=20)
    args = ap.parse_args()

    frames = json.loads(args.labels.read_text())["frames"]
    for f in frames:
        f["i"] = int(f["file"][2:7])
    ball = sorted((f for f in frames if f["ball"]), key=lambda f: f["i"])
    none = sorted((f for f in frames if not f["ball"]), key=lambda f: f["i"])
    split = {}
    for k, f in enumerate(ball):
        split[f["file"]] = "train" if k < args.n_train_ball else "val"
    for k, f in enumerate(none):
        split[f["file"]] = "train" if k < args.n_train_none else "val"
    first_val_i = min(f["i"] for f in frames if split[f["file"]] == "val")

    for s in ("train", "val"):
        for kind in ("images", "labels"):
            d = OUT / kind / s
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    model = None
    need_model = [f for f in frames if auto_labels(f["file"][:-4]) is None]
    if need_model:
        from ultralytics import YOLO
        model = YOLO(str(ROOT / "models" / "ball_hoop_avishah.pt"))
    counts = {"train": {"ball": 0, "hoop": 0, "images": 0}, "val": {"ball": 0, "hoop": 0, "images": 0}}
    hoop_src = {"auto": 0, "model": 0}
    for f in frames:
        stem = f["file"][:-4]
        s = split[f["file"]]
        lines = []
        if f["ball"]:
            cx, cy, r = f["ball"]
            lines.append(ball_line(cx, cy, r))
            counts[s]["ball"] += 1
        auto = auto_labels(stem)
        if auto is not None:
            hoops = hoops_from_auto(auto)
            hoop_src["auto"] += len(hoops)
        else:
            r = model.predict(str(FRAMES / f["file"]), imgsz=1280, conf=HOOP_CONF, classes=[1],
                              device=args.device, verbose=False)[0]
            hoops = []
            for x1, y1, x2, y2 in r.boxes.xyxy.tolist():
                hoops.append(f"1 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            hoop_src["model"] += len(hoops)
        lines += hoops
        counts[s]["hoop"] += len(hoops)
        counts[s]["images"] += 1
        shutil.copy2(FRAMES / f["file"], OUT / "images" / s / f["file"])
        (OUT / "labels" / s / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    # extra train frames: LABEL's cleaned auto ball labels, before the first val frame, not among the 120
    hand = {f["file"] for f in frames}
    extra = 0
    for txt in sorted((AUTO / "labels").glob("*/*.txt")):
        stem = txt.stem
        i = int(stem[2:7])
        if f"{stem}.jpg" in hand or i >= first_val_i:
            continue
        lines = [l for l in txt.read_text().splitlines() if l.strip()]
        balls = [l for l in lines if l.startswith("0 ") is False and l.split()[0] == "1"]
        if not balls:
            continue
        out_lines = []
        for l in lines:
            c, cx, cy, bw, bh = l.split()
            if c == "1":
                out_lines.append(f"0 {cx} {cy} {bw} {bh}")
            elif c == "2":
                out_lines.append(f"1 {cx} {cy} {bw} {bh}")
        shutil.copy2(FRAMES / f"{stem}.jpg", OUT / "images" / "train" / f"{stem}.jpg")
        (OUT / "labels" / "train" / f"{stem}.txt").write_text("\n".join(out_lines) + "\n")
        split[f"{stem}.jpg"] = "train_extra"
        extra += 1
        counts["train"]["images"] += 1
        counts["train"]["ball"] += len(balls)
        counts["train"]["hoop"] += sum(1 for l in out_lines if l.startswith("1 "))

    names = "\n".join(f"  {i}: {n}" for i, n in NAMES.items())
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    summary = {"split": split, "first_val_frame_index": first_val_i, "counts": counts,
               "hoop_sources": hoop_src, "extra_train_frames": extra}
    (OUT / "split.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "split"}))
    print(f"-> {OUT / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
