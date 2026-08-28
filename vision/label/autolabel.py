"""Auto-label extracted frames and write a YOLO dataset.

Usage:
    .venv/bin/python vision/label/autolabel.py [--frames data/frames] [--dataset data/dataset]
                                              [--every 2] [--limit N] [--device mps|cpu]

Classes (contract in docs/ORCHESTRATION.md): 0 player, 1 ball, 2 hoop, 3 referee.

Two detectors per frame (decided with ORCH, Aug 28 11:50):
  * players + referees: Grounding DINO (grounding-dino-tiny), single pass with
    the prompt "basketball player. referee." at box/text threshold 0.2. The
    combined 4-class prompt at 0.3 lost most players (13 vs 32 on 4 dev
    frames), and the multi-phrase ball pass put "balls" on orange wall
    fixtures next to the backboard.
  * ball + hoop: models/ball_hoop_avishah.pt (ultralytics, 0 Basketball,
    1 Basketball Hoop) at imgsz 1280; ball conf 0.45 and only the best ball
    per frame, hoop conf 0.5 keep all. Tight boxes, no wall-fixture balls.
A referee box wins over an overlapping player box. Bench players get boxed
too; that is accepted noise.

Output:
    data/dataset/images/{train,val}/*.jpg   (85/15 split, deterministic)
    data/dataset/labels/{train,val}/*.txt   (class cx cy w h, normalized)
    data/dataset/data.yaml                  (written first, so a partial dataset trains)
    out/label_preview.jpg                   (4x4 contact sheet; first 16 frames early,
                                             spread over the whole run at the end)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES = ROOT / "data" / "frames"
DEFAULT_DATASET = ROOT / "data" / "dataset"
PREVIEW = ROOT / "out" / "label_preview.jpg"

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
GDINO_PROMPT = "basketball player. referee."
GDINO_BOX_THRESHOLD = 0.2
GDINO_TEXT_THRESHOLD = 0.2

BALL_HOOP_WEIGHTS = ROOT / "models" / "ball_hoop_avishah.pt"
BALL_HOOP_IMGSZ = 1280
BALL_CONF = 0.45
HOOP_CONF = 0.5
BALL_HOOP_CLASS = {0: 1, 1: 2}  # model class -> our class

CLASSES = ["player", "ball", "hoop", "referee"]
# Grounding DINO returns the matched phrase per box; longest phrases first.
PHRASE_TO_CLASS = [
    ("basketball player", 0),
    ("referee", 3),
    ("player", 0),
]
COLORS = {0: (0, 200, 255), 1: (255, 80, 0), 2: (0, 255, 120), 3: (255, 230, 0)}

# Sanity limits in normalized units.
MAX_BALL_SIDE = 0.06
MAX_HOOP_SIDE = 0.25
MIN_SIDE = 0.004

Det = tuple[int, float, list[float]]  # (class_id, score, [x1, y1, x2, y2] px)


def phrase_to_class(label: str) -> int | None:
    label = label.lower().strip()
    for phrase, cid in PHRASE_TO_CLASS:
        if phrase in label:
            return cid
    return None


def pick_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using cpu")
        return "cpu"
    return requested


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Labeler:
    def __init__(self, device: str):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from ultralytics import YOLO
        self.device = device
        self.processor = AutoProcessor.from_pretrained(GDINO_ID)
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(device).eval()
        self.ball_hoop = YOLO(str(BALL_HOOP_WEIGHTS))

    def to_cpu(self) -> None:
        self.device = "cpu"
        self.gdino = self.gdino.to("cpu")

    @torch.no_grad()
    def people(self, image: Image.Image) -> list[Det]:
        inputs = self.processor(images=image, text=GDINO_PROMPT, return_tensors="pt").to(self.device)
        outputs = self.gdino(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=GDINO_BOX_THRESHOLD,
            text_threshold=GDINO_TEXT_THRESHOLD, target_sizes=[image.size[::-1]],
        )[0]
        labels = results.get("text_labels") or results["labels"]
        out: list[Det] = []
        for box, score, label in zip(results["boxes"], results["scores"], labels):
            cid = phrase_to_class(str(label))
            if cid is not None:
                out.append((cid, float(score), [float(v) for v in box.tolist()]))
        return out

    def ball_and_hoop(self, image: Image.Image) -> list[Det]:
        r = self.ball_hoop.predict(image, imgsz=BALL_HOOP_IMGSZ, conf=min(BALL_CONF, HOOP_CONF),
                                   device=self.device, verbose=False)[0]
        balls: list[Det] = []
        hoops: list[Det] = []
        for box, score, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            cid = BALL_HOOP_CLASS.get(int(cls))
            if cid == 1 and score >= BALL_CONF:
                balls.append((1, float(score), [float(v) for v in box]))
            elif cid == 2 and score >= HOOP_CONF:
                hoops.append((2, float(score), [float(v) for v in box]))
        balls.sort(key=lambda d: -d[1])
        return balls[:1] + hoops

    def detect(self, image: Image.Image) -> list[Det]:
        people = self.people(image)
        by_class: dict[int, list[Det]] = {}
        for det in people:
            by_class.setdefault(det[0], []).append(det)
        kept_all: list[Det] = []
        for dets in by_class.values():  # class-wise NMS
            dets.sort(key=lambda d: -d[1])
            kept: list[Det] = []
            for det in dets:
                if all(iou(det[2], k[2]) < 0.6 for k in kept):
                    kept.append(det)
            kept_all.extend(kept)
        referees = [d for d in kept_all if d[0] == 3]
        people = [d for d in kept_all
                  if not (d[0] == 0 and any(iou(d[2], r[2]) > 0.5 for r in referees))]
        return people + self.ball_and_hoop(image)


def to_yolo(dets: list[Det], w: int, h: int) -> list[str]:
    lines = []
    for cid, _score, (x1, y1, x2, y2) in dets:
        x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw < MIN_SIDE or bh < MIN_SIDE:
            continue
        if cid == 1 and max(bw, bh) > MAX_BALL_SIDE:
            continue
        if cid == 2 and max(bw, bh) > MAX_HOOP_SIDE:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def draw_boxes(image: Image.Image, dets: list[Det], w: int, h: int) -> Image.Image:
    img = image.copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except OSError:
        font = ImageFont.load_default()
    for line in to_yolo(dets, w, h):
        cid, cx, cy, bw, bh = line.split()
        cid = int(cid)
        cx, cy, bw, bh = (float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h)
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        col = COLORS[cid]
        d.rectangle([x1, y1, x2, y2], outline=col, width=4)
        d.text((x1 + 4, max(0, y1 - 32)), CLASSES[cid], fill=col, font=font)
    return img


def contact_sheet(tiles: list[Image.Image], cols: int = 4, tile_w: int = 640) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (tile_w, tile_w * 9 // 16), (20, 20, 20))
    tile_h = round(tile_w * tiles[0].height / tiles[0].width)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t.resize((tile_w, tile_h)), ((i % cols) * tile_w, (i // cols) * tile_h))
    return sheet


def write_yaml(dataset: Path) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (dataset / "data.yaml").write_text(
        f"path: {dataset.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--preview", type=Path, default=PREVIEW)
    ap.add_argument("--every", type=int, default=2, help="label every Nth frame")
    ap.add_argument("--limit", type=int, default=0, help="only label the first N selected frames")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = sorted(args.frames.glob("*.jpg"))[:: max(1, args.every)]
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print(f"no frames in {args.frames}, run extract_frames.py first")
        return 1

    device = pick_device(args.device)
    labeler = Labeler(device)

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = args.dataset / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
    write_yaml(args.dataset)  # first, so a partial dataset is trainable at a hard cut

    rng = random.Random(args.seed)
    order = list(range(len(frames)))
    rng.shuffle(order)
    n_val = max(1, round(len(frames) * args.val_frac)) if len(frames) > 1 else 0
    val_idx = set(order[:n_val])

    counts = {c: 0 for c in CLASSES}
    frames_with_ball = 0
    preview_tiles: list[Image.Image] = []
    early_tiles: list[Image.Image] = []
    preview_every = max(1, len(frames) // 16)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, fp in enumerate(frames):
        image = Image.open(fp).convert("RGB")
        w, h = image.size
        try:
            dets = labeler.detect(image)
        except RuntimeError as e:
            if labeler.device != "cpu":
                print(f"{labeler.device} failed ({str(e)[:80]}), falling back to cpu")
                labeler.to_cpu()
                dets = labeler.detect(image)
            else:
                raise
        lines = to_yolo(dets, w, h)
        for line in lines:
            counts[CLASSES[int(line[0])]] += 1
        if any(line.startswith("1 ") for line in lines):
            frames_with_ball += 1
        split = "val" if i in val_idx else "train"
        shutil.copy2(fp, args.dataset / "images" / split / fp.name)
        (args.dataset / "labels" / split / f"{fp.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if i % preview_every == 0 and len(preview_tiles) < 16:
            preview_tiles.append(draw_boxes(image, dets, w, h))
        if i < 16:
            early_tiles.append(draw_boxes(image, dets, w, h))
            if i == 15:
                contact_sheet(early_tiles).save(args.preview, quality=88)
                print(f"early preview (first 16 frames) -> {args.preview}")
        if (i + 1) % 25 == 0 or i + 1 == len(frames):
            rate = (i + 1) / (time.time() - t0)
            print(f"{i + 1}/{len(frames)} frames, {rate:.2f} fps, counts={counts}, frames_with_ball={frames_with_ball}", flush=True)

    contact_sheet(preview_tiles).save(args.preview, quality=88)
    summary = {
        "frames": len(frames), "every": args.every, "train": len(frames) - n_val, "val": n_val,
        "counts": counts, "frames_with_ball": frames_with_ball,
        "device": labeler.device, "seconds": round(time.time() - t0, 1),
    }
    (args.dataset / "label_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    print(f"dataset -> {args.dataset / 'data.yaml'}\npreview -> {args.preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
