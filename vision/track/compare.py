"""Compare detector combinations on a frame range, no overlay, numbers only.

    .venv/bin/python vision/track/compare.py --video data/clips/dev60.mp4 \
        --start 900 --end 3121 --stride 4

Configs: A yolo11s + avishah (current) at ball conf 0.45 / 0.35 / 0.30 with
the gate and static filter active, B best.pt alone (contract classes),
C best.pt persons + avishah ball/hoop. Reports ball frame share, mean ball
conf, players per frame, hoop share, ids, s/frame; writes out/compare.json and
40 random annotated frames per config (the same frames for every config) to
out/compare/<config>/f<frame>.jpg for eyeballing recall vs false positives.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import random

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.track.tracker import Tracker  # noqa: E402

log = logging.getLogger("compare")


def draw(frame, r):
    img = frame.copy()
    for p in r["players"]:
        x1, y1, x2, y2 = (int(v) for v in p["bbox"])
        col = {0: (255, 140, 0), 1: (0, 0, 255)}.get(p["team"], (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
        cv2.putText(img, str(p["id"]), (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    for h in r["hoops"]:
        x1, y1, x2, y2 = (int(v) for v in h["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 120), 2)
    if r["ball"]:
        cx, cy = (int(v) for v in r["ball"]["center"])
        cv2.circle(img, (cx, cy), 18, (0, 220, 255), 3)
        cv2.putText(img, f"{r['ball']['conf']:.2f}", (cx + 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 220, 255), 2)
    cv2.putText(img, f"f{r['frame']} ball {'y' if r['ball'] else '-'}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def run(name: str, tr: Tracker, video: Path, start: int, end: int, stride: int,
        sample: set[int], outdir: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    d = outdir / name.split()[0]
    d.mkdir(parents=True, exist_ok=True)
    confs, t0, idx = [], time.time(), start
    while idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % stride == 0:
            r = tr.step(frame, idx, idx / fps)
            if r["ball"]:
                confs.append(r["ball"]["conf"])
            if idx in sample:
                cv2.imwrite(str(d / f"f{idx}.jpg"), cv2.resize(draw(frame, r), (1280, 720)),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
        idx += 1
    cap.release()
    s = tr.summary()
    n = max(s["frames_processed"], 1)
    row = {"config": name, "frames": s["frames_processed"],
           "s_per_frame": round((time.time() - t0) / n, 3),
           "ball_share": s["ball_frame_share"],
           "ball_conf_mean": round(sum(confs) / max(len(confs), 1), 3),
           "players_per_frame": s["players_per_frame"], "hoop_share": s["hoop_frame_share"],
           "ids": s["track_ids"], "ball_rejected_static": s["ball_rejected_static"]}
    log.info("%s", json.dumps(row))
    return row


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=Path("data/clips/dev60.mp4"))
    p.add_argument("--start", type=int, default=900)
    p.add_argument("--end", type=int, default=3121)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--best", type=Path, default=Path("models/best.pt"))
    p.add_argument("--best-imgsz", type=int, default=960)
    p.add_argument("--out", type=Path, default=Path("out/compare.json"))
    p.add_argument("--frames-dir", type=Path, default=Path("out/compare"))
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--ball-confs", default="0.45,0.35,0.30")
    a = p.parse_args()

    processed = list(range(a.start, a.end, a.stride))
    sample = set(random.Random(0).sample(processed, min(a.samples, len(processed))))
    args = (a.video, a.start, a.end, a.stride, sample, a.frames_dir)

    rows = []
    for cb in (float(v) for v in a.ball_confs.split(",")):
        rows.append(run(f"A{cb:.2f} yolo11s+avishah ball conf {cb:.2f}", Tracker(conf_ball=cb), *args))
    if a.best.exists():
        rows.append(run("B best.pt", Tracker(weights=a.best, imgsz=a.best_imgsz), *args))
        rows.append(run("C best.pt persons + avishah ball/hoop",
                        Tracker(weights_players=a.best, person_imgsz=a.best_imgsz), *args))
    else:
        log.warning("%s missing, only config A", a.best)
    a.out.write_text(json.dumps({"video": str(a.video), "start": a.start, "end": a.end,
                                 "stride": a.stride, "rows": rows}, indent=1))
    keys = ["config", "frames", "s_per_frame", "ball_share", "ball_conf_mean",
            "players_per_frame", "hoop_share", "ids", "ball_rejected_static"]
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r[k]) for k in keys))


if __name__ == "__main__":
    main()
