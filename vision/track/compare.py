"""Compare detector combinations on a frame range, no overlay, numbers only.

    .venv/bin/python vision/track/compare.py --video data/clips/dev60.mp4 \
        --start 900 --end 3121 --stride 4

Configs: A yolo11s + avishah (current), B best.pt alone (contract classes),
C best.pt persons + avishah ball/hoop. Reports ball frame share, mean ball
conf, players per frame, hoop share, ids, s/frame. Writes out/compare.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.track.tracker import Tracker  # noqa: E402

log = logging.getLogger("compare")


def run(name: str, tr: Tracker, video: Path, start: int, end: int, stride: int) -> dict:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    confs, t0, idx = [], time.time(), start
    while idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % stride == 0:
            r = tr.step(frame, idx, idx / fps)
            if r["ball"]:
                confs.append(r["ball"]["conf"])
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
    a = p.parse_args()

    rows = [run("A yolo11s+avishah", Tracker(), a.video, a.start, a.end, a.stride)]
    if a.best.exists():
        rows.append(run("B best.pt", Tracker(weights=a.best, imgsz=a.best_imgsz),
                        a.video, a.start, a.end, a.stride))
        rows.append(run("C best.pt persons + avishah ball/hoop",
                        Tracker(weights_players=a.best, person_imgsz=a.best_imgsz),
                        a.video, a.start, a.end, a.stride))
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
