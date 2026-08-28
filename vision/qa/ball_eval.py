"""Score the tracker's ball against Sami's hand labels (out/qa/ball_labels.json).

    .venv/bin/python -m vision.qa.ball_eval [--tracks out/tracks.jsonl] [--labels out/qa/ball_labels.json]

Per labeled game10 frame (1 fps, frame = (i-1)*50+24, see ball_label.FRAME_OFFSET)
the tracks line with the same "frame": hit = tracker ball centre within 1.5 r of Sami's centre, miss = Sami
ball, tracker none, false = tracker ball on a "none" frame or farther than 3 r
(1.5 r to 3 r counts as "off", a miss for recall and not a hit for precision).
Prints recall, precision and the misses, writes out/qa/ball_eval.json and a
contact sheet out/qa/ball_eval.jpg of the 12 worst frames (Sami green, tracker
yellow). Only for game10 tracks; other clips exit quietly.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .ball_label import frame_of
from .common import BALL_COLOR, OUT, QA_DIR, ROOT, fit_height, is_predicted, put_text, qa_lock, read_json, read_tracks, save_jpg, tile, with_header

LABELS = QA_DIR / "ball_labels.json"
FRAMES_DIR = ROOT / "data" / "frames"
HIT_R, FALSE_R = 1.5, 3.0
SAMI_COLOR = (80, 230, 80)
N_WORST, TILE_H, COLS = 12, 480, 4
TRACK_CANDIDATES = (OUT / "tracks.jsonl", OUT / "game10" / "tracks.jsonl", OUT / "game10_v1" / "tracks.jsonl")


def is_game10(tracks: Path) -> bool:
    meta = read_json(tracks.with_name("tracks_meta.json")) or {}
    return "game10" in tracks.parent.name or str(meta.get("clip", "")).endswith("game10.mp4")


def pick_tracks(explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    for cand in TRACK_CANDIDATES:
        if cand.exists() and is_game10(cand):
            return cand
    return None


def evaluate(labels: list[dict], lines: dict[int, dict]) -> tuple[list[dict], dict]:
    rows = []
    for lb in labels:
        fr = frame_of(int(lb["file"][2:7]))  # from the file name; older label files carry game10_frame without the +24 offset
        line = lines.get(fr)
        if line is None and fr - 1 in lines:  # tracks with stride 2 and an odd target
            line = lines[fr - 1]
        tb = (line or {}).get("ball")
        predicted = is_predicted(tb)
        if predicted:
            tb = None  # coasting point: counted in "predicted", never as a detection
        tc = None
        if tb:
            x1, y1, x2, y2 = tb["bbox"]
            tc = tb.get("center") or [(x1 + x2) / 2, (y1 + y2) / 2]
        row = {"file": lb["file"], "frame": fr, "status": lb["status"], "sami": lb.get("ball"), "tracker": tb, "predicted": predicted, "dist_px": None, "dist_r": None}
        if line is None:
            row["result"] = "unprocessed"
        elif lb["status"] == "open":
            row["result"] = "unlabeled"
        elif lb["status"] == "none":
            row["result"] = "false" if tb else "true_none"
        else:
            cx, cy, r = lb["ball"]
            if not tb:
                row["result"] = "miss"
            else:
                d = math.hypot(tc[0] - cx, tc[1] - cy)
                row["dist_px"] = round(d, 1)
                row["dist_r"] = round(d / max(r, 1.0), 2)
                row["result"] = "hit" if d <= HIT_R * r else ("off" if d <= FALSE_R * r else "false")
        rows.append(row)
    c = {k: sum(1 for r in rows if r["result"] == k) for k in ("hit", "miss", "off", "false", "true_none", "unprocessed", "unlabeled")}
    ball_frames = sum(1 for r in rows if r["status"] == "ball" and r["result"] != "unprocessed")
    tracker_balls = sum(1 for r in rows if r["tracker"] and r["result"] != "unprocessed")
    summary = {
        **c,
        "predicted": sum(1 for r in rows if r["predicted"] and r["result"] != "unprocessed"),
        "predicted_on_ball_frames": sum(1 for r in rows if r["predicted"] and r["status"] == "ball" and r["result"] != "unprocessed"),
        "labeled_ball_frames": ball_frames,
        "labeled_none_frames": sum(1 for r in rows if r["status"] == "none" and r["result"] != "unprocessed"),
        "tracker_ball_frames": tracker_balls,
        "recall": round(c["hit"] / ball_frames, 3) if ball_frames else None,
        "precision": round(c["hit"] / tracker_balls, 3) if tracker_balls else None,
    }
    return rows, summary


def worst(rows: list[dict]) -> list[dict]:
    order = {"false": 0, "miss": 1, "off": 2}
    bad = [r for r in rows if r["result"] in order]
    bad.sort(key=lambda r: (order[r["result"]], -(r["dist_r"] or 0), r["frame"]))
    return bad[:N_WORST]


def draw_tile(row: dict) -> np.ndarray | None:
    img = cv2.imread(str(FRAMES_DIR / row["file"]))
    if img is None:
        return None
    if row["sami"]:
        cx, cy, r = (int(v) for v in row["sami"])
        cv2.circle(img, (cx, cy), max(r, 4), SAMI_COLOR, 3)
        cv2.circle(img, (cx, cy), 3, SAMI_COLOR, -1)
    tb = row["tracker"]
    if tb:
        x1, y1, x2, y2 = (int(v) for v in tb["bbox"])
        cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), BALL_COLOR, 3)
        if row["sami"]:
            cx, cy, _ = (int(v) for v in row["sami"])
            cv2.line(img, (cx, cy), ((x1 + x2) // 2, (y1 + y2) // 2), BALL_COLOR, 1)
    small = fit_height(img, TILE_H)
    txt = {"false": "FALSCH", "miss": "VERPASST", "off": "DANEBEN"}.get(row["result"], row["result"])
    detail = f"{row['dist_px']:.0f} px = {row['dist_r']:.1f} r" if row["dist_px"] is not None else ("Sami: kein Ball" if row["status"] == "none" else "Tracker: kein Ball")
    put_text(small, f"{txt}   {row['file']}  f{row['frame']}   {detail}", (10, 28), 0.7, (255, 255, 255), 2)
    return small


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--out", type=Path, default=QA_DIR)
    args = ap.parse_args(argv)

    data = read_json(args.labels)
    if not data:
        print(f"{args.labels} missing")
        return 1
    tracks = pick_tracks(args.tracks)
    if tracks is None:
        print("no game10 tracks yet (out/tracks.jsonl is another clip), nothing to score")
        return 0
    lines = {f["frame"]: f for f in read_tracks(tracks)}
    if not lines:
        print(f"{tracks}: empty")
        return 0
    rows, summary = evaluate(data["frames"], lines)
    summary.update({"tracks": str(tracks.relative_to(ROOT)), "labels": str(args.labels.relative_to(ROOT)), "labels_saved": data.get("saved"),
                    "tracks_frames": len(lines), "tracks_last_frame": max(lines), "generated": time.strftime("%Y-%m-%d %H:%M:%S")})
    misses = [r["frame"] for r in rows if r["result"] == "miss"]
    falses = [r["frame"] for r in rows if r["result"] == "false"]
    offs = [r["frame"] for r in rows if r["result"] == "off"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ball_eval.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=1))
    tiles = [t for t in (draw_tile(r) for r in worst(rows)) if t is not None]
    rec = f"{100 * summary['recall']:.0f}%" if summary["recall"] is not None else "n/a"
    prec = f"{100 * summary['precision']:.0f}%" if summary["precision"] is not None else "n/a"
    head = [
        f"ball eval   {summary['tracks']} ({len(lines)} frames, last {max(lines)})   vs {args.labels.name}   "
        f"recall {rec}   precision {prec}   hit {summary['hit']}  miss {summary['miss']}  off {summary['off']}  false {summary['false']}  "
        f"unprocessed {summary['unprocessed']}   predicted (Kalman) {summary['predicted']}",
        f"gruen = Samis Kreis, gelb = Tracker-Box.  hit <= {HIT_R} r, off <= {FALSE_R} r, false > {FALSE_R} r oder Ball auf einem 'kein Ball' Frame.  {N_WORST} schlechteste Frames",
        summary["generated"],
    ]
    if tiles:
        save_jpg(args.out / "ball_eval.jpg", with_header(tile(tiles, COLS), head, 0.9), 88)
    print(
        f"{summary['tracks']}: recall {rec} ({summary['hit']}/{summary['labeled_ball_frames']}), precision {prec} ({summary['hit']}/{summary['tracker_ball_frames']}), "
        f"miss {summary['miss']}, off {summary['off']}, false {summary['false']}, true none {summary['true_none']}, unprocessed {summary['unprocessed']}, "
        f"predicted {summary['predicted']} (on ball frames {summary['predicted_on_ball_frames']})"
    )
    print(f"misses (frames): {misses}")
    print(f"false (frames): {falses}")
    if offs:
        print(f"off 1.5..3 r (frames): {offs}")
    return 0


if __name__ == "__main__":
    with qa_lock():
        raise SystemExit(main())
