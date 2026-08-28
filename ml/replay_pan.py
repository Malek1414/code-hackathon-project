#!/usr/bin/env python3
"""Turn a tracked clip into servo pan commands — offline control-loop testing
and demo insurance (replay a good take to the rig with no live tracking).

Reads tracks.jsonl from analyze_video.py, targets the ball when visible
(largest person otherwise), and simulates the same control law the iOS app
uses plus the firmware's slew limit. Writes angles.csv; optionally streams
the angles to the Arduino in real time.

Usage:
  ml/.venv/bin/python ml/replay_pan.py clip_analysis/tracks.jsonl --video clip.mov
  ml/.venv/bin/python ml/replay_pan.py ... --serial /dev/cu.usbmodem14101   # live replay
"""
import argparse
import csv
import json
import time
from pathlib import Path

ANGLE_MIN, ANGLE_MAX = 40.0, 140.0
GAIN = 12.0          # deg per full frame of normalized error (matches app)
SLEW = 2.0           # deg per tick, firmware-matched (15 ms tick ~ frame at 30fps)


def target_cx(objects, width):
    balls = [o for o in objects if o["cls"] == "sports ball"]
    if balls:
        box = max(balls, key=lambda o: o["conf"])["xyxy"]
    else:
        persons = [o for o in objects if o["cls"] == "person"]
        if not persons:
            return None
        box = max(persons, key=lambda o: (o["xyxy"][2] - o["xyxy"][0]) *
                  (o["xyxy"][3] - o["xyxy"][1]))["xyxy"]
    return (box[0] + box[2]) / 2 / width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", help="tracks.jsonl from analyze_video.py")
    ap.add_argument("--video", help="source clip (to read frame width + fps)")
    ap.add_argument("--width", type=float, help="frame width if no --video")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--serial", help="stream angles to Arduino at clip fps")
    ap.add_argument("--out", default=None, help="angles csv (default next to tracks)")
    a = ap.parse_args()

    width, fps = a.width, a.fps
    if a.video:
        import cv2
        cap = cv2.VideoCapture(a.video)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.release()
    if not width:
        ap.error("need --video or --width")

    ser = None
    if a.serial:
        import serial
        ser = serial.Serial(a.serial, 115200, timeout=0.1)
        time.sleep(2)  # Arduino auto-reset on connect

    angle = 90.0
    rows = []
    with open(a.tracks) as f:
        for line in f:
            rec = json.loads(line)
            cx = target_cx(rec["objects"], width)
            if cx is not None:
                desired = angle + GAIN * (cx - 0.5)
                step = max(-SLEW, min(SLEW, desired - angle))
                angle = max(ANGLE_MIN, min(ANGLE_MAX, angle + step))
            rows.append((rec["frame"], round(rec["frame"] / fps, 3), round(angle, 1)))
            if ser:
                ser.write(f"A{int(round(angle))}\n".encode())
                time.sleep(1 / fps)

    out = Path(a.out) if a.out else Path(a.tracks).parent / "angles.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_sec", "angle_deg"])
        w.writerows(rows)

    angles = [r[2] for r in rows]
    print(f"{len(rows)} frames -> {out}")
    print(f"angle range {min(angles):.0f}-{max(angles):.0f} deg"
          + (", streamed to rig" if ser else ""))


if __name__ == "__main__":
    main()
