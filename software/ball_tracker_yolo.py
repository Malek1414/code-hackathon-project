#!/usr/bin/env python3
"""FollowCam ball tracker with a real detector: YOLO ball -> pan angle over serial.

Same CLI and serial protocol as ball_tracker.py (Malek's HSV tracker stays the
fallback): `A<angle>\\n` at 115200 baud, servo 40 to 140 degrees, KP, deadband,
EMA. Detection is ultralytics with models/ball_hoop_avishah.pt (class 0 =
ball) at imgsz 640, conf 0.35, on the CPU by default (the GPU is scheduled for
other jobs and must stay free during the live demo); `--device mps` is
optional. Of several balls the one nearest the previous position wins. When
the model finds nothing for LOST_FRAMES frames in a row, the HSV mask from
ball_tracker.py takes over until the model sees the ball again.

Usage:
  .venv/bin/python software/ball_tracker_yolo.py                       # webcam 0, preview only
  .venv/bin/python software/ball_tracker_yolo.py --cam 1               # iPhone via Continuity Camera
  .venv/bin/python software/ball_tracker_yolo.py --video data/clips/dev60.mp4
  .venv/bin/python software/ball_tracker_yolo.py --serial /dev/tty.usbserial-XXXX
  .venv/bin/python software/ball_tracker_yolo.py --video clip.mp4 --headless --max-frames 600 --save-frames 3

Keys: q quit. Prints fps and ball hit rate on exit (and every 100 frames headless).
Deps: the repo .venv (ultralytics, opencv-python), pyserial for --serial.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "ball_hoop_avishah.pt"
BALL_CLASS = 0
IMGSZ = 640
ROI = 320               # once the ball is known, detect in a ROIxROI crop around it at native resolution
CONF = 0.35
LOST_FRAMES = 15        # frames without a model ball before HSV takes over
MAX_BALL_PX = 120       # a "ball" wider than this at 960x540 is not the ball

# Same knobs and values as ball_tracker.py
HSV_LO = np.array([5, 120, 90])
HSV_HI = np.array([20, 255, 255])
SERVO_CENTER = 90
SERVO_MIN, SERVO_MAX = 40, 140
KP = 0.06
DEADBAND_PX = 25
EMA = 0.35
MIN_AREA = 300
MIN_ROUNDNESS = 0.5     # blob area / enclosing-circle area; only used by the HSV fallback here
FRAME_W, FRAME_H = 960, 540


def open_serial(port):
    import serial
    s = serial.Serial(port, 115200, timeout=0.05)
    time.sleep(2)  # Arduino auto-reset
    return s


def hsv_ball(frame):
    """Malek's HSV path, unchanged: (x, y, r) of the largest orange blob or None."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (11, 11), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= MIN_AREA:
        return None
    (x, y), r = cv2.minEnclosingCircle(c)
    if area < MIN_ROUNDNESS * np.pi * r * r:
        return None  # a line, a wall box or a jersey edge, not a ball
    return int(x), int(y), int(r)


def yolo_ball(model, frame, device, prev):
    """Best ball box as (x, y, r): nearest to prev when there is a prev, else highest conf.

    With a previous position the model looks at a ROI x ROI crop around it at
    native resolution (the ball keeps its pixels, the model runs at imgsz ROI,
    4x cheaper than 640); without one it sees the whole frame at imgsz 640."""
    h, w = frame.shape[:2]
    ox = oy = 0
    src = frame
    imgsz = IMGSZ
    if prev is not None:
        ox = int(min(max(prev[0] - ROI // 2, 0), w - ROI))
        oy = int(min(max(prev[1] - ROI // 2, 0), h - ROI))
        src = frame[oy:oy + ROI, ox:ox + ROI]
        imgsz = ROI
    r = model.predict(src, imgsz=imgsz, conf=CONF, classes=[BALL_CLASS], device=device, verbose=False)[0]
    cands = []
    for box, conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        x1, y1, x2, y2 = box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy
        w, h = x2 - x1, y2 - y1
        if max(w, h) > MAX_BALL_PX:
            continue
        cands.append(((x1 + x2) / 2, (y1 + y2) / 2, max(w, h) / 2, conf))
    if not cands:
        return None
    if prev is not None:
        cands.sort(key=lambda c: (c[0] - prev[0]) ** 2 + (c[1] - prev[1]) ** 2)
    else:
        cands.sort(key=lambda c: -c[3])
    x, y, rad, _ = cands[0]
    return int(x), int(y), max(3, int(rad))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--device", default="cpu", help="cpu (default, keeps the GPU free) or mps")
    ap.add_argument("--headless", action="store_true", help="no windows (tests, ssh)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (tests)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after N seconds (tests)")
    ap.add_argument("--no-loop", action="store_true", help="do not loop a video file")
    ap.add_argument("--save-frames", type=int, default=0, help="save N frames with a detection to --out-dir")
    ap.add_argument("--out-dir", default=str(ROOT / "out" / "rig"))
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    cap = cv2.VideoCapture(args.video if args.video else args.cam)
    if not cap.isOpened():
        raise SystemExit("could not open video source")
    ser = open_serial(args.serial) if args.serial else None
    out_dir = Path(args.out_dir)
    if args.save_frames:
        out_dir.mkdir(parents=True, exist_ok=True)

    smooth_x = None
    angle = float(SERVO_CENTER)
    prev = None
    lost = LOST_FRAMES  # start in "lost" so HSV is allowed until the model sees a ball
    n = hits_model = hits_hsv = saved = 0
    t0 = time.time()
    save_every = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            if args.video and not args.no_loop and not args.max_frames and not args.max_seconds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        h, w = frame.shape[:2]
        n += 1

        target = yolo_ball(model, frame, args.device, prev)
        source = "yolo"
        if target is not None:
            lost = 0
            hits_model += 1
        else:
            lost += 1
            if lost >= 3:
                prev = None  # widen the search to the full frame
            if lost >= LOST_FRAMES:
                target = hsv_ball(frame)
                source = "hsv"
                if target is not None:
                    hits_hsv += 1

        if target:
            x, y, r = target
            prev = (x, y)
            smooth_x = x if smooth_x is None else EMA * x + (1 - EMA) * smooth_x
            err = smooth_x - w / 2
            if abs(err) > DEADBAND_PX:
                angle = float(np.clip(angle - KP * err, SERVO_MIN, SERVO_MAX))
                # sign convention: if the camera runs away from the ball, flip KP's sign
            color = (0, 255, 0) if source == "yolo" else (0, 200, 255)
            cv2.circle(frame, (x, y), r, color, 2)
            cv2.line(frame, (int(smooth_x), 0), (int(smooth_x), h), color, 1)

        if ser:
            ser.write(f"A{int(round(angle))}\n".encode())

        fps = n / max(1e-6, time.time() - t0)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        label = f"angle {angle:5.1f}  {('BALL ' + source) if target else 'lost'}  {fps:4.1f} fps"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if target else (0, 0, 255), 2)

        save_every = max(0, save_every - 1)
        if args.save_frames and target and source == "yolo" and saved < args.save_frames and save_every == 0:
            cv2.imwrite(str(out_dir / f"rig_{n:05d}.jpg"), frame)
            saved += 1
            save_every = 60  # spread the saved frames out

        if not args.headless:
            cv2.imshow("followcam", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
        elif n % 100 == 0:
            print(f"{n} frames  {fps:4.1f} fps  ball yolo {hits_model / n:.0%}  hsv {hits_hsv / n:.0%}", flush=True)
        if args.max_frames and n >= args.max_frames:
            break
        if args.max_seconds and time.time() - t0 >= args.max_seconds:
            break

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    dt = time.time() - t0
    print(f"{n} frames in {dt:.1f} s = {n / max(dt, 1e-6):.1f} fps on {args.device}, "
          f"ball by model in {hits_model}/{n} frames ({hits_model / max(n, 1):.0%}), "
          f"by HSV fallback in {hits_hsv} ({hits_hsv / max(n, 1):.0%}), "
          f"last angle {angle:.1f}" + (f", {saved} frames saved to {out_dir}" if saved else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
