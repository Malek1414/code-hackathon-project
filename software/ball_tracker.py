#!/usr/bin/env python3
"""FollowCam ball tracker: HSV color tracking -> pan angle over serial.

Usage:
  python3 ball_tracker.py                    # webcam 0, no serial (preview only)
  python3 ball_tracker.py --cam 1            # iPhone via Continuity Camera is often index 1
  python3 ball_tracker.py --video clip.mp4   # test on a file
  python3 ball_tracker.py --serial /dev/tty.usbserial-XXXX

Keys: q quit · s save HSV snapshot · trackbars tune the mask live.
Deps: pip install opencv-python pyserial
"""
import argparse, time

import cv2
import numpy as np

# Orange basketball starting range; tune live with the trackbars.
HSV_LO = np.array([5, 120, 90])
HSV_HI = np.array([20, 255, 255])

SERVO_CENTER = 90
SERVO_MIN, SERVO_MAX = 40, 140   # keep inside linkage range; widen after fit test
KP = 0.06          # degrees of servo per pixel of error — first knob to tune
DEADBAND_PX = 25   # ignore error smaller than this (stops oscillation)
EMA = 0.35         # ball position smoothing, 0..1 (higher = snappier)
MIN_AREA = 300     # px^2 — reject noise blobs


def open_serial(port):
    import serial
    s = serial.Serial(port, 115200, timeout=0.05)
    time.sleep(2)  # Arduino auto-reset
    return s


def make_trackbars():
    cv2.namedWindow("mask")
    for i, (name, val, mx) in enumerate([
        ("H lo", HSV_LO[0], 179), ("S lo", HSV_LO[1], 255), ("V lo", HSV_LO[2], 255),
        ("H hi", HSV_HI[0], 179), ("S hi", HSV_HI[1], 255), ("V hi", HSV_HI[2], 255),
    ]):
        cv2.createTrackbar(name, "mask", int(val), mx, lambda _: None)


def read_trackbars():
    g = lambda n: cv2.getTrackbarPos(n, "mask")
    return (np.array([g("H lo"), g("S lo"), g("V lo")]),
            np.array([g("H hi"), g("S hi"), g("V hi")]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else args.cam)
    if not cap.isOpened():
        raise SystemExit("could not open video source")
    ser = open_serial(args.serial) if args.serial else None

    make_trackbars()
    smooth_x = None
    angle = float(SERVO_CENTER)

    while True:
        ok, frame = cap.read()
        if not ok:
            if args.video:  # loop test clips
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        frame = cv2.resize(frame, (960, 540))
        h, w = frame.shape[:2]

        lo, hi = read_trackbars()
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (11, 11), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target = None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > MIN_AREA:
                (x, y), r = cv2.minEnclosingCircle(c)
                target = (int(x), int(y), int(r))

        if target:
            x, y, r = target
            smooth_x = x if smooth_x is None else EMA * x + (1 - EMA) * smooth_x
            err = smooth_x - w / 2
            if abs(err) > DEADBAND_PX:
                angle = float(np.clip(angle - KP * err, SERVO_MIN, SERVO_MAX))
                # sign convention: if the camera runs away from the ball, flip KP's sign
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.line(frame, (int(smooth_x), 0), (int(smooth_x), h), (0, 255, 0), 1)

        if ser:
            ser.write(f"A{int(round(angle))}\n".encode())

        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        cv2.putText(frame, f"angle {angle:5.1f}  {'BALL' if target else 'lost'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if target else (0, 0, 255), 2)
        cv2.imshow("followcam", frame)
        cv2.imshow("mask", mask)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            print("HSV_LO =", lo.tolist(), " HSV_HI =", hi.tolist())

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
