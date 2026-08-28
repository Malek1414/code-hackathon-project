#!/usr/bin/env python3
"""Bench control for the FollowCam servo — flash servo_pan.ino first.

Interactive: j/k nudge -/+5deg, s = one full 40-140 sweep, c = center 90,
a number + Enter = absolute angle, q = quit.

Usage:
  python3 software/servo_test.py            # auto-detect /dev/cu.usbmodem*
  python3 software/servo_test.py --port /dev/cu.usbmodem14101 --sweep
"""
import argparse
import glob
import sys
import time

import serial

ANGLE_MIN, ANGLE_MAX = 40, 140


def find_port():
    ports = glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*")
    if not ports:
        sys.exit("no Arduino found (looked for /dev/cu.usbmodem*, /dev/cu.usbserial*)")
    return ports[0]


def send(ser, angle):
    angle = max(ANGLE_MIN, min(ANGLE_MAX, int(angle)))
    ser.write(f"A{angle}\n".encode())
    return angle


def sweep(ser):
    print("sweep 90 -> 140 -> 40 -> 90")
    for target in (140, 40, 90):
        send(ser, target)
        time.sleep(1.2)  # firmware slews ~133 deg/s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--sweep", action="store_true", help="one sweep, then exit")
    a = ap.parse_args()

    port = a.port or find_port()
    ser = serial.Serial(port, a.baud, timeout=0.1)
    print(f"connected {port} — waiting for Arduino reset...")
    time.sleep(2)
    angle = send(ser, 90)

    if a.sweep:
        sweep(ser)
        return

    print("j/k = -/+5deg | s = sweep | c = center | <number> = absolute | q = quit")
    while True:
        cmd = input(f"[{angle:3d} deg] > ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "j":
            angle = send(ser, angle - 5)
        elif cmd == "k":
            angle = send(ser, angle + 5)
        elif cmd == "s":
            sweep(ser)
            angle = 90
        elif cmd == "c":
            angle = send(ser, 90)
        elif cmd.isdigit():
            angle = send(ser, int(cmd))


if __name__ == "__main__":
    main()
