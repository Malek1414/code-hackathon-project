#!/usr/bin/env python3
"""Phone -> Arduino bridge: WebSocket in, serial out.

The FollowCam iOS app connects to ws://<this laptop>:8765 and sends pan
commands ("A95"). Each one is forwarded to the Arduino running
servo_pan.ino as "A95\n".

Usage:
  python3 software/pan_bridge.py --port /dev/cu.usbmodem14101
  python3 software/pan_bridge.py --dry-run          # no Arduino, just print
"""
import argparse
import asyncio

import websockets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="Arduino serial port, e.g. /dev/cu.usbmodem14101")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dry-run", action="store_true", help="print angles instead of writing serial")
    a = ap.parse_args()

    if a.dry_run:
        ser = None
    else:
        if not a.port:
            ap.error("--port required unless --dry-run")
        import serial
        ser = serial.Serial(a.port, a.baud, timeout=0.1)

    async def handle(ws):
        print(f"phone connected: {ws.remote_address}")
        async for msg in ws:
            line = msg.strip()
            if ser:
                ser.write((line + "\n").encode())
            print(f"-> {line}", end="\r")

    async def run():
        async with websockets.serve(handle, "0.0.0.0", 8765):
            print("bridge listening on ws://0.0.0.0:8765"
                  + (" (dry run)" if ser is None else f" -> {a.port}"))
            await asyncio.Future()

    asyncio.run(run())


if __name__ == "__main__":
    main()
