"""Celebration render for the pitch video: overlay + a 1.5 s animation at each made basket.

    .venv/bin/python -m vision.live.celebrate --video data/clips/whatsapp_1515.mp4 \
        --tracks out/whatsapp_1515/tracks.jsonl --events out/whatsapp_1515/events.json \
        --out out/whatsapp_1515/celebration.mp4
    # fallback without events: --events-manual "36.8:three,44.0:dunk"

Animation: expanding ring from the rim, particles, a big word (THREE / DUNK /
BASKET) scaling up and fading, team-colour accent. Classification without
calibration: DUNK = release within 1.5 hoop widths of the rim with the
shooter's box top above the rim line; THREE = release more than 4 hoop
widths from the rim horizontally or the arc apex more than 3 hoop heights
above the rim; else BASKET. Works in portrait (all sizes scale with the frame).
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision.live.overlay import BALL_COLOR, HOOP_COLOR, TEAM_COLOR  # noqa: E402

ACCENT = {0: (246, 130, 59), 1: (68, 68, 239), -1: (0, 165, 255)}  # BGR: blue, red, orange
DURATION_S = 1.5


def load_records(path: Path) -> tuple[list[int], dict[int, dict]]:
    recs = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        recs[int(r["frame"])] = r
    return sorted(recs), recs


def record_at(frames: list[int], recs: dict[int, dict], idx: int) -> dict | None:
    pos = bisect.bisect_right(frames, idx) - 1
    return recs[frames[pos]] if pos >= 0 else None


def classify(shot: dict, frames: list[int], recs: dict[int, dict]) -> str:
    """DUNK / THREE / BASKET from the tracks around the shot (no calibration)."""
    hoop = shot.get("hoop_bbox")
    if not hoop:
        return "BASKET"
    hw, hh = max(hoop[2] - hoop[0], 1.0), max(hoop[3] - hoop[1], 1.0)
    cx, rim_y = (hoop[0] + hoop[2]) / 2, hoop[1]
    rel_f = shot.get("release_frame")
    release = None
    box_top = None
    if rel_f is not None:
        r = record_at(frames, recs, int(rel_f))
        if r and r.get("ball"):
            release = r["ball"]["center"]
        if r and shot.get("player_id") is not None:
            for p in r.get("players") or []:
                if p["id"] == shot["player_id"]:
                    box_top = p["bbox"][1]
    apex = None
    if rel_f is not None:
        for f in frames:
            if int(rel_f) <= f <= int(shot["frame"]):
                b = recs[f].get("ball")
                if b and not b.get("predicted"):
                    apex = b["center"][1] if apex is None else min(apex, b["center"][1])
    if release is not None:
        dist = math.hypot(release[0] - cx, release[1] - rim_y)
        if dist <= 1.5 * hw and box_top is not None and box_top < rim_y:
            return "DUNK"
        if abs(release[0] - cx) > 4 * hw:
            return "THREE"
    if apex is not None and rim_y - apex > 3 * hh:
        return "THREE"
    return "BASKET"


def parse_manual(spec: str, fps: float) -> list[dict]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        t, _, kind = item.partition(":")
        out.append({"t": float(t), "frame": int(round(float(t) * fps)), "kind": (kind or "basket").upper(),
                    "team": -1, "made": True, "hoop_bbox": None})
    return out


def draw_overlay(frame: np.ndarray, rec: dict | None, trail: list[tuple[int, int]], k: float) -> None:
    if rec:
        for p in rec.get("players") or []:
            x1, y1, x2, y2 = (int(v) for v in p["bbox"])
            col = TEAM_COLOR.get(p.get("team", -1), TEAM_COLOR[-1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, max(1, int(2 * k)))
        for h in rec.get("hoops") or []:
            x1, y1, x2, y2 = (int(v) for v in h["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), HOOP_COLOR, max(1, int(2 * k)))
    for i in range(1, len(trail)):
        a = i / len(trail)
        cv2.line(frame, trail[i - 1], trail[i], tuple(int(c * a) for c in BALL_COLOR), max(1, int(2 * k)), cv2.LINE_AA)
    if trail:
        cv2.circle(frame, trail[-1], max(3, int(7 * k)), BALL_COLOR, max(1, int(2 * k)), cv2.LINE_AA)


class Celebration:
    def __init__(self, event: dict, rim: tuple[int, int], color, seed: int = 0) -> None:
        self.event, self.rim, self.color = event, rim, color
        rng = random.Random(seed)
        self.particles = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.0)) for _ in range(36)]

    def draw(self, frame: np.ndarray, age_s: float, k: float) -> None:
        if age_s < 0 or age_s > DURATION_S:
            return
        p = age_s / DURATION_S
        h, w = frame.shape[:2]
        base = min(w, h)
        # expanding rings from the rim
        for j in range(3):
            pr = p - j * 0.12
            if 0 <= pr <= 1:
                r = int(base * (0.05 + 0.75 * pr))
                thick = max(1, int(base * 0.012 * (1 - pr)))
                cv2.circle(frame, self.rim, r, self.color, thick, cv2.LINE_AA)
        # particles
        for ang, speed in self.particles:
            d = base * 0.6 * p * speed
            x, y = int(self.rim[0] + math.cos(ang) * d), int(self.rim[1] + math.sin(ang) * d)
            rad = max(1, int(base * 0.012 * (1 - p)))
            cv2.circle(frame, (x, y), rad, self.color, -1, cv2.LINE_AA)
        # the word: scales up, then fades
        word = self.event["kind"]
        scale = (0.6 + 1.4 * min(p / 0.3, 1.0)) * base / 400.0
        alpha = 1.0 if p < 0.6 else max(0.0, 1 - (p - 0.6) / 0.4)
        thick = max(2, int(scale * 3))
        (tw, th), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        x, y = (w - tw) // 2, int(h * 0.42) + th // 2
        overlay = frame.copy()
        cv2.putText(overlay, word, (x + thick, y + thick), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick * 3, cv2.LINE_AA)
        cv2.putText(overlay, word, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
        cv2.putText(overlay, word, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, self.color, max(1, thick // 3), cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def open_writer(path: Path, w: int, h: int, fps: float) -> subprocess.Popen:
    import imageio_ffmpeg

    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--events", default=None, help="events.json; made shots become celebrations")
    ap.add_argument("--events-manual", default=None, help='"36.8:three,44.0:dunk" (seconds:kind)')
    ap.add_argument("--out", required=True)
    ap.add_argument("--trail", type=int, default=12)
    args = ap.parse_args(argv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow, oh = w - w % 2, h - h % 2
    k = min(w, h) / 1080.0 * 1.6  # line widths relative to the frame

    frames, recs = ([], {})
    if args.tracks and Path(args.tracks).exists():
        frames, recs = load_records(Path(args.tracks))

    events: list[dict] = []
    if args.events and Path(args.events).exists():
        for s in json.loads(Path(args.events).read_text())["shots"]:
            if s.get("made"):
                events.append({**s, "kind": classify(s, frames, recs)})
    if args.events_manual:
        events += parse_manual(args.events_manual, fps)
    events.sort(key=lambda e: e["t"])
    print(f"{len(events)} celebrations:", [(round(e['t'], 2), e['kind']) for e in events])

    writer = open_writer(Path(args.out), ow, oh, fps)
    trail: list[tuple[int, int]] = []
    active: list[Celebration] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            rec = record_at(frames, recs, idx)
            if rec and rec.get("ball") and not rec["ball"].get("predicted"):
                trail.append(tuple(int(v) for v in rec["ball"]["center"]))
                trail = trail[-args.trail:]
            elif rec is None or not rec.get("ball"):
                trail = trail[1:] if trail and idx % 3 == 0 else trail
            draw_overlay(frame, rec, trail, k)
            for e in events:
                if e["frame"] <= idx < e["frame"] + 2 and not any(c.event is e for c in active):
                    hoop = e.get("hoop_bbox") or ((rec.get("hoops") or [{}])[0].get("bbox") if rec else None)
                    rim = (int((hoop[0] + hoop[2]) / 2), int(hoop[1])) if hoop else (w // 2, int(h * 0.3))
                    active.append(Celebration(e, rim, ACCENT.get(e.get("team", -1), ACCENT[-1]), seed=idx))
            for c in active:
                c.draw(frame, t - c.event["t"], k)
            active = [c for c in active if t - c.event["t"] <= DURATION_S]
            if (ow, oh) != (w, h):
                frame = frame[:oh, :ow]
            writer.stdin.write(frame.tobytes())
            idx += 1
    finally:
        writer.stdin.close()
        writer.wait()
        cap.release()
    print(f"{idx} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
