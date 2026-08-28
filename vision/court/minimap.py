"""Top-down 2D court video from tracks.jsonl + court_calib.json.

    .venv/bin/python vision/court/minimap.py [--tracks out/tracks.jsonl] [--calib out/court_calib.json] [--out out/minimap.mp4]

Every player foot and the ball are projected to metres with the same helper
STATS uses (vision/court/project.py) and drawn on a 28 x 15 m court at 40 px/m,
team-coloured, with ids and a short ball trail. Output is H.264 so the
dashboard can embed it.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision.court.geometry import FIBA, SurfaceSpec, polylines  # noqa: E402
from vision.court.project import Calibration, iter_tracks, load_calibration  # noqa: E402
from vision.court.video import FfmpegWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# BGR. Team colours match the dashboard (#3B82F6 blue, #EF4444 red).
BG = (24, 24, 24)
COURT = (48, 44, 40)
LINE = (215, 215, 215)
TEAM = {0: (246, 130, 59), 1: (68, 68, 239), -1: (160, 160, 160)}
BALL = (0, 165, 255)
TEXT = (235, 235, 235)


class CourtCanvas:
    def __init__(self, spec: SurfaceSpec = FIBA, scale: float = 40.0, margin_m: float = 1.0):
        self.spec, self.scale, self.margin = spec, scale, margin_m
        self.w = int(round((spec.length_m + 2 * margin_m) * scale))
        self.h = int(round((spec.width_m + 2 * margin_m) * scale))
        self.base = self._draw_base()
        self.history: dict[int, deque] = {}  # per track id: last samples (frame, x_m, y_m) for the temporal median

    def px(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Metres -> canvas pixels. y is mirrored: court y grows upwards, image y downwards."""
        return (int(round((x_m + self.margin) * self.scale)),
                int(round(self.h - (y_m + self.margin) * self.scale)))

    def _draw_base(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), BG, np.uint8)
        cv2.rectangle(img, self.px(0, 0), self.px(self.spec.length_m, self.spec.width_m), COURT, -1)
        for poly in polylines(self.spec):
            pts = np.array([self.px(x, y) for x, y in poly], np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, LINE, 2, cv2.LINE_AA)
        for hx, hy in self.spec.hoops:
            cv2.circle(img, self.px(hx, hy), int(0.225 * self.scale), (0, 120, 255), 2, cv2.LINE_AA)
        return img


DISPLAY_TOLERANCE_M = 2.0
"""Points further than this outside the court are tracking or calibration errors, not players."""
BALL_JUMP_M = 3.0
"""A ball that moves more than this between two samples (40 ms apart) is a different ball
(fixture, false detection) or a calibration jump: the trail is broken there, never bridged."""
MEDIAN_SAMPLES = 3
HISTORY_MAX_GAP = 10  # frames; an id that vanished longer than this starts a fresh history


def _smoothed(canvas: CourtCanvas, pid: int, frame: int, xy) -> tuple[float, float]:
    """Temporal median over the last MEDIAN_SAMPLES positions of this track id."""
    h = canvas.history.get(pid)
    if h is None or (h and frame - h[-1][0] > HISTORY_MAX_GAP):
        h = canvas.history[pid] = deque(maxlen=MEDIAN_SAMPLES)
    h.append((frame, float(xy[0]), float(xy[1])))
    arr = np.array([[x, y] for _, x, y in h])
    med = np.median(arr, axis=0)
    return float(med[0]), float(med[1])


def render_frame(canvas: CourtCanvas, cal: Calibration, rec: dict, trail: deque, show_ids: bool) -> np.ndarray:
    img = canvas.base.copy()
    frame = int(rec["frame"])
    players = rec.get("players") or []
    calibrated = cal is not None and np.isfinite(cal.H_m_to_px(frame)).all()
    uncertain = calibrated and cal.is_uncertain(frame)
    if not calibrated:
        cv2.putText(img, "uncalibrated", (canvas.w - 190, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 240), 2, cv2.LINE_AA)
        players = []
        trail.append(None)
    elif uncertain:
        cv2.putText(img, "Kamera unsicher", (canvas.w - 230, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        players = []
    if players:
        feet = cal.project(frame, [p["foot"] for p in players])
        ok = cal.on_court(feet, DISPLAY_TOLERANCE_M)
        for p, xy, keep in zip(players, feet, ok):
            if not keep:
                continue
            xy = _smoothed(canvas, int(p["id"]), frame, xy)
            c = canvas.px(float(xy[0]), float(xy[1]))
            colour = TEAM.get(int(p.get("team", -1)), TEAM[-1])
            cv2.circle(img, c, 9, colour, -1, cv2.LINE_AA)
            cv2.circle(img, c, 9, (20, 20, 20), 1, cv2.LINE_AA)
            if show_ids:
                cv2.putText(img, str(p["id"]), (c[0] + 11, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
    ball = rec.get("ball")
    predicted_px = None
    if calibrated and not uncertain and ball and ball.get("center"):
        # a ball in flight projects "too far" along the ground plane; the bottom of its box is closer
        bx, by = ball["center"]
        if ball.get("bbox"):
            by = ball["bbox"][3]
        xy = cal.project(frame, [[bx, by]])[0]
        if cal.on_court(xy, DISPLAY_TOLERANCE_M)[0]:
            if ball.get("predicted"):
                # Kalman coasting from TRACK (conf 0): shown hollow, never part of the trail
                predicted_px = canvas.px(float(xy[0]), float(xy[1]))
                trail.append(None)
            else:
                last = next((p for p in reversed(trail) if p is not None), None)
                if last is not None and np.hypot(xy[0] - last[1], xy[1] - last[2]) > BALL_JUMP_M:
                    trail.append(None)  # break the trail: a 3 m jump in 40 ms is not the same ball
                trail.append((canvas.px(float(xy[0]), float(xy[1])), float(xy[0]), float(xy[1])))
        else:
            trail.append(None)
    else:
        trail.append(None)
    # draw only runs of consecutive samples, never across a None
    run: list = []
    runs: list = []
    for p in trail:
        if p is None:
            if run:
                runs.append(run)
            run = []
        else:
            run.append(p[0])
    if run:
        runs.append(run)
    n = sum(len(r) for r in runs) or 1
    k = 0
    for r in runs:
        for i in range(1, len(r)):
            k += 1
            a = 0.3 + 0.7 * k / n
            cv2.line(img, r[i - 1], r[i], tuple(int(v * a) for v in BALL), 2, cv2.LINE_AA)
    if trail and trail[-1] is not None:
        cv2.circle(img, trail[-1][0], 6, BALL, -1, cv2.LINE_AA)
    elif predicted_px is not None:
        cv2.circle(img, predicted_px, 6, BALL, 2, cv2.LINE_AA)
    t = rec.get("t", frame / (cal.fps if cal else 50.0))
    cv2.putText(img, f"{t:6.1f} s", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, default=ROOT / "out" / "tracks.jsonl")
    ap.add_argument("--calib", type=Path, default=None, help="Standard: out/court_calib_<clip>.json bei --clip, sonst out/court_calib.json")
    ap.add_argument("--clip", type=Path, default=None, help="Clip, zu dem tracks.jsonl gehört (wählt die Kalibrierdatei)")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "minimap.mp4")
    ap.add_argument("--scale", type=float, default=40.0, help="px per metre")
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: clip fps / detected stride)")
    ap.add_argument("--trail", type=float, default=0.6, help="ball trail length in seconds")
    ap.add_argument("--no-ids", action="store_true")
    ap.add_argument("--preview", type=Path, default=None, help="also write one PNG of the first frame with players")
    args = ap.parse_args(argv)

    if args.calib is None:
        per_clip = ROOT / "out" / f"court_calib_{args.clip.stem}.json" if args.clip else None
        args.calib = per_clip if per_clip and per_clip.exists() else ROOT / "out" / "court_calib.json"
    cal = load_calibration(args.calib) if args.calib.exists() else None
    if cal is None:
        print(f"{args.calib} fehlt: Platzhalter ohne Spieler, als uncalibrated markiert.")
    canvas = CourtCanvas(scale=args.scale)
    records = list(iter_tracks(args.tracks))
    if not records:
        raise SystemExit(f"keine Frames in {args.tracks}")
    frames = np.array([r["frame"] for r in records])
    stride = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
    fps = args.fps or (cal.fps if cal else 50.0) / max(stride, 1)
    trail: deque = deque(maxlen=max(1, int(args.trail * fps)))
    print(f"{len(records)} Frames, Stride {stride}, {fps:.1f} fps, Kalibrierung: {cal.mode if cal else 'keine'}, Canvas {canvas.w}x{canvas.h}")

    preview_done = args.preview is None
    with FfmpegWriter(args.out, canvas.w, canvas.h, fps) as writer:
        for rec in records:
            img = render_frame(canvas, cal, rec, trail, not args.no_ids)
            writer.write(img)
            if not preview_done and rec.get("players"):
                cv2.imwrite(str(args.preview), img)
                preview_done = True
    print(f"gespeichert: {args.out} ({writer.frames} Frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
