"""Blur heads (and everyone off court, once a calibration exists) using only out/tracks.jsonl.

Usage:
    .venv/bin/python vision/privacy/blur.py --video out/overlay.mp4 --tracks out/tracks.jsonl --out out/overlay_blurred.mp4
    .venv/bin/python vision/privacy/blur.py --video data/clips/dev60.mp4 --tracks out/tracks.jsonl \
        --stride 2 --out out/dev60_blurred.mp4        # raw clip: --stride from out/tracks_meta.json

No model runs here: the boxes come from TRACK's tracks.jsonl, so this is CPU
only and never touches the GPU. Per frame, for every box in "players" and in
the optional "others" list (referees, bench, spectators, if TRACK writes it):
  * head region = top HEAD_FRAC (22 percent) of the box, widened by
    WIDEN (20 percent) on each side, pixelated (mosaic + Gaussian blur).
  * when --calib (out/court_calib.json, H_px_to_m) is given, the foot point is
    projected to court meters; a box whose foot lands outside the court plus
    OFF_COURT_MARGIN_M is blurred entirely (spectators, bench, table).
Frame indexing: tracks lines carry the SOURCE frame index. For a video that
was written from the same lines (overlay.mp4, 1501 frames for 1501 lines)
--stride 1 means video frame n uses tracks line n. For the raw clip,
--stride N (tracks_meta.json "stride") means video frame v uses the line
with frame == v rounded down to a multiple of N. Frames without a line
are passed through unchanged (counted and printed).
Output: H.264 through the imageio_ffmpeg binary, yuv420p, faststart, no audio.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HEAD_FRAC = 0.22
WIDEN = 0.20
MOSAIC = 12          # pixelation block size in px, before the blur
OFF_COURT_MARGIN_M = 1.0
COURT_LENGTH_M = 28.0
COURT_WIDTH_M = 15.0


def load_tracks(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def row_for(video_frame: int, rows: list[dict], by_frame: dict[int, dict], stride: int) -> dict | None:
    """--stride 1: the video was written from the lines, video frame n is line n.
    --stride N: the video is the raw clip, video frame v is the line with
    source frame v rounded down to a multiple of N."""
    if stride == 1:
        return rows[video_frame] if video_frame < len(rows) else None
    return by_frame.get(video_frame - video_frame % stride)


def load_calib(path: Path | None):
    if path is None:
        return None
    data = json.loads(path.read_text())
    H = np.asarray(data["H_px_to_m"], dtype=np.float64)
    court = data.get("court_m", {})
    return H, float(court.get("length", COURT_LENGTH_M)), float(court.get("width", COURT_WIDTH_M))


def off_court(foot, calib) -> bool:
    if calib is None:
        return False
    H, length, width = calib
    p = H @ np.array([foot[0], foot[1], 1.0])
    if abs(p[2]) < 1e-9:
        return True
    x, y = p[0] / p[2], p[1] / p[2]
    m = OFF_COURT_MARGIN_M
    return not (-m <= x <= length + m and -m <= y <= width + m)


def head_region(bbox, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    hx1 = x1 - WIDEN * bw
    hx2 = x2 + WIDEN * bw
    hy1 = y1
    hy2 = y1 + HEAD_FRAC * bh
    return clamp(hx1, hy1, hx2, hy2, w, h)


def clamp(x1, y1, x2, y2, w: int, h: int) -> tuple[int, int, int, int]:
    return (max(0, int(x1)), max(0, int(y1)), min(w, int(round(x2))), min(h, int(round(y2))))


def pixelate(frame: np.ndarray, region: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = region
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False
    patch = frame[y1:y2, x1:x2]
    small = cv2.resize(patch, (max(1, (x2 - x1) // MOSAIC), max(1, (y2 - y1) // MOSAIC)),
                       interpolation=cv2.INTER_LINEAR)
    big = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    k = max(3, ((x2 - x1) // 8) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(big, (k, k), 0)
    return True


def blur_frame(frame: np.ndarray, row: dict | None, calib) -> tuple[int, int]:
    """Blur in place; returns (heads blurred, full boxes blurred)."""
    if row is None:
        return 0, 0
    h, w = frame.shape[:2]
    heads = full = 0
    # "others" is the optional list of non-player person boxes (referees,
    # bench, spectators) TRACK may write; treated exactly like players.
    for p in list(row.get("players", [])) + list(row.get("others", [])):
        bbox = p["bbox"]
        if off_court(p.get("foot", [(bbox[0] + bbox[2]) / 2, bbox[3]]), calib):
            full += pixelate(frame, clamp(*bbox, w, h))
        else:
            heads += pixelate(frame, head_region(bbox, w, h))
    return heads, full


def ffmpeg_writer(out: Path, w: int, h: int, fps: float) -> subprocess.Popen:
    import imageio_ffmpeg
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, default=Path("out/tracks.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="source frames per video frame: 1 for overlay.mp4, tracks_meta.json 'stride' for the raw clip")
    ap.add_argument("--calib", type=Path, default=None, help="out/court_calib.json; enables off-court full-box blur")
    ap.add_argument("--limit", type=int, default=0, help="stop after N video frames (testing)")
    args = ap.parse_args()

    if args.calib is not None and not args.calib.exists():
        print(f"calibration {args.calib} not found, off-court blur disabled")
        args.calib = None
    calib = load_calib(args.calib)
    rows = load_tracks(args.tracks)
    by_frame = {int(r["frame"]): r for r in rows}

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"cannot open {args.video}")
        return 1
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.stride == 1 and n_video != len(rows):
        print(f"warning: {n_video} video frames but {len(rows)} tracks lines; with --stride 1 they must "
              f"come from the same run (is TRACK still writing {args.tracks}?)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = ffmpeg_writer(args.out, w, h, fps)

    n = heads = full = missing = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.limit and n >= args.limit):
            break
        row = row_for(n, rows, by_frame, args.stride)
        if row is None:
            missing += 1
        hb, fb = blur_frame(frame, row, calib)
        heads += hb
        full += fb
        writer.stdin.write(frame.tobytes())
        n += 1
        if n % 250 == 0:
            print(f"{n} frames, {heads} heads, {full} full boxes, {missing} frames without tracks", flush=True)
    cap.release()
    writer.stdin.close()
    rc = writer.wait()
    print(f"{n} frames -> {args.out}: {heads} heads blurred, {full} off-court boxes blurred, "
          f"{missing} frames without tracks, calibration {'on' if calib else 'off'}, ffmpeg rc={rc}")
    return 0 if rc == 0 and n else 1


if __name__ == "__main__":
    sys.exit(main())
