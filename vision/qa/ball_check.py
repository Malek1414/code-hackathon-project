"""Ball check video: raw clip + accepted ball box with a 1 s trail (yellow),
rejected ball candidates (red x + reason letter), hoop boxes (green) and a
running counter. Lets a human spot every ball box that sits on a wall object.

    .venv/bin/python -m vision.qa.ball_check [--rejects out/dev60/ball_rejects.jsonl]

Output: out/qa/ball_check.mp4 (H.264, 720p, normal speed) + ball_check.json.
Rejects file is optional (TRACK writes it); lines need "frame" and a "bbox"
or "center", "reason" is shown as its first letter (S static, R radius,
H head, G gate).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from .common import (
    BALL_COLOR,
    HOOP_COLOR,
    OUT,
    QA_DIR,
    ROOT,
    TRACKS,
    FrameGrabber,
    fmt_t,
    is_predicted,
    meta_for,
    put_text,
    qa_lock,
    read_tracks,
    resolve_clip,
)

REJECT_COLOR = (40, 40, 255)
TRAIL_S = 1.0
OUT_H = 720
DEFAULT_REJECTS = (OUT / "dev60" / "ball_rejects.jsonl", OUT / "ball_rejects.jsonl")


def read_rejects(path: Path | None) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    if path is None or not path.exists():
        return by_frame
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break
            if "frame" not in r:
                continue
            c = r.get("center")
            if c is None and r.get("bbox"):
                x1, y1, x2, y2 = r["bbox"]
                c = [(x1 + x2) / 2, (y1 + y2) / 2]
            if c is None and r.get("xyxy"):
                x1, y1, x2, y2 = r["xyxy"]
                c = [(x1 + x2) / 2, (y1 + y2) / 2]
            if c is None:
                continue
            by_frame[int(r["frame"])].append({"center": c, "reason": str(r.get("reason") or "?")})
    return by_frame


def draw_frame(img: np.ndarray, line: dict, rejects: list[dict], trail: deque, counter: str, t: float) -> None:
    for hp in line.get("hoops") or []:
        x1, y1, x2, y2 = (int(v) for v in hp["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), HOOP_COLOR, 3)
    for r in rejects:
        x, y = (int(v) for v in r["center"])
        cv2.line(img, (x - 10, y - 10), (x + 10, y + 10), REJECT_COLOR, 3)
        cv2.line(img, (x - 10, y + 10), (x + 10, y - 10), REJECT_COLOR, 3)
        put_text(img, r["reason"][:1].upper(), (x + 13, y + 6), 0.7, REJECT_COLOR, 2)
    pts = list(trail)
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], BALL_COLOR, 2 + i // 8)
    b = line.get("ball")
    if is_predicted(b):
        cx, cy = b.get("center") or ((b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2)
        cv2.circle(img, (int(cx), int(cy)), 22, BALL_COLOR, 2)  # hollow: coasting point, not a detection
    elif b:
        x1, y1, x2, y2 = (int(v) for v in b["bbox"])
        cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), BALL_COLOR, 4)
        put_text(img, f"{b.get('conf', 0):.2f}", (x1, max(30, y1 - 12)), 0.8, BALL_COLOR, 2)
    cv2.rectangle(img, (0, 0), (720, 96), (0, 0, 0), -1)
    put_text(img, counter, (16, 40), 1.0, (255, 255, 255), 2)
    put_text(img, f"{fmt_t(t)}   gelb = Ball + Spur 1 s, hohler Kreis = vorhergesagt, rot x = verworfen, gruen = Korb", (16, 80), 0.7, (200, 200, 200), 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", type=Path, default=TRACKS)
    ap.add_argument("--rejects", type=Path, default=None)
    ap.add_argument("--clip", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=QA_DIR)
    args = ap.parse_args(argv)

    frames = read_tracks(args.tracks)
    if not frames:
        print(f"{args.tracks}: no frames yet")
        return 1
    meta = meta_for(args.tracks)
    clip = resolve_clip(str(args.clip) if args.clip else None, meta.get("clip"))
    local_rejects = args.tracks.with_name("ball_rejects.jsonl")
    rejects_path = args.rejects or next((p for p in (local_rejects, *DEFAULT_REJECTS) if p.exists()), None)
    rejects = read_rejects(rejects_path)
    grab = FrameGrabber(clip)
    stride = max(1, int(np.median(np.diff([f["frame"] for f in frames]))) if len(frames) > 1 else 1)
    fps = grab.fps / stride
    trail_len = int(round(TRAIL_S * fps))
    out_w = int(round(grab.width * OUT_H / grab.height / 2)) * 2
    dst = args.out / "ball_check.mp4"
    tmp = dst.with_name("ball_check.tmp.mp4")
    args.out.mkdir(parents=True, exist_ok=True)
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{OUT_H}", "-r", f"{fps:.3f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-threads", "2",  # ORCH 13:41: renders must not starve the training / tracking runs
        str(tmp),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    trail: deque = deque(maxlen=trail_len)
    n_ball = n_rej = n = n_pred = 0
    t0 = time.time()
    try:
        for line in frames:
            img = grab.get(line["frame"])
            if img is None:
                continue
            n += 1
            rj = rejects.get(line["frame"], [])
            n_rej += len(rj)
            b = line.get("ball")
            if is_predicted(b):
                n_pred += 1
                cx, cy = b.get("center") or ((b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2)
                trail.append((int(cx), int(cy)))  # the trail follows the coasting point, the box does not
            elif b:
                n_ball += 1
                cx, cy = b.get("center") or ((b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2)
                trail.append((int(cx), int(cy)))
            else:
                trail.clear()
            counter = f"Ball sichtbar: {n_ball}/{n} Frames, verworfen: {n_rej}" + (f", vorhergesagt: {n_pred}" if n_pred else "")
            draw_frame(img, line, rj, trail, counter, line.get("t", 0.0))
            small = cv2.resize(img, (out_w, OUT_H), interpolation=cv2.INTER_AREA)
            proc.stdin.write(small.tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
        grab.close()
    if rc != 0 or not tmp.exists():
        print(f"ffmpeg failed ({rc})")
        return 1
    tmp.replace(dst)
    stamp = time.strftime("%H%M%S")
    stamped = args.out / f"ball_check_{stamp}.mp4"
    shutil.copyfile(dst, stamped)
    for old in sorted(args.out.glob("ball_check_[0-9]*.mp4"))[:-3]:
        old.unlink()
    clip_label = str(clip.relative_to(ROOT)) if clip.is_relative_to(ROOT) else str(clip)
    info = {
        "video": stamped.name,
        "tracks": str(args.tracks.relative_to(ROOT)) if args.tracks.is_relative_to(ROOT) else str(args.tracks),
        "clip": clip_label,
        "frames": n,
        "ball_frames": n_ball,
        "ball_share": round(n_ball / n, 4) if n else 0,
        "rejects": n_rej,
        "predicted_frames": n_pred,
        "rejects_file": str(rejects_path.relative_to(ROOT)) if rejects_path and rejects_path.is_relative_to(ROOT) else (str(rejects_path) if rejects_path else None),
        "fps": fps,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out / "ball_check.json").write_text(json.dumps(info, indent=1))
    print(f"{n} frames, ball in {n_ball} ({100 * info['ball_share']:.0f}%), {n_rej} rejects{'' if rejects_path else ' (no rejects file)'} -> {dst} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with qa_lock():
        raise SystemExit(main())
