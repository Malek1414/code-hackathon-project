"""Ball recall sample: 40 random processed frames with the ball box (or "no ball").

    .venv/bin/python -m vision.qa.ball_recall [--n 40] [--seed 0]

Each tile: the frame scaled to 480 px wide with the ball box drawn, plus a 2x
zoom inset around the box so you can tell a ball from a head. Frames without a
ball box get a red "no ball" label. Count in one minute: box on a real ball =
hit, box elsewhere = false positive, "no ball" while a ball is visible = miss.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from .common import (
    BALL_COLOR,
    META,
    QA_DIR,
    ROOT,
    TRACKS,
    FrameGrabber,
    fit_width,
    fmt_t,
    meta_for,
    put_text,
    qa_lock,
    read_json,
    read_tracks,
    resolve_clip,
    save_jpg,
    tile,
    with_header,
)

TILE_W, COLS, INSET, ZOOM = 480, 8, 150, 2.0


def render_tile(img: np.ndarray, line: dict) -> np.ndarray:
    ball = line.get("ball")
    small = fit_width(img, TILE_W)
    s = TILE_W / img.shape[1]
    if ball:
        x1, y1, x2, y2 = ball["bbox"]
        cv2.rectangle(small, (int(x1 * s) - 2, int(y1 * s) - 2), (int(x2 * s) + 2, int(y2 * s) + 2), BALL_COLOR, 2)
        cx, cy = ball.get("center") or ((x1 + x2) / 2, (y1 + y2) / 2)
        half = int(INSET / ZOOM / 2)
        ax1, ay1 = int(max(0, cx - half)), int(max(0, cy - half))
        ax2, ay2 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
        crop = img[ay1:ay2, ax1:ax2]
        if crop.size:
            crop = cv2.resize(crop, (INSET, INSET), interpolation=cv2.INTER_LINEAR)
            zs = INSET / (ax2 - ax1)
            cv2.rectangle(
                crop,
                (int((x1 - ax1) * zs), int((y1 - ay1) * zs)),
                (int((x2 - ax1) * zs), int((y2 - ay1) * zs)),
                BALL_COLOR,
                2,
            )
            cv2.rectangle(crop, (0, 0), (INSET - 1, INSET - 1), BALL_COLOR, 2)
            small[4 : 4 + INSET, TILE_W - INSET - 4 : TILE_W - 4] = crop
        put_text(small, f"ball {ball.get('conf', 0):.2f}", (8, 24), 0.6, BALL_COLOR, 2)
    else:
        put_text(small, "no ball", (8, 24), 0.7, (60, 60, 255), 2)
    put_text(small, f"f{line['frame']}  {fmt_t(line.get('t', 0))}", (8, small.shape[0] - 10), 0.5)
    return small


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", type=Path, default=TRACKS)
    ap.add_argument("--clip", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=QA_DIR)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    frames = read_tracks(args.tracks)
    if not frames:
        print(f"{args.tracks}: no frames yet")
        return 1
    meta = meta_for(args.tracks)
    clip = resolve_clip(str(args.clip) if args.clip else None, meta.get("clip"))
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(frames, min(args.n, len(frames))), key=lambda f: f["frame"])
    grab = FrameGrabber(clip)
    tiles, listed = [], []
    for line in picked:
        img = grab.get(line["frame"])
        if img is None:
            continue
        tiles.append(render_tile(img, line))
        listed.append({"frame": line["frame"], "t": line.get("t"), "ball": bool(line.get("ball")), "conf": (line.get("ball") or {}).get("conf")})
    grab.close()
    with_ball = sum(1 for l in listed if l["ball"])
    total_ball = sum(1 for f in frames if f.get("ball"))
    clip_label = str(clip.relative_to(ROOT)) if clip.is_relative_to(ROOT) else str(clip)
    head = [
        f"ball recall sample   {clip_label}   {len(tiles)} of {len(frames)} processed frames (seed {args.seed})   "
        f"ball box in {with_ball}/{len(tiles)} sampled, {total_ball}/{len(frames)} overall ({100 * total_ball / len(frames):.0f}%)",
        "count: yellow box on the real ball = hit,  box on something else = false positive,  'no ball' while a ball is visible = miss.  inset = 2x zoom on the box",
        time.strftime("generated %Y-%m-%d %H:%M:%S"),
    ]
    sheet = with_header(tile(tiles, COLS), head, 0.9)
    args.out.mkdir(parents=True, exist_ok=True)
    save_jpg(args.out / "ball_recall.jpg", sheet, 88)
    (args.out / "ball_recall.json").write_text(
        json.dumps({"clip": clip_label, "seed": args.seed, "frames_total": len(frames), "ball_frames_total": total_ball, "sampled": listed}, indent=1)
    )
    print(f"{len(tiles)} tiles, ball box in {with_ball} of them, {total_ball}/{len(frames)} frames overall -> {args.out / 'ball_recall.jpg'}")
    return 0


if __name__ == "__main__":
    with qa_lock():
        raise SystemExit(main())
