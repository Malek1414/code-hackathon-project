"""Team assignment sample: 60 random player crops grouped by assigned team.

    .venv/bin/python -m vision.qa.team_check [--n 60] [--seed 0]

One band per team (0, 1, unknown), crops framed in the team's overlay color.
A crop in the wrong band = team error. Count per band in one look.
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
    BG,
    META,
    QA_DIR,
    ROOT,
    TEAM_COLORS,
    TEAM_NAMES,
    TRACKS,
    FrameGrabber,
    band_label,
    put_text,
    read_json,
    read_tracks,
    resolve_clip,
    save_jpg,
    tile,
    with_header,
)

CROP_H, CROP_W, COLS, MIN_CONF, MARGIN = 240, 120, 15, 0.5, 0.08
TEAM_ORDER = (0, 1, -1)
JERSEY_HINT = {0: "blue jerseys", 1: "black or red jerseys", -1: "grey/white or unsure"}


def crop_player(img: np.ndarray, p: dict, frame: int) -> np.ndarray:
    x1, y1, x2, y2 = p["bbox"]
    w, h = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - MARGIN * w)), int(min(img.shape[1], x2 + MARGIN * w))
    y1, y2 = int(max(0, y1 - MARGIN * h)), int(min(img.shape[0], y2 + MARGIN * h))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((10, 10, 3), np.uint8)
    s = min(CROP_H / crop.shape[0], CROP_W / crop.shape[1])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((CROP_H, CROP_W, 3), BG, np.uint8)
    y0, x0 = (CROP_H - crop.shape[0]) // 2, (CROP_W - crop.shape[1]) // 2
    canvas[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
    color = TEAM_COLORS.get(p.get("team", -1), TEAM_COLORS[-1])
    cv2.rectangle(canvas, (0, 0), (CROP_W - 1, CROP_H - 1), color, 3)
    put_text(canvas, f"#{p['id']} f{frame}", (5, CROP_H - 8), 0.45, (255, 255, 255), 1)
    return canvas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", type=Path, default=TRACKS)
    ap.add_argument("--clip", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=QA_DIR)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    frames = read_tracks(args.tracks)
    if not frames:
        print(f"{args.tracks}: no frames yet")
        return 1
    meta = read_json(META) or {}
    clip = resolve_clip(str(args.clip) if args.clip else None, meta.get("clip"))
    rng = random.Random(args.seed)
    # one player per frame while frames last, so the sample spans the clip
    order = frames[:]
    rng.shuffle(order)
    picks: list[tuple[int, dict]] = []
    rounds = 0
    while len(picks) < args.n and rounds < 5:
        for f in order:
            cands = [p for p in f.get("players", []) if p.get("conf", 1) >= MIN_CONF]
            if not cands:
                continue
            taken = {p["id"] for fr, p in picks if fr == f["frame"]}
            cands = [p for p in cands if p["id"] not in taken]
            if not cands:
                continue
            picks.append((f["frame"], rng.choice(cands)))
            if len(picks) >= args.n:
                break
        rounds += 1
    picks.sort(key=lambda fp: fp[0])
    grab = FrameGrabber(clip)
    crops: dict[int, list[np.ndarray]] = {t: [] for t in TEAM_ORDER}
    listed = []
    last_frame, img = None, None
    for frame, p in picks:
        if frame != last_frame:
            img = grab.get(frame)
            last_frame = frame
        if img is None:
            continue
        team = p.get("team", -1) if p.get("team", -1) in crops else -1
        crops[team].append(crop_player(img, p, frame))
        listed.append({"frame": frame, "id": p["id"], "team": p.get("team", -1), "conf": p.get("conf")})
    grab.close()

    counts_all = {t: 0 for t in TEAM_ORDER}
    for f in frames:
        for p in f.get("players", []):
            counts_all[p.get("team", -1) if p.get("team", -1) in counts_all else -1] += 1
    bands = []
    width = COLS * (CROP_W + 6) + 6
    for t in TEAM_ORDER:
        color = TEAM_COLORS[t]
        bands.append(band_label(width, f"{TEAM_NAMES[t]}   {JERSEY_HINT[t]}   {len(crops[t])} crops here, {counts_all[t]} detections in tracks", color))
        grid = tile(crops[t], COLS) if crops[t] else band_label(width, "none", (120, 120, 120))
        if grid.shape[1] < width:
            pad = np.full((grid.shape[0], width - grid.shape[1], 3), BG, np.uint8)
            grid = np.hstack([grid, pad])
        bands.append(grid)
    body = np.vstack(bands)
    clip_label = str(clip.relative_to(ROOT)) if clip.is_relative_to(ROOT) else str(clip)
    head = [
        f"team check   {clip_label}   {len(listed)} random player crops from {len({f for f, _ in picks})} frames (seed {args.seed})",
        "every crop sits in the band of its assigned team. a jersey that does not match its band = team error. frame border = overlay color of that team",
        time.strftime("generated %Y-%m-%d %H:%M:%S"),
    ]
    sheet = with_header(body, head, 0.9)
    args.out.mkdir(parents=True, exist_ok=True)
    save_jpg(args.out / "teams.jpg", sheet, 88)
    (args.out / "teams.json").write_text(json.dumps({"clip": clip_label, "seed": args.seed, "counts_all": counts_all, "sampled": listed}, indent=1))
    print(f"{len(listed)} crops: " + ", ".join(f"{TEAM_NAMES[t]} {len(crops[t])}" for t in TEAM_ORDER) + f" -> {args.out / 'teams.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
