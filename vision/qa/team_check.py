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
    OUT,
    QA_DIR,
    ROOT,
    TEAM_COLORS,
    TEAM_NAMES,
    TRACKS,
    FrameGrabber,
    band_label,
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

CROP_H, CROP_W, COLS, MIN_CONF, MARGIN = 240, 120, 15, 0.5, 0.08
TEAM_ORDER = (0, 1, -1)
JERSEY_HINT = {0: "blue jerseys", 1: "black or red jerseys", -1: "grey/white or unsure"}
OFF = "off"  # pseudo band: off court per COURT's calibration (bench, spectators, referees at the table)
OFF_COLOR = (60, 60, 255)
ON_COURT_TOL_M = 0.5  # COURT's default 1.5 m keeps most of the bench "on court" (dev60: 4 % off at 1.5 m, 17 % at 0.5 m, 20 % at 0 m)


def load_court(clip: Path):
    """COURT's calibration for this clip if present (vision.court.project), else None."""
    for cand in (OUT / f"court_calib_{clip.stem}.json", OUT / "court_calib.json"):
        if cand.exists():
            try:
                from vision.court.project import load_calibration  # COURT owns this module

                return load_calibration(cand), cand
            except Exception as exc:  # calibration format moved on, the sheet must still render
                print(f"calibration {cand.name} unusable: {exc}")
                return None, None
    return None, None


def on_court_flags(cal, frame: int, feet: list[list[float]]) -> list[bool | None]:
    """True on court, False off court, None when the frame is uncalibrated."""
    if cal is None or not feet:
        return [None] * len(feet)
    pts = cal.project(frame, feet)
    if not np.isfinite(pts).all():
        return [None] * len(feet)
    return [bool(v) for v in cal.on_court(pts, tolerance_m=ON_COURT_TOL_M)]


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
    meta = meta_for(args.tracks)
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
    cal, calib_path = load_court(clip)
    grab = FrameGrabber(clip)
    crops: dict = {t: [] for t in TEAM_ORDER}
    crops[OFF] = []
    listed = []
    last_frame, img = None, None
    for frame, p in picks:
        if frame != last_frame:
            img = grab.get(frame)
            last_frame = frame
        if img is None:
            continue
        team = p.get("team", -1) if p.get("team", -1) in TEAM_COLORS else -1
        on = on_court_flags(cal, frame, [p.get("foot") or [(p["bbox"][0] + p["bbox"][2]) / 2, p["bbox"][3]]])[0]
        tile_img = crop_player(img, p, frame)
        if on is False:
            put_text(tile_img, "AUS", (CROP_W - 46, 22), 0.6, OFF_COLOR, 2)
            crops[OFF].append(tile_img)
        else:
            crops[team].append(tile_img)
        listed.append({"frame": frame, "id": p["id"], "team": p.get("team", -1), "conf": p.get("conf"), "on_court": on})
    grab.close()

    counts_all = {t: 0 for t in TEAM_ORDER}
    off_all = uncal_all = 0
    for f in frames:
        pl = f.get("players", [])
        for p in pl:
            counts_all[p.get("team", -1) if p.get("team", -1) in counts_all else -1] += 1
        if cal is not None and pl:
            flags = on_court_flags(cal, f["frame"], [p.get("foot") or [(p["bbox"][0] + p["bbox"][2]) / 2, p["bbox"][3]] for p in pl])
            off_all += sum(1 for v in flags if v is False)
            uncal_all += sum(1 for v in flags if v is None)
    total_all = sum(counts_all.values())
    bands = []
    width = COLS * (CROP_W + 6) + 6
    for t in TEAM_ORDER + (OFF,):
        color = OFF_COLOR if t == OFF else TEAM_COLORS[t]
        if t == OFF:
            if cal is None:
                continue
            head_txt = (
                f"off court per {calib_path.name}   {len(crops[OFF])} crops here, {off_all} of {total_all - uncal_all} calibrated detections "
                f"({100 * off_all / max(1, total_all - uncal_all):.0f}%) stand more than {ON_COURT_TOL_M:g} m outside the lines: bench, spectators, table"
            )
        else:
            head_txt = f"{TEAM_NAMES[t]}   {JERSEY_HINT[t]}   {len(crops[t])} crops here, {counts_all[t]} detections in tracks"
        bands.append(band_label(width, head_txt, color))
        grid = tile(crops[t], COLS) if crops[t] else band_label(width, "none", (120, 120, 120))
        if grid.shape[1] < width:
            pad = np.full((grid.shape[0], width - grid.shape[1], 3), BG, np.uint8)
            grid = np.hstack([grid, pad])
        bands.append(grid)
    body = np.vstack(bands)
    clip_label = str(clip.relative_to(ROOT)) if clip.is_relative_to(ROOT) else str(clip)
    head = [
        f"team check   {clip_label}   {len(listed)} random player crops from {len({f for f, _ in picks})} frames (seed {args.seed})",
        "every crop sits in the band of its assigned team. a jersey that does not match its band = team error. frame border = overlay color of that team"
        + ("; last band = off court per calibration (AUS), these should not count as players" if cal is not None else ""),
        time.strftime("generated %Y-%m-%d %H:%M:%S"),
    ]
    sheet = with_header(body, head, 0.9)
    args.out.mkdir(parents=True, exist_ok=True)
    save_jpg(args.out / "teams.jpg", sheet, 88)
    (args.out / "teams.json").write_text(
        json.dumps(
            {
                "clip": clip_label, "seed": args.seed, "counts_all": counts_all,
                "calibration": str(calib_path.relative_to(ROOT)) if calib_path else None,
                "off_court_all": off_all if cal is not None else None, "uncalibrated_all": uncal_all if cal is not None else None,
                "sampled": listed,
            },
            indent=1,
        )
    )
    print(
        f"{len(listed)} crops: " + ", ".join(f"{TEAM_NAMES[t]} {len(crops[t])}" for t in TEAM_ORDER)
        + (f", off court {len(crops[OFF])} ({off_all}/{total_all - uncal_all} overall)" if cal is not None else ", no calibration")
        + f" -> {args.out / 'teams.jpg'}"
    )
    return 0


if __name__ == "__main__":
    with qa_lock():
        raise SystemExit(main())
