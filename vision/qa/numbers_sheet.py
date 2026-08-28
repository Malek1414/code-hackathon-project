"""Torso crops per identified player for the "Nummern pruefen" section.

For every player in out/identities.json with a number, plus the N longest
players without one, three crops from the largest boxes of its tracks
(distinct moments, at least 1 s apart) are tiled into out/qa/num_<key>.jpg.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .common import BG, OUT, TEAM_COLORS, FrameGrabber, fmt_t, put_text, read_json, save_jpg

IDENTITIES = OUT / "identities.json"
CROP_H, N_CROPS, MIN_GAP_FRAMES = 320, 3, 50  # 320 px = about 3x the native torso height on these clips
TOP, BOTTOM, SIDE = 0.02, 0.68, 0.12  # crop rows 2..68 % of the box (head to waist), 12 % extra width
TEAM_LETTER = {0: "A", 1: "B", -1: "X"}


def select_players(identities: dict, n_unnumbered: int = 10) -> list[dict]:
    players = identities.get("players") or []
    numbered = [p for p in players if p.get("number") is not None]
    rest = sorted((p for p in players if p.get("number") is None), key=lambda p: -p.get("frames", 0))
    return numbered + rest[:n_unnumbered]


def pick_boxes(
    frames: list[dict], track_ids: set[int], switch_t: dict[int, float] | None = None
) -> list[tuple[int, float, int, list[float]]]:
    """(frame, t, id, bbox) for the largest boxes of these tracks, spread in time.
    `switch_t[id]` = moment a ByteTrack id jumped to another player (NUMBERS);
    the number and team belong to the later segment, so earlier boxes are skipped."""
    switch_t = switch_t or {}
    cands = []
    for f in frames:
        t = f.get("t", 0.0)
        for p in f.get("players") or []:
            if p["id"] in track_ids and t >= switch_t.get(p["id"], -1.0):
                x1, y1, x2, y2 = p["bbox"]
                cands.append(((x2 - x1) * (y2 - y1), f["frame"], t, p["id"], p["bbox"]))
    cands.sort(reverse=True)
    # back views read best: among the larger half of the boxes prefer the
    # widest relative to its height (shoulders square to the camera)
    keep = max(6, len(cands) // 2)
    pool = cands[:keep]
    pool.sort(key=lambda c: (c[4][2] - c[4][0]) / max(1.0, c[4][3] - c[4][1]), reverse=True)
    picked: list[tuple[int, float, int, list[float]]] = []
    for _, frame, t, pid, bbox in pool:
        if all(abs(frame - q[0]) >= MIN_GAP_FRAMES for q in picked):
            picked.append((frame, t, pid, bbox))
        if len(picked) >= N_CROPS:
            break
    return sorted(picked)


def crop_torso(img: np.ndarray, bbox: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    ax1, ax2 = int(max(0, x1 - SIDE * w)), int(min(img.shape[1], x2 + SIDE * w))
    ay1, ay2 = int(max(0, y1 + TOP * h)), int(min(img.shape[0], y1 + BOTTOM * h))
    crop = img[ay1:ay2, ax1:ax2]
    if crop.size == 0:
        return np.full((CROP_H, CROP_H // 2, 3), BG, np.uint8)
    s = CROP_H / crop.shape[0]
    return cv2.resize(crop, (max(1, int(crop.shape[1] * s)), CROP_H), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)


def safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")


def build_number_cards(frames: list[dict], grab: FrameGrabber, out: Path, identities_path: Path = IDENTITIES) -> list[dict]:
    identities = read_json(identities_path)
    if not identities:
        return []
    players = select_players(identities)
    switch_t = {
        int(tid): float(info["switch_t"])
        for tid, info in (identities.get("tracks") or {}).items()
        if isinstance(info, dict) and info.get("switch_t") is not None
    }
    jobs = []  # (frame, player index, t, id, bbox)
    cards = []
    for i, p in enumerate(players):
        ids = {int(t) for t in p.get("track_ids", [])}
        boxes = pick_boxes(frames, ids, switch_t)
        cards.append(
            {
                "key": p.get("key"),
                "team": p.get("team", -1),
                "team_letter": TEAM_LETTER.get(p.get("team", -1), "X"),
                "detected": p.get("number"),
                "track_ids": sorted(ids),
                "frames": p.get("frames"),
                "first_t": p.get("first_t"),
                "last_t": p.get("last_t"),
                "crops": [{"frame": b[0], "t": b[1], "id": b[2]} for b in boxes],
                "switch_t": {str(t): switch_t[t] for t in ids if t in switch_t} or None,
                "img": f"num_{safe_name(p.get('key') or str(i))}.jpg" if boxes else None,
            }
        )
        for b in boxes:
            jobs.append((b[0], i, b[1], b[2], b[3]))
    jobs.sort()
    tiles: dict[int, list[np.ndarray]] = {i: [] for i in range(len(players))}
    last_frame, img = None, None
    for frame, i, t, pid, bbox in jobs:
        if frame != last_frame:
            img = grab.get(frame)
            last_frame = frame
        if img is None:
            continue
        tile = crop_torso(img, bbox)
        color = TEAM_COLORS.get(cards[i]["team"], TEAM_COLORS[-1])
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), color, 3)
        put_text(tile, f"{fmt_t(t)}  #{pid}", (5, tile.shape[0] - 8), 0.45, (255, 255, 255), 1)
        tiles[i].append(tile)
    for i, card in enumerate(cards):
        if not tiles[i] or not card["img"]:
            card["img"] = None
            continue
        pad = np.full((CROP_H, 6, 3), BG, np.uint8)
        row = [pad]
        for t in tiles[i]:
            row += [t, pad]
        save_jpg(out / card["img"], np.hstack(row), 90)
    keep = {c["img"] for c in cards if c["img"]}
    for stale in out.glob("num_*.jpg"):
        if stale.name not in keep:
            stale.unlink()
    return cards
