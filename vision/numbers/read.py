"""Jersey-number reads per track id (NUMBERS role).

For every track id in out/tracks.jsonl take up to MAX_CROPS torso crops spread
over the track's lifetime (rows 15-60 % of the box, full width), rescale to
MAX_SIDE px (x3 was the plan; measured on CPU: a 1400 px crop takes 12-27 s,
256 px takes 0.8 s with the same reads, so small crops are upscaled and big
ones downscaled), run EasyOCR (CPU only: the MPS is scheduled for other jobs)
with a digit-only allowlist and keep reads of 1-2 digits with conf >= MIN_CONF.

Vote per track: sum of confidences per number. A number is assigned when the
winner has >= MIN_READS reads and >= WIN_SHARE of the vote mass.

Writes:
  out/numbers_reads.json   per track: team, votes, reads, number, conf, crops
  out/numbers_preview.jpg  contact sheet: crop, read text, confidence
  out/numbers_cache.json   OCR cache keyed by (frame, rounded bbox) so a re-run
                           after tracks.jsonl changed only OCRs new crops
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("numbers")

ROOT = Path(__file__).resolve().parents[2]
TRACKS = ROOT / "out" / "tracks.jsonl"
META = ROOT / "out" / "tracks_meta.json"
READS_OUT = ROOT / "out" / "numbers_reads.json"
PREVIEW_OUT = ROOT / "out" / "numbers_preview.jpg"
CACHE = ROOT / "out" / "numbers_cache.json"

MAX_CROPS = 12
TORSO_TOP, TORSO_BOTTOM = 0.15, 0.60
MAX_SIDE = 256  # px, longest side of the crop fed to the OCR
CPU_THREADS = 4
MIN_CONF = 0.4
MIN_READS = 2
WIN_SHARE = 0.6
MIN_BOX_H = 60  # px; smaller boxes have no legible number at 1080p
ALLOWLIST = "0123456789"

TILE_W, TILE_H = 120, 130  # preview tile


def load_tracks(path: Path = TRACKS) -> tuple[dict[int, list[dict]], str]:
    """id -> [{frame, t, team, bbox}], sorted by frame. Tolerates a partial last line."""
    tracks: dict[int, list[dict]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # writer is mid-line
            for p in d.get("players", []):
                tracks[int(p["id"])].append(
                    {"frame": d["frame"], "t": d["t"], "team": p.get("team", -1), "bbox": p["bbox"]}
                )
    for v in tracks.values():
        v.sort(key=lambda r: r["frame"])
    clip = "data/clips/dev60.mp4"
    for meta in (path.parent / "tracks_meta.json", META):
        if meta.exists():
            try:
                clip = json.load(open(meta)).get("clip", clip)
                break
            except json.JSONDecodeError:
                pass
    return dict(tracks), clip


def majority_team(rows: list[dict]) -> int:
    c = Counter(r["team"] for r in rows)
    known = [(n, t) for t, n in c.items() if t in (0, 1)]
    if not known:
        return -1
    return max(known)[1]


def pick_samples(rows: list[dict], k: int = MAX_CROPS) -> list[dict]:
    """k samples spread over the lifetime; within each time bin the tallest box wins."""
    rows = [r for r in rows if (r["bbox"][3] - r["bbox"][1]) >= MIN_BOX_H]
    if len(rows) <= k:
        return rows
    edges = np.linspace(0, len(rows), k + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = rows[a:b]
        out.append(max(seg, key=lambda r: r["bbox"][3] - r["bbox"][1]))
    return out


def torso_crop(frame: np.ndarray, bbox) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    ty1, ty2 = int(y1 + TORSO_TOP * h), int(y1 + TORSO_BOTTOM * h)
    tx1, tx2 = int(x1), int(x2)
    H, W = frame.shape[:2]
    ty1, ty2, tx1, tx2 = max(ty1, 0), min(ty2, H), max(tx1, 0), min(tx2, W)
    if ty2 - ty1 < 8 or tx2 - tx1 < 8:
        return None
    crop = frame[ty1:ty2, tx1:tx2]
    h, w = crop.shape[:2]
    s = MAX_SIDE / max(h, w)
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(crop, (max(8, int(w * s)), max(8, int(h * s))), interpolation=interp)


def crop_key(clip: str, frame_idx: int, bbox) -> str:
    return f"{Path(clip).stem}|{frame_idx}:" + ",".join(str(int(round(v))) for v in bbox)


def get_reader():
    import easyocr  # slow import, only when needed
    import torch

    torch.set_num_threads(CPU_THREADS)
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def ocr_digits(reader, img: np.ndarray) -> list[tuple[str, float]]:
    """All digit reads (1-2 digits, conf >= MIN_CONF) in the crop."""
    res = reader.readtext(img, allowlist=ALLOWLIST, detail=1, paragraph=False, canvas_size=MAX_SIDE, mag_ratio=1.0,
                          text_threshold=0.5, low_text=0.3)
    out = []
    for _box, text, conf in res:
        text = text.strip()
        if 1 <= len(text) <= 2 and text.isdigit() and conf >= MIN_CONF:
            out.append((text, float(conf)))
    return out


def vote(reads: list[tuple[str, float]]) -> tuple[int | None, float, dict[str, float], dict[str, int]]:
    mass: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for text, conf in reads:
        num = str(int(text))  # "07" -> "7"
        mass[num] += conf
        count[num] += 1
    if not mass:
        return None, 0.0, {}, {}
    total = sum(mass.values())
    winner = max(mass, key=mass.get)
    share = mass[winner] / total
    if count[winner] >= MIN_READS and share >= WIN_SHARE:
        return int(winner), round(share, 3), dict(mass), dict(count)
    return None, round(share, 3), dict(mass), dict(count)


def load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.load(open(CACHE))
        except json.JSONDecodeError:
            pass
    return {}


def read_frames(video: Path, wanted: set[int]):
    """Yield (frame_idx, frame) for wanted frame indices, decoding sequentially."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    last = max(wanted)
    idx = 0
    while idx <= last:
        if idx in wanted:
            ok, fr = cap.read()
            if not ok:
                break
            yield idx, fr
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()


def build_preview(tiles: list[tuple[np.ndarray, str, str]], out: Path, cols: int = 12) -> None:
    """tiles: (crop_bgr, header, footer). One row per track, MAX_CROPS tiles wide."""
    if not tiles:
        return
    imgs = []
    for crop, head, foot in tiles:
        tile = np.full((TILE_H, TILE_W, 3), 30, np.uint8)
        if crop is not None and crop.size:
            h, w = crop.shape[:2]
            s = min((TILE_W - 4) / w, (TILE_H - 34) / h)
            r = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
            y0 = 16
            x0 = (TILE_W - r.shape[1]) // 2
            tile[y0 : y0 + r.shape[0], x0 : x0 + r.shape[1]] = r
        cv2.putText(tile, head, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        color = (80, 220, 80) if foot and not foot.startswith("-") else (120, 120, 255)
        cv2.putText(tile, foot, (3, TILE_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        imgs.append(tile)
    while len(imgs) % cols:
        imgs.append(np.full((TILE_H, TILE_W, 3), 30, np.uint8))
    rows = [np.hstack(imgs[i : i + cols]) for i in range(0, len(imgs), cols)]
    sheet = np.vstack(rows)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])


def run(tracks_path: Path = TRACKS, video: Path | None = None, preview: bool = True) -> dict:
    t0 = time.time()
    tracks, clip = load_tracks(tracks_path)
    video = video or (ROOT / clip)
    cache = load_cache()

    # which crops do we need
    plan: dict[int, list[dict]] = {tid: pick_samples(rows) for tid, rows in tracks.items()}
    wanted_frames: set[int] = set()
    todo: dict[str, list[tuple[int, dict]]] = defaultdict(list)  # key -> [(tid, row)]
    for tid, rows in plan.items():
        for r in rows:
            k = crop_key(clip, r["frame"], r["bbox"])
            if k not in cache or preview:
                wanted_frames.add(r["frame"])
            todo[k].append((tid, r))
    need_ocr = [k for k in todo if k not in cache]
    log.info("tracks %d, crops %d, cached %d, to OCR %d, frames to decode %d",
             len(tracks), len(todo), len(todo) - len(need_ocr), len(need_ocr), len(wanted_frames))

    reader = get_reader() if need_ocr else None
    crops: dict[str, np.ndarray] = {}
    n_done = 0
    if wanted_frames:
        for fidx, frame in read_frames(video, wanted_frames):
            for k, users in todo.items():
                if users[0][1]["frame"] != fidx:
                    continue
                crop = torso_crop(frame, users[0][1]["bbox"])
                if crop is None:
                    cache[k] = []
                    continue
                if preview:
                    crops[k] = crop
                if k not in cache:
                    cache[k] = ocr_digits(reader, crop)
                    n_done += 1
                    if n_done % 100 == 0:
                        log.info("ocr %d/%d  %.1f s", n_done, len(need_ocr), time.time() - t0)
        json.dump(cache, open(CACHE, "w"))

    # vote
    result_tracks: dict[str, dict] = {}
    tiles: list[tuple[np.ndarray, str, str]] = []
    assigned = 0
    for tid in sorted(tracks):
        rows = tracks[tid]
        reads: list[tuple[str, float]] = []
        per_crop = []
        for r in plan[tid]:
            k = crop_key(clip, r["frame"], r["bbox"])
            rd = [tuple(x) for x in cache.get(k, [])]
            reads.extend(rd)
            per_crop.append((k, r, rd))
        number, share, mass, count = vote(reads)
        team = majority_team(rows)
        if number is not None:
            assigned += 1
        result_tracks[str(tid)] = {
            "team": team,
            "number": number,
            "conf": share if number is not None else 0.0,
            "votes": {k: round(v, 3) for k, v in sorted(mass.items(), key=lambda kv: -kv[1])},
            "counts": count,
            "reads": len(reads),
            "crops": len(per_crop),
            "frames": len(rows),
            "first_t": rows[0]["t"],
            "last_t": rows[-1]["t"],
        }
        if preview:
            head = f"id{tid} {'AB'[team] if team in (0, 1) else '?'}" + (f" #{number}" if number is not None else " -")
            for i, (k, r, rd) in enumerate(per_crop[:MAX_CROPS]):
                foot = " ".join(f"{t}:{c:.2f}" for t, c in rd[:2]) if rd else "-"
                tiles.append((crops.get(k), head if i == 0 else f"f{r['frame']}", foot))
            for _ in range(len(per_crop), MAX_CROPS):
                tiles.append((None, "", ""))

    out = {"clip": clip, "video": str(video.relative_to(ROOT)) if video.is_relative_to(ROOT) else str(video),
           "params": {"max_crops": MAX_CROPS, "min_conf": MIN_CONF, "min_reads": MIN_READS, "win_share": WIN_SHARE},
           "n_tracks": len(tracks), "n_assigned": assigned, "tracks": result_tracks}
    tmp = READS_OUT.with_suffix(".tmp")
    json.dump(out, open(tmp, "w"), indent=1)
    tmp.replace(READS_OUT)
    if preview:
        build_preview(tiles, PREVIEW_OUT, cols=MAX_CROPS)
    log.info("numbers: %d/%d tracks got a number (%.0f%%)  %.1f s  -> %s",
             assigned, len(tracks), 100 * assigned / max(1, len(tracks)), time.time() - t0, READS_OUT)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, default=TRACKS)
    ap.add_argument("--video", type=Path, default=None, help="default: clip from out/tracks_meta.json")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    run(a.tracks, a.video, preview=not a.no_preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
