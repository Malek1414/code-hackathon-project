"""Jersey-number reads per track id (NUMBERS role).

For every track id in out/tracks.jsonl take up to MAX_CROPS torso crops spread
over the track's lifetime (rows 15-60 % of the box, full width), rescale to
MAX_SIDE px (x3 was the plan; measured on CPU: a 1400 px crop takes 12-27 s,
256 px takes 0.8 s with the same reads, so small crops are upscaled and big
ones downscaled), run EasyOCR (CPU only: the MPS is scheduled for other jobs)
with a digit-only allowlist and keep reads of 1-2 digits with conf >= MIN_CONF.
Black/red jerseys are read on the red channel (red_channel), blue ones plain.

Vote per track: sum of confidences per number. A number is assigned when the
winner has >= MIN_READS reads and >= WIN_SHARE of the vote mass, or when the
only read of the track is a single one with conf >= STRONG_READ.

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

from vision.track.teams import rule_label, torso_color  # TRACK's jersey color rule, per crop

log = logging.getLogger("numbers")

ROOT = Path(__file__).resolve().parents[2]
TRACKS = ROOT / "out" / "tracks.jsonl"
META = ROOT / "out" / "tracks_meta.json"
# outputs live next to the tracks file: out/ for the contract paths, out/game10/ for an archive copy
READS_NAME = "numbers_reads.json"
PREVIEW_NAME = "numbers_preview.jpg"
CACHE_NAME = "numbers_cache_v3.json"  # v3: {"reads": [...], "team": 0/1/-1, "mode": "orig"|"red"} per crop

MAX_CROPS = 6  # per track, spread over its lifetime, tallest box per bin
# (min track length s, crops) per pass, long tracks first so identities.json is useful early; cache makes each
# later pass pay only for its new tracks
PASSES = ((10.0, MAX_CROPS), (5.0, MAX_CROPS), (2.0, MAX_CROPS), (0.0, MAX_CROPS))
TORSO_TOP, TORSO_BOTTOM = 0.15, 0.60
TORSO_X0, TORSO_X1 = 0.05, 0.95  # middle 90 % of the width: neighbours out, a 55 still whole (80 % cut it to 5)
SWITCH_MIN_CROPS = 2  # crops of the final color after >= 1 crop of the other color = id switch
MAX_SIDE = 256  # px, longest side of the crop fed to the OCR
CPU_THREADS = int(__import__("os").environ.get("NUMBERS_THREADS", "2"))  # ORCH: keep the OCR at ~2 cores while LABEL trains
MIN_CONF = 0.4
MIN_READS = 2
STRONG_READ = 0.9  # a single read this sure with no competing read is enough (measured: 1.00 reads were all right)
WIN_SHARE = 0.6
MIN_BOX_H = 60  # px; smaller boxes have no legible number at 1080p
ALLOWLIST = "0123456789"

TILE_W, TILE_H = 120, 130  # preview tile


def load_tracks(path: Path = TRACKS) -> tuple[dict[int, list[dict]], str, int]:
    """(id -> [{frame, t, team, bbox}] sorted by frame, clip, frames read). Tolerates a partial last line."""
    tracks: dict[int, list[dict]] = defaultdict(list)
    n_frames = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # writer is mid-line
            n_frames += 1
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
    return dict(tracks), clip, n_frames


def majority_team(rows: list[dict]) -> tuple[int, float]:
    """(team, share): majority over the track's frames, share = its part of the 0/1 frames."""
    c = Counter(r["team"] for r in rows)
    known = [(n, t) for t, n in c.items() if t in (0, 1)]
    if not known:
        return -1, 0.0
    n, t = max(known)
    return t, round(n / sum(m for m, _ in known), 3)


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
    w = x2 - x1
    tx1, tx2 = int(x1 + TORSO_X0 * w), int(x1 + TORSO_X1 * w)
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
    cv2.setNumThreads(1)
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def crop_team(frame: np.ndarray, bbox) -> int:
    """Jersey color of this box in this frame (0/1/-1), TRACK's rule, no history."""
    feat = torso_color(frame, bbox)
    return -1 if feat is None else int(rule_label(feat))


def red_channel(img: np.ndarray) -> np.ndarray:
    """Red minus the other channels, stretched: a red number on a black jersey becomes
    white on black. Measured on dev60 track 805: the outlined red 9 reads as "0" (conf
    1.0) on the plain crop and as "9" (conf 1.0) on this image, 6 of 6 crops."""
    b, g, r = cv2.split(img)
    red = cv2.subtract(r, cv2.max(g, b))
    red = cv2.normalize(red, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)


def ocr_crop(reader, img: np.ndarray, team: int) -> tuple[list[tuple[str, float]], str]:
    """Reads for one crop by jersey color: blue (0) plain image; black/red (1) red
    channel, plain as fallback; unknown (-1) plain, red channel as fallback."""
    if team == 1:
        order = ("red", "orig")
    else:
        order = ("orig", "red") if team == -1 else ("orig",)
    for mode in order:
        reads = ocr_digits(reader, red_channel(img) if mode == "red" else img)
        if reads:
            return reads, mode
    return [], order[-1]


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


def pixel_team(per_crop: list[tuple]) -> tuple[int, float | None]:
    """(team by crop colors, switch_t). Team = the color of the LAST crops (the
    player the id ends on); switch_t = t of the first crop of that color when at
    least one crop of the other team precedes a run of >= SWITCH_MIN_CROPS crops
    of it (ByteTrack id jump); a single stray crop at the end is not a switch."""
    colored = [(r["t"], ct) for _, r, _, ct in per_crop if ct in (0, 1)]
    if not colored:
        return -1, None
    colored.sort()
    last = colored[-1][1]
    n_last = sum(1 for _, ct in colored if ct == last)
    n_other = len(colored) - n_last
    if n_last < SWITCH_MIN_CROPS and n_other >= SWITCH_MIN_CROPS:
        last = 1 - last  # one stray crop at the end: majority wins, no switch
        n_last, n_other = n_other, n_last
    if n_other >= 1 and n_last >= SWITCH_MIN_CROPS:
        idx = max(i for i, (_, ct) in enumerate(colored) if ct != last)
        if idx + 1 < len(colored):
            return last, colored[idx + 1][0]  # first crop of the final color run
    return last, None


def vote(reads: list[tuple[str, float]]) -> tuple[int | None, float, dict[str, float], dict[str, int]]:
    mass: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for text, conf in reads:
        num = str(int(text))  # "07" -> "7"
        mass[num] += conf
        count[num] += 1
    # ORCH rule: a one-digit read that is a substring of a two-digit read of the same
    # track (5 on a 55 shirt, 2 on a 23) is half a vote for the two-digit number
    for d in [n for n in mass if len(n) == 1]:
        hosts = [n for n in mass if len(n) == 2 and d in n]
        if hosts:
            host = max(hosts, key=mass.get)
            mass[host] += 0.5 * mass.pop(d)
            count[host] += count.pop(d)
    if not mass:
        return None, 0.0, {}, {}
    total = sum(mass.values())
    winner = max(mass, key=mass.get)
    share = mass[winner] / total
    if count[winner] >= MIN_READS and share >= WIN_SHARE:
        return int(winner), round(share, 3), dict(mass), dict(count)
    if len(mass) == 1 and count[winner] == 1 and mass[winner] >= STRONG_READ:
        return int(winner), round(share, 3), dict(mass), dict(count)
    return None, round(share, 3), dict(mass), dict(count)


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            return json.load(open(cache_path))
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


def track_seconds(rows: list[dict]) -> float:
    return rows[-1]["t"] - rows[0]["t"]


def run(tracks_path: Path = TRACKS, video: Path | None = None, preview: bool = True,
        min_track_s: float = 0.0, max_crops: int = MAX_CROPS) -> dict:
    """OCR + vote. Tracks shorter than min_track_s get no crops (they stay in the
    output without a number) unless the cache already holds reads for them."""
    t0 = time.time()
    tracks, clip, n_frames = load_tracks(tracks_path)
    video = video or (ROOT / clip)
    out_dir = tracks_path.parent
    cache_path, reads_out, preview_out = out_dir / CACHE_NAME, out_dir / READS_NAME, out_dir / PREVIEW_NAME
    cache = load_cache(cache_path)

    # which crops do we need
    plan: dict[int, list[dict]] = {}
    for tid, rows in tracks.items():
        samples = pick_samples(rows, max_crops)
        if track_seconds(rows) < min_track_s:
            samples = [r for r in samples if crop_key(clip, r["frame"], r["bbox"]) in cache]
        plan[tid] = samples
    wanted_frames: set[int] = set()
    todo: dict[str, list[tuple[int, dict]]] = defaultdict(list)  # key -> [(tid, row)]
    for tid, rows in plan.items():
        for r in rows:
            k = crop_key(clip, r["frame"], r["bbox"])
            if k not in cache or preview:
                wanted_frames.add(r["frame"])  # preview needs the pixels even when cached
            todo[k].append((tid, r))
    need_ocr = [k for k in todo if k not in cache]
    log.info("tracks %d (>= %.1f s: %d), crops %d, cached %d, to OCR %d, frames to decode %d",
             len(tracks), min_track_s, sum(1 for r in tracks.values() if track_seconds(r) >= min_track_s),
             len(todo), len(todo) - len(need_ocr), len(need_ocr), len(wanted_frames))

    reader = get_reader() if need_ocr else None
    crops: dict[str, np.ndarray] = {}
    by_frame: dict[int, list[str]] = defaultdict(list)
    for k, users in todo.items():
        by_frame[users[0][1]["frame"]].append(k)
    n_done = 0
    if wanted_frames:
        # decode once, keep the crops; OCR afterwards ordered by track length so a
        # killed run has read the long (useful) tracks first
        color: dict[str, int] = {}
        for fidx, frame in read_frames(video, wanted_frames):
            for k in by_frame[fidx]:
                bbox = todo[k][0][1]["bbox"]
                crop = torso_crop(frame, bbox)
                if crop is None:
                    cache[k] = {"reads": [], "team": -1}
                else:
                    crops[k] = crop
                    color[k] = crop_team(frame, bbox)
        order = sorted((k for k in need_ocr if k in crops), key=lambda k: -len(tracks[todo[k][0][0]]))
        for k in order:
            reads, mode = ocr_crop(reader, crops[k], color[k])
            cache[k] = {"reads": reads, "team": color[k], "mode": mode}
            n_done += 1
            if n_done % 100 == 0:
                log.info("ocr %d/%d  %.1f s", n_done, len(order), time.time() - t0)
                json.dump(cache, open(cache_path, "w"))
        json.dump(cache, open(cache_path, "w"))

    # vote
    result_tracks: dict[str, dict] = {}
    tiles: list[tuple[np.ndarray, str, str]] = []
    assigned = 0
    for tid in sorted(tracks):
        rows = tracks[tid]
        per_crop = []
        for r in plan[tid]:
            k = crop_key(clip, r["frame"], r["bbox"])
            entry = cache.get(k, {"reads": [], "team": -1})
            per_crop.append((k, r, [tuple(x) for x in entry["reads"]], entry["team"]))
        team, team_share = majority_team(rows)
        team_px, switch_t = pixel_team(per_crop)
        if team_px in (0, 1):
            team = team_px  # the color of the crops beats the id's sticky history vote
        reads = [x for _, r, rd, ct in per_crop for x in rd
                 if ct == team or team not in (0, 1) or (switch_t is not None and r["t"] >= switch_t and ct == -1)]
        number, share, mass, count = vote(reads)
        if number is not None:
            assigned += 1
        result_tracks[str(tid)] = {
            "team": team,
            "team_share": team_share,
            "team_px": team_px,
            "switch_t": switch_t,
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
            if not per_crop:
                continue  # fragment without crops: no row on the sheet
            for i, (k, r, rd, ct) in enumerate(per_crop[:max_crops]):
                mode = cache.get(k, {}).get("mode", "orig")
                foot = ("AB"[ct] if ct in (0, 1) else "?") + ("r " if mode == "red" else " ") + \
                    (" ".join(f"{t}:{c:.2f}" for t, c in rd[:2]) if rd else "-")
                tiles.append((crops.get(k), head if i == 0 else f"f{r['frame']}", foot))
            for _ in range(len(per_crop), max_crops):
                tiles.append((None, "", ""))

    st = tracks_path.stat()
    out = {"clip": clip, "video": str(video.relative_to(ROOT)) if video.is_relative_to(ROOT) else str(video),
           "tracks_path": str(tracks_path.relative_to(ROOT)) if tracks_path.is_relative_to(ROOT) else str(tracks_path),
           "tracks_mtime": round(st.st_mtime, 3), "tracks_frames": n_frames, "tracks_ids": len(tracks),
           "params": {"max_crops": max_crops, "min_track_s": min_track_s, "min_conf": MIN_CONF,
                      "min_reads": MIN_READS, "win_share": WIN_SHARE},
           "n_tracks": len(tracks), "n_assigned": assigned, "tracks": result_tracks}
    tmp = reads_out.with_suffix(".tmp")
    json.dump(out, open(tmp, "w"), indent=1)
    tmp.replace(reads_out)
    if preview:
        build_preview(tiles, preview_out, cols=max_crops)
    log.info("numbers: %d/%d tracks got a number (%.0f%%)  %.1f s  -> %s",
             assigned, len(tracks), 100 * assigned / max(1, len(tracks)), time.time() - t0, reads_out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, default=TRACKS)
    ap.add_argument("--video", type=Path, default=None, help="default: clip from out/tracks_meta.json")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--min-s", type=float, default=0.0, help="only OCR tracks at least this long (s)")
    ap.add_argument("--max-crops", type=int, default=MAX_CROPS)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    run(a.tracks, a.video, preview=not a.no_preview, min_track_s=a.min_s, max_crops=a.max_crops)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
