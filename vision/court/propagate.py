"""Follow the panning camera: one homography per frame between hand-clicked keyframes.

    .venv/bin/python vision/court/propagate.py data/clips/dev60.mp4 [--tracks out/tracks.jsonl] [--preview]

Camera-following code (frame_to_frame, player_mask, re-anchoring idea) copied from
~/Desktop/APP/courtside/engine/court/homography.py (Courtside, Sami Magdouli).

Method: sparse optical flow between consecutive frames gives the pixel->pixel
homography of the camera motion; chaining it from a hand keyframe carries that
keyframe's court mapping forward. Chains drift, so every segment between two
keyframes is computed forward from the first and backward from the second and
blended by time: exact at both keyframes, smooth in between. Players are masked
out of the feature selection (from tracks.jsonl when TRACK has run) because they
move independently of the camera. Result: out/court_H.npz (frames, H_px_to_m,
H_m_to_px) and a "per_frame" pointer in out/court_calib.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision.court.calibrate import draw_court, rel  # noqa: E402
from vision.court.geometry import FIBA  # noqa: E402
from vision.court.project import interpolate_m_to_px, iter_tracks  # noqa: E402
from vision.court.video import FfmpegWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def frame_to_frame(previous_grey: np.ndarray, current_grey: np.ndarray, *, mask: np.ndarray | None = None,
                   max_features: int = 1200) -> tuple[np.ndarray | None, int]:
    """Homography mapping `previous` pixels onto `current` pixels (Courtside)."""
    corners = cv2.goodFeaturesToTrack(previous_grey, maxCorners=max_features, qualityLevel=0.01,
                                      minDistance=12, mask=mask, blockSize=7)
    if corners is None or len(corners) < 12:
        return None, 0
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_grey, current_grey, corners, None, winSize=(21, 21),
                                                maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if moved is None:
        return None, 0
    keep = status.ravel() == 1
    src, dst = corners[keep].reshape(-1, 2), moved[keep].reshape(-1, 2)
    if len(src) < 12:
        return None, len(src)
    matrix, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=2.0)
    if matrix is None:
        return None, len(src)
    return matrix, int(inlier_mask.sum()) if inlier_mask is not None else len(src)


def player_mask(shape: tuple[int, int], boxes: np.ndarray, pad: int = 8) -> np.ndarray:
    """White where features may be taken, black over players (Courtside)."""
    mask = np.full(shape[:2], 255, np.uint8)
    for box in np.asarray(boxes).reshape(-1, 4):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(mask, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 0, thickness=-1)
    return mask


def load_boxes(tracks_path: Path | None) -> dict[int, np.ndarray]:
    """frame -> [N, 4] boxes of everything that moves on its own (players, referees, ball)."""
    if tracks_path is None or not Path(tracks_path).exists():
        return {}
    out: dict[int, np.ndarray] = {}
    for rec in iter_tracks(tracks_path):
        boxes = [p["bbox"] for p in rec.get("players") or []]
        boxes += [r["bbox"] for r in rec.get("referees") or []]
        if rec.get("ball"):
            boxes.append(rec["ball"]["bbox"])
        if boxes:
            out[int(rec["frame"])] = np.array(boxes, np.float64)
    return out


def _normalise(H: np.ndarray) -> np.ndarray:
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def chain_camera(clip: Path, *, scale: float, boxes: dict[int, np.ndarray], keyframes: dict[int, np.ndarray],
                 end: int | None, stride: int, reanchor_every: int, log=print) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cumulative camera motion C_f: pixels of frame 0 -> pixels of frame f, for every frame.

    Runs at `scale` resolution for speed; matrices are converted back to full
    resolution. At every hand keyframe the chain is re-based (the truth is known
    there), and every `reanchor_every` frames a direct match against the last
    keyframe image resets accumulated error while the view still overlaps.
    """
    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if end is None else min(end, total)
    S = np.diag([scale, scale, 1.0])
    S_inv = np.linalg.inv(S)

    frames, mats = [], []
    thumbs: dict[int, np.ndarray] = {}
    thumb_w, thumb_h = 192, 108
    C = np.eye(3)
    prev_grey = None
    prev_boxes: np.ndarray | None = None
    anchor_grey, anchor_C = None, None
    failed, reanchors = 0, 0
    t0 = time.time()
    index = -1
    while True:
        ok = cap.grab()
        index += 1
        if not ok or index >= end:
            break
        if index % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else frame
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if index in boxes:
            prev_boxes = boxes[index]
        if prev_grey is not None:
            mask = player_mask(grey.shape, prev_boxes * scale) if prev_boxes is not None and len(prev_boxes) else None
            step, _ = frame_to_frame(prev_grey, grey, mask=mask)
            if step is None:
                failed += 1
                anchor_grey = None  # the chain broke here, an older anchor could re-attach to the wrong shot
            else:
                C = _normalise((S_inv @ step @ S) @ C)
            if anchor_grey is not None and reanchor_every and len(frames) % reanchor_every == 0:
                direct, inliers = frame_to_frame(anchor_grey, grey, mask=mask)
                if direct is not None and inliers >= 40:
                    C = _normalise((S_inv @ direct @ S) @ anchor_C)
                    reanchors += 1
        if index in keyframes:
            anchor_grey, anchor_C = grey, C.copy()
        frames.append(index)
        mats.append(C.copy())
        if index % THUMB_EVERY == 0:
            thumbs[index] = cv2.resize(grey, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        prev_grey = grey
        if len(frames) % 500 == 0:
            el = time.time() - t0
            log(f"  {index}/{end} frames, {el:.0f} s, {len(frames) / el:.1f} fps, {failed} failed, {reanchors} re-anchors")
    cap.release()
    frames_arr, mats_arr = np.array(frames, np.int64), np.array(mats, np.float64)
    if prev_grey is None:
        raise SystemExit(f"kein Frame aus {clip} lesbar.")
    cuts, aligned = detect_cuts(frames_arr, mats_arr, thumbs, full_size=(prev_grey.shape[1] / scale, prev_grey.shape[0] / scale))
    return frames_arr, mats_arr, {"failed": failed, "reanchors": reanchors, "cuts": cuts, "aligned_diff": aligned,
                                  "fps_processed": len(frames) / max(time.time() - t0, 1e-6)}


THUMB_EVERY = 5
CUT_LAG = 50  # frames, 1 s at 50 fps
CUT_THRESHOLD = 25.0
"""Mean grey difference (0..255) between a frame and the frame one second earlier
AFTER aligning the earlier one with the tracked camera motion. A pan aligns
away, moving players cost a few units, a cut or a dissolve to another view
does not align at all and scores far higher. Measured on dev60: game segments
sit at 10 to 20, jump cuts at 28 to 34, the dissolve at 47, hard cuts at 60 to 88;
the pre-game close-ups sit at 25 to 47 throughout and become one junk segment."""


def detect_cuts(frames: np.ndarray, C: np.ndarray, thumbs: dict[int, np.ndarray], full_size: tuple[float, float]) -> tuple[list[int], dict[int, float]]:
    """Frames where the content changed in a way camera motion cannot explain."""
    if not thumbs:
        return [], {}
    W, H = full_size
    sample = next(iter(thumbs.values()))
    th, tw = sample.shape[:2]
    S = np.diag([tw / W, th / H, 1.0])
    S_inv = np.linalg.inv(S)
    pos = {int(f): i for i, f in enumerate(frames)}
    aligned: dict[int, float] = {}
    for t, thumb in thumbs.items():
        t0 = t - CUT_LAG
        if t0 not in thumbs or t0 not in pos or t not in pos:
            continue
        T = C[pos[t]] @ np.linalg.inv(C[pos[t0]])  # px(t0) -> px(t), full res
        Ts = S @ T @ S_inv
        if not np.all(np.isfinite(Ts)):
            aligned[t] = 255.0
            continue
        warped = cv2.warpPerspective(thumbs[t0], Ts.astype(np.float64), (tw, th), flags=cv2.INTER_LINEAR, borderValue=0)
        valid = cv2.warpPerspective(np.full_like(thumbs[t0], 255), Ts.astype(np.float64), (tw, th), flags=cv2.INTER_NEAREST, borderValue=0) > 0
        if valid.sum() < 0.3 * valid.size:
            continue  # a fast pan the chain followed: nothing to compare, not a cut
        aligned[t] = float(np.abs(warped.astype(np.float32) - thumb.astype(np.float32))[valid].mean())
    return cuts_from_signal(aligned, first_frame=int(frames[0])), aligned


def cuts_from_signal(aligned: dict[int, float], first_frame: int = 0, threshold: float = CUT_THRESHOLD,
                     min_run: int = 2, min_gap: int = 2 * CUT_LAG) -> list[int]:
    """One cut per run of at least `min_run` consecutive high samples, placed at the
    start of the run minus half the lag so it lands inside a dissolve, not after it."""
    cuts: list[int] = []
    ts = [t for t in sorted(aligned) if aligned[t] < 254]  # 255 = no overlap marker from older caches
    i = 0
    while i < len(ts):
        if aligned[ts[i]] > threshold:
            j = i
            while j < len(ts) and aligned[ts[j]] > threshold:
                j += 1
            if j - i >= min_run:
                # first high sample t means the change happened in (t - THUMB_EVERY, t]
                cut = max(first_frame, ts[i] - THUMB_EVERY // 2)
                if not cuts or cut - cuts[-1] >= min_gap:
                    cuts.append(cut)
            i = j
        else:
            i += 1
    return cuts


def per_frame_homographies(frames: np.ndarray, C: np.ndarray, keyframes: dict[int, np.ndarray],
                           cuts: list[int] | None = None) -> tuple[np.ndarray, dict, list[dict]]:
    """Court->pixel H for every frame from the camera chain and the keyframe truths.

    The clip is split at cuts; keyframes only ever carry within their own
    segment, never across a cut. A segment without a keyframe gets NaN
    matrices, which every consumer treats as "position unknown"."""
    from vision.court.homography import apply_h

    keys = sorted(keyframes)
    pos = {int(f): i for i, f in enumerate(frames)}
    snapped = {}
    for k in keys:
        if k not in pos:
            i = int(np.clip(np.searchsorted(frames, k), 0, len(frames) - 1))
            print(f"  Hinweis: Keyframe {k} liegt nicht in der Kette, auf Frame {int(frames[i])} gesetzt.")
            snapped[int(frames[i])] = keyframes[k]
        else:
            snapped[k] = keyframes[k]
    keyframes, keys = snapped, sorted(snapped)
    inv = np.linalg.inv
    bounds = sorted(set([int(frames[0])] + [c for c in (cuts or []) if frames[0] < c <= frames[-1]] + [int(frames[-1]) + 1]))
    segments = list(zip(bounds[:-1], bounds[1:]))

    def carried(k: int, i: int) -> np.ndarray:
        # T(k -> f) = C_f @ inv(C_k); court->px at f = T @ H_m2px(k)
        return _normalise(C[i] @ inv(C[pos[k]]) @ keyframes[k])

    out = np.full_like(C, np.nan)
    drift_px: dict[str, float] = {}
    report: list[dict] = []
    corners = np.float64([[0, 0], [28, 0], [28, 15], [0, 15]])
    for seg_start, seg_end in segments:
        seg_keys = [k for k in keys if seg_start <= k < seg_end]
        report.append({"start": seg_start, "end": seg_end - 1, "keyframes": seg_keys})
        if not seg_keys:
            continue
        for i, f in enumerate(frames):
            f = int(f)
            if not (seg_start <= f < seg_end):
                continue
            if f <= seg_keys[0]:
                out[i] = carried(seg_keys[0], i)
            elif f >= seg_keys[-1]:
                out[i] = carried(seg_keys[-1], i)
            else:
                a = max(k for k in seg_keys if k <= f)
                b = min(k for k in seg_keys if k >= f)
                s = (f - a) / (b - a) if b > a else 0.0
                out[i] = interpolate_m_to_px(carried(a, i), carried(b, i), s)
        # how far the forward chain is off when it reaches the next keyframe: the honest drift number
        for a, b in zip(seg_keys[:-1], seg_keys[1:]):
            est = carried(a, pos[b])
            d = np.linalg.norm(apply_h(est, corners) - apply_h(keyframes[b], corners), axis=1)
            mean = float(np.nanmean(d)) if np.isfinite(d).any() else None
            drift_px[f"{a}->{b}"] = round(mean, 1) if mean is not None else None
    return out, drift_px, report


# --- auto anchors: tie keyframe-less segments to a hand keyframe by direct feature matching ----

AUTO_MIN_INLIERS = 80
AUTO_MIN_RATIO = 0.6
"""Measured on dev60 at half resolution: the same wide camera across a dissolve
gives 240+ SIFT inliers at a 0.77 ratio, the scoreboard 41 at 0.42, a close-up
28 at 0.45. The hall wall, doors and benches are what match, not the players."""


def _sift_features(grey: np.ndarray, mask: np.ndarray | None):
    sift = cv2.SIFT_create(nfeatures=3000)
    return sift.detectAndCompute(grey, mask)


def _match_h(kp_a, des_a, kp_b, des_b) -> tuple[np.ndarray | None, int, float]:
    """Homography a -> b from SIFT matches, with inlier count and ratio."""
    if des_a is None or des_b is None or len(des_a) < 8 or len(des_b) < 8:
        return None, 0, 0.0
    matches = cv2.BFMatcher().knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, len(good), 0.0
    src = np.float32([kp_a[g.queryIdx].pt for g in good])
    dst = np.float32([kp_b[g.trainIdx].pt for g in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or mask is None:
        return None, len(good), 0.0
    inl = int(mask.sum())
    return H, inl, inl / len(good)


def auto_anchors(clip: Path, frames: np.ndarray, keyframes: dict[int, np.ndarray], cuts: list[int],
                 boxes: dict[int, np.ndarray], scale: float = 0.5, log=print) -> tuple[dict[int, np.ndarray], list[dict]]:
    """One synthetic keyframe per segment that has no hand keyframe, found by matching a
    frame of that segment directly against every hand keyframe image (same tripod, same
    hall, so the static background relates any two wide shots by a homography)."""
    cap = cv2.VideoCapture(str(clip))
    S = np.diag([scale, scale, 1.0])
    S_inv = np.linalg.inv(S)

    def read_small(index: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, img = cap.read()
        if not ok:
            return None, None
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        near = min(boxes, key=lambda f: abs(f - index)) if boxes else None
        mask = player_mask(grey.shape, boxes[near] * scale, pad=12) if near is not None and abs(near - index) <= 25 else None
        return grey, mask

    refs = {}
    for k in sorted(keyframes):
        grey, mask = read_small(k)
        if grey is not None:
            refs[k] = _sift_features(grey, mask)
    bounds = sorted(set([int(frames[0])] + [c for c in cuts if frames[0] < c <= frames[-1]] + [int(frames[-1]) + 1]))
    anchors: dict[int, np.ndarray] = {}
    report: list[dict] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if any(a <= k < b for k in keyframes):
            continue
        length = b - a
        cands = sorted({a + min(15, length // 3), a + length // 2, b - 1 - min(15, length // 3)})
        cands = [c for c in cands if a <= c < b]
        best = None
        for cand in cands:
            grey, mask = read_small(cand)
            if grey is None:
                continue
            kp, des = _sift_features(grey, mask)
            for k, (kp_r, des_r) in refs.items():
                T, inl, ratio = _match_h(kp_r, des_r, kp, des)
                if T is None or inl < AUTO_MIN_INLIERS or ratio < AUTO_MIN_RATIO:
                    continue
                if best is None or inl > best[2]:
                    best = (cand, k, inl, ratio, S_inv @ T @ S)
        if best is None:
            report.append({"start": a, "end": b - 1, "auto": None})
            continue
        cand, k, inl, ratio, T_full = best
        anchors[cand] = _normalise(T_full @ keyframes[k])
        report.append({"start": a, "end": b - 1, "auto": cand, "from_keyframe": k, "inliers": inl, "ratio": round(ratio, 2)})
        log(f"  Segment {a}..{b - 1}: automatisch verankert über Frame {cand} an Keyframe {k} ({inl} Inlier, {ratio:.2f})")
    cap.release()
    return anchors, report


def write_cuts(clip: Path, frames: np.ndarray, cuts: list[int], threshold: float, out_dir: Path) -> Path:
    """out/cuts_<clip>.json for TRACK/STATS: reset ids and ball history at these frames."""
    fps = cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FPS) or 50.0
    bounds = [int(frames[0])] + [int(c) for c in cuts] + [int(frames[-1]) + 1]
    data = {
        "clip": rel(clip), "fps": fps, "first_frame": int(frames[0]), "last_frame": int(frames[-1]),
        "cuts": [int(c) for c in cuts], "cut_times_s": [round(int(c) / fps, 2) for c in cuts],
        "segments": [{"start": a, "end": b - 1, "start_s": round(a / fps, 2), "end_s": round((b - 1) / fps, 2)}
                     for a, b in zip(bounds[:-1], bounds[1:])],
        "method": f"frame vs frame {CUT_LAG} frames earlier, aligned by tracked camera motion, mean grey diff > {threshold}; "
                  "a cut frame is the first frame that no longer belongs to the previous camera shot (dissolves: mid-fade)",
    }
    path = out_dir / f"cuts_{clip.stem}.json"
    path.write_text(json.dumps(data, indent=1))
    print(f"Schnittliste: {rel(path)}")
    return path


def write_preview(clip: Path, frames: np.ndarray, H_m_to_px: np.ndarray, out: Path, every: int, scale: float = 0.5) -> None:
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    S = np.diag([scale, scale, 1.0])
    pos = {int(f): i for i, f in enumerate(frames)}
    index = -1
    with FfmpegWriter(out, w, h, fps / every) as writer:
        while True:  # sequential grab is 10x faster than seeking every Nth frame in h264
            if not cap.grab():
                break
            index += 1
            if index not in pos or pos[index] % every:
                continue
            i = pos[index]
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            if np.isfinite(H_m_to_px[i]).all():
                draw_court(frame, S @ H_m_to_px[i], FIBA, thickness=2)
            else:
                cv2.putText(frame, "unkalibriert", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 240), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frame {index}  t={index / fps:5.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    cap.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--calib", type=Path, default=None, help="Standard: out/court_calib_<clip>.json, sonst out/court_calib.json")
    ap.add_argument("--tracks", type=Path, default=ROOT / "out" / "tracks.jsonl", help="Spielerboxen als Maske (optional)")
    ap.add_argument("--out", type=Path, default=None, help="Standard: out/court_H_<clip>.npz")
    ap.add_argument("--scale", type=float, default=0.5, help="Tracking-Auflösung relativ zum Frame")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--end", type=int, default=None, help="nur bis zu diesem Frame")
    ap.add_argument("--reanchor-every", type=int, default=0,
                    help="direkte Neuverankerung alle N Frames (0 = aus; bei geschnittenem Material aus lassen)")
    ap.add_argument("--no-cache", action="store_true", help="Kamerakette neu rechnen")
    ap.add_argument("--cut-threshold", type=float, default=CUT_THRESHOLD, help="Schnitt-Schwelle auf der ausgerichteten Differenz")
    ap.add_argument("--no-auto", action="store_true", help="keine automatische Verankerung keyframeloser Segmente")
    ap.add_argument("--chain-only", action="store_true", help="nur die Kamerakette cachen, keine Keyframes nötig")
    ap.add_argument("--preview", action="store_true", help="out/court_propagate_preview.mp4 mit Linien-Overlay schreiben")
    ap.add_argument("--preview-every", type=int, default=5)
    args = ap.parse_args(argv)

    if args.out is None:
        args.out = ROOT / "out" / f"court_H_{args.clip.stem}.npz"
    if args.calib is None:
        per_clip = ROOT / "out" / f"court_calib_{args.clip.stem}.json"
        args.calib = per_clip if per_clip.exists() else ROOT / "out" / "court_calib.json"
    data = json.loads(args.calib.read_text()) if args.calib.exists() else {}
    if data and data.get("clip") not in (None, rel(args.clip)):
        raise SystemExit(f"{rel(args.calib)} gehört zu {data.get('clip')}, nicht zu {rel(args.clip)}.")
    keyframes = {int(k): np.array(v["H_m_to_px"], np.float64) for k, v in (data.get("frames") or {}).items()}
    if not keyframes and data:
        keyframes = {int(data["frame"]): np.linalg.inv(np.array(data["H_px_to_m"], np.float64))}
    if not keyframes and not args.chain_only:
        raise SystemExit(f"keine Keyframes in {args.calib}, erst calibrate.py laufen lassen (oder --chain-only).")
    boxes = load_boxes(args.tracks)
    print(f"{len(keyframes)} Keyframes {sorted(keyframes)}, Maske aus {len(boxes)} Frames mit Boxen"
          + ("" if boxes else " (keine tracks.jsonl, ohne Spielermaske)"))

    # The camera chain does not depend on the keyframes (only re-anchoring does, and
    # that needs their images, not their H), so it is cached per clip and reused
    # when Sami refines his clicks. --no-cache forces a fresh run.
    cache = args.out.with_name(f"court_chain_{args.clip.stem}_s{args.stride}_x{args.scale:g}.npz")
    if cache.exists() and not args.no_cache:
        npz = np.load(cache)
        frames, C = npz["frames"], npz["C"]
        aligned = dict(zip(npz["aligned_t"].tolist(), npz["aligned_v"].tolist())) if "aligned_t" in npz else {}
        stats = {"failed": int(npz["failed"]), "reanchors": 0, "cached": str(cache), "aligned_diff": aligned,
                 "cuts": cuts_from_signal(aligned, int(frames[0]), args.cut_threshold) if aligned else [int(c) for c in npz["cuts"]]}
        print(f"Kamerakette aus Cache: {rel(cache)} ({len(frames)} Frames)")
    else:
        frames, C, stats = chain_camera(args.clip, scale=args.scale, boxes=boxes, keyframes=keyframes, end=args.end,
                                        stride=args.stride, reanchor_every=args.reanchor_every)
        if args.end is None:
            al = stats.get("aligned_diff") or {}
            np.savez_compressed(cache, frames=frames, C=C, failed=stats["failed"], cuts=np.array(stats["cuts"], np.int64),
                                aligned_t=np.array(sorted(al), np.int64), aligned_v=np.array([al[k] for k in sorted(al)], np.float64))
        vals = np.array(list(stats["aligned_diff"].values())) if stats.get("aligned_diff") else np.zeros(1)
        print(f"Schnitt-Signal (ausgerichtete Differenz) Median {np.median(vals):.1f}, 90% {np.percentile(vals, 90):.1f}, max {vals.max():.1f}")
    cuts = cuts_from_signal(stats["aligned_diff"], int(frames[0]), args.cut_threshold) if stats.get("aligned_diff") else stats.get("cuts", [])
    print(f"{len(cuts)} Schnitte/Überblendungen erkannt (Schwelle {args.cut_threshold}): {cuts}")
    write_cuts(args.clip, frames, cuts, args.cut_threshold, args.out.parent)
    if args.chain_only:
        print("nur Kamerakette berechnet.")
        return 0
    anchors, auto_report = ({}, []) if args.no_auto else auto_anchors(args.clip, frames, keyframes, cuts, boxes, scale=args.scale)
    if not args.no_auto:
        missing = [r for r in auto_report if r["auto"] is None]
        print(f"{len(anchors)} Segmente automatisch verankert, {len(missing)} ohne Treffer (Nahaufnahmen?)")
    H_m_to_px, drift, segments = per_frame_homographies(frames, C, {**keyframes, **anchors}, cuts)
    auto_frames = set(anchors)
    for seg in segments:
        seg["auto"] = [k for k in seg["keyframes"] if k in auto_frames]
        seg["keyframes"] = [k for k in seg["keyframes"] if k not in auto_frames]
    H_px_to_m = np.full_like(H_m_to_px, np.nan)
    ok = np.isfinite(H_m_to_px).all(axis=(1, 2))
    ok[ok] = np.abs(np.linalg.det(H_m_to_px[ok])) > 1e-12
    H_px_to_m[ok] = np.linalg.inv(H_m_to_px[ok])
    H_m_to_px[~ok] = np.nan
    for seg in segments:
        state = (f"Keyframes {seg['keyframes']}" if seg["keyframes"] else
                 f"automatisch verankert {seg['auto']}" if seg.get("auto") else "OHNE Keyframe, bleibt unkalibriert")
        fps = float(data.get("fps") or 50.0)
        print(f"  Segment {seg['start']}..{seg['end']} ({seg['start'] / fps:.0f}s bis {seg['end'] / fps:.0f}s): {state}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, frames=frames, H_m_to_px=H_m_to_px, H_px_to_m=H_px_to_m)

    data["per_frame"] = rel(args.out)
    data["propagation"] = {"frames": int(len(frames)), "stride": args.stride, "scale": args.scale,
                           "failed_transitions": stats["failed"], "reanchors": stats["reanchors"],
                           "cuts": cuts, "segments": segments, "calibrated_frames": int(ok.sum()),
                           "auto_anchors": auto_report,
                           "chain_drift_px_at_next_keyframe": drift}
    args.calib.write_text(json.dumps(data, indent=1, allow_nan=False))
    contract = args.calib.with_name("court_calib.json")
    if args.calib != contract:
        contract.write_text(json.dumps(data, indent=1))
    print(f"gespeichert: {rel(args.out)} ({len(frames)} Frames), Drift zum nächsten Keyframe: {drift or 'nur ein Keyframe'}")
    if stats["failed"]:
        print(f"  Hinweis: {stats['failed']} Übergänge ohne Bewegungsschätzung (Kamera dort als still angenommen).")

    if args.preview:
        prev = args.out.with_name("court_propagate_preview.mp4")
        write_preview(args.clip, frames, H_m_to_px, prev, args.preview_every)
        print(f"Vorschau: {rel(prev)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
