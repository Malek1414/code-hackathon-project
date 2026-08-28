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
        prev_grey = grey
        if len(frames) % 500 == 0:
            el = time.time() - t0
            log(f"  {index}/{end} frames, {el:.0f} s, {len(frames) / el:.1f} fps, {failed} failed, {reanchors} re-anchors")
    cap.release()
    return np.array(frames, np.int64), np.array(mats, np.float64), {"failed": failed, "reanchors": reanchors, "fps_processed": len(frames) / max(time.time() - t0, 1e-6)}


def per_frame_homographies(frames: np.ndarray, C: np.ndarray, keyframes: dict[int, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Court->pixel H for every frame from the camera chain and the keyframe truths."""
    keys = sorted(keyframes)
    pos = {int(f): i for i, f in enumerate(frames)}
    for k in keys:
        if k not in pos:
            raise SystemExit(f"Keyframe {k} wurde nicht verarbeitet (stride/end prüfen).")
    inv = np.linalg.inv

    def carried(k: int, i: int) -> np.ndarray:
        # T(k -> f) = C_f @ inv(C_k); court->px at f = T @ H_m2px(k)
        return _normalise(C[i] @ inv(C[pos[k]]) @ keyframes[k])

    out = np.empty_like(C)
    drift_px = {}
    for i, f in enumerate(frames):
        f = int(f)
        if f <= keys[0]:
            out[i] = carried(keys[0], i)
        elif f >= keys[-1]:
            out[i] = carried(keys[-1], i)
        else:
            a = max(k for k in keys if k <= f)
            b = min(k for k in keys if k >= f)
            s = (f - a) / (b - a) if b > a else 0.0
            out[i] = interpolate_m_to_px(carried(a, i), carried(b, i), s)
    # how far the forward chain is off when it reaches the next keyframe: the honest drift number
    corners = np.float64([[0, 0], [28, 0], [28, 15], [0, 15]])
    for a, b in zip(keys[:-1], keys[1:]):
        est = carried(a, pos[b])
        from vision.court.homography import apply_h
        d = np.linalg.norm(apply_h(est, corners) - apply_h(keyframes[b], corners), axis=1)
        drift_px[f"{a}->{b}"] = round(float(np.nanmean(d)), 1)
    return out, drift_px


def write_preview(clip: Path, frames: np.ndarray, H_m_to_px: np.ndarray, out: Path, every: int, scale: float = 0.5) -> None:
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    S = np.diag([scale, scale, 1.0])
    with FfmpegWriter(out, w, h, fps / every) as writer:
        for i, f in enumerate(frames):
            if i % every:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            draw_court(frame, S @ H_m_to_px[i], FIBA, thickness=2)
            cv2.putText(frame, f"frame {int(f)}  t={f / fps:5.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    cap.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--calib", type=Path, default=ROOT / "out" / "court_calib.json")
    ap.add_argument("--tracks", type=Path, default=ROOT / "out" / "tracks.jsonl", help="Spielerboxen als Maske (optional)")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "court_H.npz")
    ap.add_argument("--scale", type=float, default=0.5, help="Tracking-Auflösung relativ zum Frame")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--end", type=int, default=None, help="nur bis zu diesem Frame")
    ap.add_argument("--reanchor-every", type=int, default=60)
    ap.add_argument("--preview", action="store_true", help="out/court_propagate_preview.mp4 mit Linien-Overlay schreiben")
    ap.add_argument("--preview-every", type=int, default=5)
    args = ap.parse_args(argv)

    data = json.loads(args.calib.read_text())
    keyframes = {int(k): np.array(v["H_m_to_px"], np.float64) for k, v in (data.get("frames") or {}).items()}
    if not keyframes:
        keyframes = {int(data["frame"]): np.linalg.inv(np.array(data["H_px_to_m"], np.float64))}
    boxes = load_boxes(args.tracks)
    print(f"{len(keyframes)} Keyframes {sorted(keyframes)}, Maske aus {len(boxes)} Frames mit Boxen"
          + ("" if boxes else " (keine tracks.jsonl, ohne Spielermaske)"))

    frames, C, stats = chain_camera(args.clip, scale=args.scale, boxes=boxes, keyframes=keyframes, end=args.end,
                                    stride=args.stride, reanchor_every=args.reanchor_every)
    H_m_to_px, drift = per_frame_homographies(frames, C, keyframes)
    H_px_to_m = np.linalg.inv(H_m_to_px)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, frames=frames, H_m_to_px=H_m_to_px, H_px_to_m=H_px_to_m)

    data["per_frame"] = rel(args.out)
    data["propagation"] = {"frames": int(len(frames)), "stride": args.stride, "scale": args.scale,
                           "failed_transitions": stats["failed"], "reanchors": stats["reanchors"],
                           "chain_drift_px_at_next_keyframe": drift}
    args.calib.write_text(json.dumps(data, indent=1))
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
