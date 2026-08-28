"""Projection helper: pixels -> court metres for any frame of the clip.

This is the one function everybody (minimap, dashboard, STATS distance_m) calls:

    from vision.court.project import load_calibration
    cal = load_calibration("out/court_calib.json")
    xy_m = cal.project(frame, [[x, y], ...])      # [N, 2] metres, NaN if unmappable
    dist = cal.player_distances("out/tracks.jsonl")  # {player_id: metres}

Frame handling, best available first:
  1. per-frame matrices from propagate.py (out/court_H.npz),
  2. hand keyframes in court_calib.json "frames", blended by time,
  3. the single top-level H.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision.court.geometry import FIBA, SurfaceSpec
from vision.court.homography import apply_h

ROOT = Path(__file__).resolve().parents[2]

MAX_SPEED_M_S = 11.0
"""Faster than any human on a court. Steps above it are tracking jumps
(id switch, re-detection) and must not count as distance run."""


def interpolate_m_to_px(H_a: np.ndarray, H_b: np.ndarray, s: float, spec: SurfaceSpec = FIBA) -> np.ndarray:
    """Court->pixel homography between two others, s in [0, 1].

    Homographies are not blended entry by entry; the court corners are projected
    with both, blended in pixel space and a fresh H fitted through them, which
    is what a camera panning smoothly between the two views actually does."""
    import cv2

    if s <= 0:
        return H_a
    if s >= 1:
        return H_b
    corners = np.float64([[0, 0], [spec.length_m, 0], [spec.length_m, spec.width_m], [0, spec.width_m]])
    pa, pb = apply_h(H_a, corners), apply_h(H_b, corners)
    px = (1 - s) * pa + s * pb
    return cv2.getPerspectiveTransform(corners.astype(np.float32), px.astype(np.float32)).astype(np.float64)


@dataclass
class Calibration:
    single_px_to_m: np.ndarray
    keyframes: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> H_m_to_px
    per_frame_index: np.ndarray | None = None  # sorted frame numbers
    per_frame_m_to_px: np.ndarray | None = None  # [N, 3, 3]
    fps: float = 50.0
    spec: SurfaceSpec = field(default_factory=lambda: FIBA)
    meta: dict = field(default_factory=dict)

    @property
    def mode(self) -> str:
        if self.per_frame_index is not None:
            return "per_frame"
        if len(self.keyframes) > 1:
            return "keyframes"
        return "single"

    def H_m_to_px(self, frame: int | None = None) -> np.ndarray:
        if frame is None:
            return np.linalg.inv(self.single_px_to_m)
        if self.per_frame_index is not None:
            i = int(np.searchsorted(self.per_frame_index, frame))
            i = min(max(i, 0), len(self.per_frame_index) - 1)
            if i > 0 and abs(self.per_frame_index[i - 1] - frame) < abs(self.per_frame_index[i] - frame):
                i -= 1
            if abs(int(self.per_frame_index[i]) - frame) > 2:  # beyond the tracked range, not "nearby"
                return np.full((3, 3), np.nan)
            return self.per_frame_m_to_px[i]
        if self.keyframes:
            keys = sorted(self.keyframes)
            if frame <= keys[0]:
                return self.keyframes[keys[0]]
            if frame >= keys[-1]:
                return self.keyframes[keys[-1]]
            for a, b in zip(keys[:-1], keys[1:]):
                if a <= frame <= b:
                    return interpolate_m_to_px(self.keyframes[a], self.keyframes[b], (frame - a) / (b - a), self.spec)
        return np.linalg.inv(self.single_px_to_m)

    def H_px_to_m(self, frame: int | None = None) -> np.ndarray:
        H = self.H_m_to_px(frame)
        return np.linalg.inv(H) if np.isfinite(H).all() else np.full((3, 3), np.nan)

    def project(self, frame: int | None, points) -> np.ndarray:
        """Pixels -> metres for one frame. [N, 2] in, [N, 2] out."""
        pts = np.asarray(points, np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return pts
        return apply_h(self.H_px_to_m(frame), pts)

    def to_px(self, frame: int | None, points_m) -> np.ndarray:
        return apply_h(self.H_m_to_px(frame), points_m)

    def court_polygon_px(self, frame: int | None = None) -> np.ndarray | None:
        """Court corners in pixels for this frame ([4, 2] float32, bottom-left, bottom-right,
        top-right, top-left in court terms), or None when the frame is uncalibrated."""
        H = self.H_m_to_px(frame)
        if not np.isfinite(H).all():
            return None
        corners = np.float64([[0, 0], [self.spec.length_m, 0], [self.spec.length_m, self.spec.width_m], [0, self.spec.width_m]])
        px = apply_h(H, corners)
        return px.astype(np.float32) if np.isfinite(px).all() else None

    def on_court(self, xy_m, tolerance_m: float = 1.5) -> np.ndarray:
        xy = np.asarray(xy_m, np.float64).reshape(-1, 2)
        ok = np.isfinite(xy).all(axis=1)
        ok &= (xy[:, 0] >= -tolerance_m) & (xy[:, 0] <= self.spec.length_m + tolerance_m)
        ok &= (xy[:, 1] >= -tolerance_m) & (xy[:, 1] <= self.spec.width_m + tolerance_m)
        return ok

    def player_distances(self, tracks_path: str | Path, smooth_s: float = 0.4) -> dict[int, float]:
        """Metres run per player id from tracks.jsonl. Positions are median-smoothed
        over `smooth_s` seconds and implausible jumps are dropped."""
        per_id: dict[int, list[tuple[int, float, float]]] = {}
        for rec in iter_tracks(tracks_path):
            players = rec.get("players") or []
            if not players:
                continue
            feet = self.project(rec["frame"], [p["foot"] for p in players])
            for p, xy in zip(players, feet):
                if np.isfinite(xy).all():
                    per_id.setdefault(int(p["id"]), []).append((int(rec["frame"]), float(xy[0]), float(xy[1])))
        out: dict[int, float] = {}
        win = max(1, int(round(smooth_s * self.fps)))
        for pid, rows in per_id.items():
            rows.sort()
            arr = np.array(rows, np.float64)
            xy = _running_median(arr[:, 1:], win)
            frames = arr[:, 0]
            dist = 0.0
            for k in range(1, len(arr)):
                dt = (frames[k] - frames[k - 1]) / self.fps
                if dt <= 0 or dt > 2.0:
                    continue
                step = float(np.linalg.norm(xy[k] - xy[k - 1]))
                if step / dt <= MAX_SPEED_M_S:
                    dist += step
            out[pid] = round(dist, 1)
        return out


def _running_median(xy: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(xy) < win:
        return xy
    out = np.empty_like(xy)
    half = win // 2
    for i in range(len(xy)):
        lo, hi = max(0, i - half), min(len(xy), i + half + 1)
        out[i] = np.median(xy[lo:hi], axis=0)
    return out


def iter_tracks(path: str | Path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_calibration(path: str | Path = ROOT / "out" / "court_calib.json") -> Calibration:
    path = Path(path)
    data = json.loads(path.read_text())
    cal = Calibration(single_px_to_m=np.array(data["H_px_to_m"], np.float64), fps=float(data.get("fps", 50.0)), meta=data)
    for key, kf in (data.get("frames") or {}).items():
        H = kf.get("H_m_to_px")
        cal.keyframes[int(key)] = np.array(H, np.float64) if H else np.linalg.inv(np.array(kf["H_px_to_m"], np.float64))
    per_frame = data.get("per_frame")
    if per_frame:
        npz_path = Path(per_frame)
        if not npz_path.is_absolute():
            npz_path = ROOT / npz_path
        if npz_path.exists():
            npz = np.load(npz_path)
            cal.per_frame_index = npz["frames"].astype(np.int64)
            cal.per_frame_m_to_px = npz["H_m_to_px"].astype(np.float64)
    return cal


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="print distance_m per player from tracks.jsonl")
    ap.add_argument("--calib", default=ROOT / "out" / "court_calib.json")
    ap.add_argument("--tracks", default=ROOT / "out" / "tracks.jsonl")
    args = ap.parse_args()
    cal = load_calibration(args.calib)
    print("mode:", cal.mode)
    for pid, d in sorted(cal.player_distances(args.tracks).items()):
        print(f"player {pid}: {d} m")
