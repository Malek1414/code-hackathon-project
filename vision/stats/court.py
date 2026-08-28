"""Court-side helpers: pixel → metres with COURT's calibration, on-court test,
distance run. Reads the `out/court_calib.json` contract (docs/ORCHESTRATION.md):
`H_px_to_m` (3x3), optional per-keyframe `frames: {"<frame>": H}`, `court_m`."""

from __future__ import annotations

import math
from collections import defaultdict

from .io import Frame, Point

MAX_PLAYER_SPEED_MS = 12.0  # steps faster than this are id swaps, not running


def _as_px_to_m(entry) -> list[list[float]] | None:
    """A keyframe entry is either a bare 3x3 H_px_to_m or COURT's dict with
    H_px_to_m and/or H_m_to_px."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        if entry.get("H_px_to_m"):
            return entry["H_px_to_m"]
        if entry.get("H_m_to_px"):
            import numpy as np

            return np.linalg.inv(np.asarray(entry["H_m_to_px"], float)).tolist()
        return None
    return entry


def homography_for(calib: dict, frame_no: int) -> list[list[float]] | None:
    """Nearest keyframe's H_px_to_m, else the top-level one. COURT's
    vision.court.project blends keyframes; this is the dependency-free
    fallback used when that module is not importable."""
    per_frame = calib.get("frames")
    if per_frame:
        nearest = min(per_frame, key=lambda k: abs(int(k) - frame_no))
        H = _as_px_to_m(per_frame[nearest])
        if H is not None:
            return H
    return calib.get("H_px_to_m")


_COURT_CAL = {}


def _court_calibration(calib: dict):
    """COURT's Calibration object for this calib dict (cached), or None."""
    key = id(calib)
    if key in _COURT_CAL:
        return _COURT_CAL[key]
    cal = None
    try:
        import json as _json
        import tempfile

        from vision.court.project import load_calibration

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            _json.dump(calib, fh)
        cal = load_calibration(fh.name)
    except Exception:  # noqa: BLE001 - fall back to the local homography helper
        cal = None
    _COURT_CAL[key] = cal
    return cal


def project(H: list[list[float]], p: Point) -> Point:
    x, y = p
    w = H[2][0] * x + H[2][1] * y + H[2][2]
    if abs(w) < 1e-9:
        return (math.nan, math.nan)
    return ((H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w)


def court_size(calib: dict) -> tuple[float, float]:
    c = calib.get("court_m") or {}
    return float(c.get("length", 28.0)), float(c.get("width", 15.0))


def on_court(calib: dict, frame_no: int, foot: Point, margin_m: float = 1.0) -> bool | None:
    """True/False if the foot point projects inside the court (+margin);
    None when no homography applies to this frame. Uses COURT's
    Calibration (keyframe blending, per-frame npz) when available."""
    cal = _court_calibration(calib)
    if cal is not None:
        xy = cal.project(frame_no, [list(foot)])[0]
        if not (math.isfinite(xy[0]) and math.isfinite(xy[1])):
            return None  # no homography for this frame (segment without keyframe)
        x, y = float(xy[0]), float(xy[1])
        length, width = court_size(calib)
        return -margin_m <= x <= length + margin_m and -margin_m <= y <= width + margin_m
    H = homography_for(calib, frame_no)
    if H is None:
        return None
    x, y = project(H, foot)
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    length, width = court_size(calib)
    return -margin_m <= x <= length + margin_m and -margin_m <= y <= width + margin_m


def distances_m(frames: list[Frame], calib: dict) -> dict[int, float]:
    """Path length of every player's projected foot point, in metres."""
    last: dict[int, tuple[float, Point]] = {}
    total: dict[int, float] = defaultdict(float)
    for fr in frames:
        H = homography_for(calib, fr.frame)
        if H is None:
            continue
        for p in fr.players:
            m = project(H, p.foot)
            if not all(math.isfinite(v) for v in m):
                continue
            if p.id in last:
                t0, m0 = last[p.id]
                dt = fr.t - t0
                d = math.hypot(m[0] - m0[0], m[1] - m0[1])
                if 0 < dt <= 0.5 and d / dt <= MAX_PLAYER_SPEED_MS:
                    total[p.id] += d
            last[p.id] = (fr.t, m)
    return dict(total)
