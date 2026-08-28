"""Draw the calibrated court onto a video frame.

    from vision.court.project import load_calibration
    from vision.court.draw import court_lines
    cal = load_calibration("out/court_calib.json")
    frame = court_lines(frame_bgr, frame_index, cal)   # in place, also returned

Sidelines, baselines, paint, free-throw circles, centre line and circle,
three-point lines, hoop marks. Boundary and three-point lines are drawn in the
highlight colour, the rest in the base colour, so a viewer sees at once that
the out-of-bounds line and the arc were recognised. Frames without a
calibration (NaN homography) come back untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision.court.geometry import FIBA, Arc, Line, SurfaceSpec, polylines  # noqa: E402
from vision.court.homography import apply_h  # noqa: E402
from vision.court.project import Calibration  # noqa: E402

BASE = (235, 235, 235)       # BGR, thin white
HIGHLIGHT = (80, 220, 80)    # BGR, green for boundary + three-point line


def _is_highlight(shape, spec: SurfaceSpec) -> bool:
    """Boundary rectangle and the three-point straights + arcs."""
    if isinstance(shape, Line):
        on_edge = all(v in (0.0, spec.length_m) for v in (shape.x1, shape.x2)) and shape.y1 == shape.y2 \
            or all(v in (0.0, spec.width_m) for v in (shape.y1, shape.y2)) and shape.x1 == shape.x2
        boundary = (shape.y1 == shape.y2 and shape.y1 in (0.0, spec.width_m)) or \
                   (shape.x1 == shape.x2 and shape.x1 in (0.0, spec.length_m))
        three_straight = shape.y1 == shape.y2 and shape.y1 in (0.9, spec.width_m - 0.9)
        return boundary or three_straight
    if isinstance(shape, Arc):
        return shape.r > 6.0  # three-point arcs (6.75 m); free-throw circles are 1.8 m
    return False


def draw_h(frame: np.ndarray, H_m_to_px: np.ndarray, spec: SurfaceSpec = FIBA, base=BASE, highlight=HIGHLIGHT,
           thickness: int = 1, highlight_thickness: int = 2) -> np.ndarray:
    """Court markings for one court->pixel homography, in place."""
    if H_m_to_px is None or not np.isfinite(H_m_to_px).all():
        return frame
    h, w = frame.shape[:2]
    limit = 4 * max(h, w)
    polys = polylines(spec)
    for shape, poly in zip(spec.shapes, polys):
        px = apply_h(H_m_to_px, poly)
        color, th = (highlight, highlight_thickness) if _is_highlight(shape, spec) else (base, thickness)
        for (x1, y1), (x2, y2) in zip(px[:-1], px[1:]):
            if not np.all(np.isfinite([x1, y1, x2, y2])) or max(abs(x1), abs(y1), abs(x2), abs(y2)) > limit:
                continue
            cv2.line(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, th, cv2.LINE_AA)
    return frame


def court_lines(frame_bgr: np.ndarray, frame_index: int | None, cal: Calibration, **style) -> np.ndarray:
    """Draw the calibrated court for `frame_index` onto `frame_bgr` (in place) and return it."""
    return draw_h(frame_bgr, cal.H_m_to_px(frame_index), cal.spec, **style)


def court_polygon_px(cal: Calibration, frame_index: int | None) -> np.ndarray | None:
    """The court boundary in pixels for `frame_index`, [4, 2] float32, or None if uncalibrated."""
    return cal.court_polygon_px(frame_index)


def on_court_px(cal: Calibration, frame_index: int | None, points_px, tolerance_m: float = 0.5) -> np.ndarray:
    """Bool per pixel point: inside the court plus 0.5 m (a foot on the line counts, the bench does not)."""
    return cal.on_court(cal.project(frame_index, points_px), tolerance_m)
