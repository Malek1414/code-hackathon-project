"""Synthetic check: render a court through a known camera, solve it back, compare.

    .venv/bin/python vision/court/test_synthetic.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision.court.calibrate import calib_dict, draw_court, overlay_image, solve  # noqa: E402
from vision.court.geometry import FIBA  # noqa: E402
from vision.court.homography import apply_h  # noqa: E402

W, H = 1920, 1080


def fake_camera() -> np.ndarray:
    """Court metres -> pixels for a tripod on the far sideline, slightly elevated."""
    src = np.float32([[0, 0], [28, 0], [28, 15], [0, 15]])  # bottom-left, bottom-right, top-right, top-left
    dst = np.float32([[120, 1000], [1800, 1000], [1400, 420], [520, 420]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def main() -> int:
    H_m_to_px = fake_camera()
    frame = np.full((H, W, 3), (70, 90, 110), np.uint8)
    draw_court(frame, H_m_to_px, FIBA, color=(230, 230, 230), thickness=3)

    rng = np.random.default_rng(0)
    clicks = {}
    for lm in FIBA.landmarks:
        px = apply_h(H_m_to_px, [[lm.x, lm.y]])[0] + rng.normal(0, 1.5, 2)  # human click jitter
        clicks[lm.id] = (float(px[0]), float(px[1]))
    clicks.pop("paint_1_front_top")  # a skipped one
    fit = solve(FIBA, clicks)
    data = calib_dict(FIBA, {0: clicks}, {0: fit}, "synthetic", 50.0, (W, H))

    # a player standing at centre court must project to (14, 7.5)
    centre_px = apply_h(H_m_to_px, [[14, 7.5]])
    back = apply_h(np.array(data["H_px_to_m"]), centre_px)[0]
    err_centre = float(np.linalg.norm(back - [14, 7.5]))
    print(f"fit: {fit.mean_error_px:.2f} px, {fit.mean_error_m:.3f} m, inliers {fit.inliers}/{fit.total}")
    print(f"centre court round trip error: {err_centre:.3f} m")
    print("warnings:", fit.warnings)

    out = Path(tempfile.gettempdir()) / "court_synthetic.jpg"
    cv2.imwrite(str(out), overlay_image(frame, FIBA, clicks, fit))
    json.dumps(data)  # must be serialisable
    assert fit.mean_error_m < 0.2, fit.mean_error_m
    assert err_centre < 0.2, err_centre
    assert fit.usable
    print("OK", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
