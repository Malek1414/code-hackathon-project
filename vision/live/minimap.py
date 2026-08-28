"""Live 2D court panel: COURT's minimap renderer (vision/court/minimap.py)
on the last tracks line, plus jersey numbers (out/identities.json), the
possession highlight and a calibration notice. Rendered every frame from
the last known boxes, so it runs in real time next to the video."""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from vision.court.minimap import BALL, CourtCanvas, TEAM, render_frame
from vision.court.project import Calibration, load_calibration

log = logging.getLogger(__name__)

TEXT = (235, 235, 235)
HIGHLIGHT = (255, 255, 255)


def load_numbers(path: str | Path = "out/identities.json") -> dict[int, str]:
    """track id -> jersey label ("12"), from NUMBERS' identities.json if present."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[int, str] = {}
    for pl in d.get("players") or []:
        if pl.get("number") is not None:
            for tid in pl.get("track_ids") or []:
                out[int(tid)] = str(pl["number"])
    for tid, tr in (d.get("tracks") or {}).items():
        if tr.get("number") is not None:
            out.setdefault(int(tid), str(tr["number"]))
    return out


def try_load_calibration(path: str | Path = "out/court_calib.json") -> Calibration | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        cal = load_calibration(p)
        log.info("court calibration %s (%s)", p, cal.mode)
        return cal
    except Exception as exc:  # noqa: BLE001 - a broken file must not stop the live view
        log.warning("court calibration unusable (%s): minimap stays uncalibrated", exc)
        return None


class MiniMap:
    def __init__(self, calib: Calibration | None, *, numbers: dict[int, str] | None = None,
                 scale: float = 40.0, trail_len: int = 25) -> None:
        self.cal = calib
        self.numbers = numbers or {}
        self.canvas = CourtCanvas(scale=scale)
        self.trail: deque = deque(maxlen=trail_len)

    @property
    def size(self) -> tuple[int, int]:
        return self.canvas.w, self.canvas.h

    def render(self, record: dict | None, holder: int | None, pan_deg: float | None = None) -> np.ndarray:
        rec = record or {"frame": 0, "t": 0.0, "players": [], "ball": None, "hoops": []}
        panning = pan_deg is not None and abs(pan_deg) >= 1.0
        if panning:
            # the static calibration does not hold while the rig pans: no moving dots, only the notice
            rec = {**rec, "players": [], "ball": None}
            self.trail.clear()
        img = render_frame(self.canvas, self.cal, rec, self.trail, show_ids=False)  # COURT's path: dots, trail, limits
        frame = int(rec.get("frame", 0))
        calibrated = self.cal is not None and np.isfinite(self.cal.H_m_to_px(frame)).all()
        uncertain = calibrated and hasattr(self.cal, "is_uncertain") and bool(self.cal.is_uncertain(frame))
        if calibrated and rec.get("players") and not uncertain:
            players = rec["players"]
            feet = self.cal.project(frame, [p["foot"] for p in players])
            ok = self.cal.on_court(feet)
            for p, xy, keep in zip(players, feet, ok):
                if not keep:
                    continue
                c = self.canvas.px(float(xy[0]), float(xy[1]))
                if p["id"] == holder:
                    cv2.circle(img, c, 14, HIGHLIGHT, 2, cv2.LINE_AA)
                    cv2.circle(img, c, 17, BALL, 1, cv2.LINE_AA)
                label = self.numbers.get(int(p["id"]))
                if label:
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    org = (c[0] + 12, c[1] + th // 2)
                    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3, cv2.LINE_AA)
                    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
        if panning:
            cv2.putText(img, f"Kamera schwenkt ({pan_deg:+.0f} deg): Kalibrierung aus", (12, self.canvas.h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 240), 2, cv2.LINE_AA)
        elif calibrated and self.cal.mode == "single":
            cv2.putText(img, "static calibration", (12, self.canvas.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        TEXT, 1, cv2.LINE_AA)
        self._legend(img)
        return img

    def _legend(self, img: np.ndarray) -> None:
        x, y = self.canvas.w - 150, self.canvas.h - 14
        cv2.circle(img, (x, y - 4), 6, TEAM[0], -1, cv2.LINE_AA)
        cv2.putText(img, "A", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
        cv2.circle(img, (x + 40, y - 4), 6, TEAM[1], -1, cv2.LINE_AA)
        cv2.putText(img, "B", (x + 50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
        cv2.circle(img, (x + 80, y - 4), 5, BALL, -1, cv2.LINE_AA)
        cv2.putText(img, "ball", (x + 90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)


def compose_side_by_side(video: np.ndarray, panel: np.ndarray, out_w: int, out_h: int, panel_frac: float = 1 / 3) -> np.ndarray:
    """Video on the left (2/3), court panel on the right (1/3), letterboxed."""
    pw = int(round(out_w * panel_frac))
    vw = out_w - pw
    out = np.full((out_h, out_w, 3), (24, 24, 24), np.uint8)
    vh = int(round(video.shape[0] * vw / video.shape[1]))
    if vh > out_h:
        vh = out_h
        vw2 = int(round(video.shape[1] * vh / video.shape[0]))
        v = cv2.resize(video, (vw2, vh))
        out[0:vh, (vw - vw2) // 2:(vw - vw2) // 2 + vw2] = v
    else:
        v = cv2.resize(video, (vw, vh))
        out[(out_h - vh) // 2:(out_h - vh) // 2 + vh, 0:vw] = v
    ph = int(round(panel.shape[0] * pw / panel.shape[1]))
    if ph > out_h:
        ph = out_h
        pw2 = int(round(panel.shape[1] * ph / panel.shape[0]))
        p = cv2.resize(panel, (pw2, ph))
        out[0:ph, vw + (pw - pw2) // 2:vw + (pw - pw2) // 2 + pw2] = p
    else:
        p = cv2.resize(panel, (pw, ph))
        out[(out_h - ph) // 2:(out_h - ph) // 2 + ph, vw:vw + pw] = p
    return out
