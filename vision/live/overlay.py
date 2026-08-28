"""Drawing for the live view: boxes from the last tracks line, score bar,
made-basket flash, help line. Pure functions on BGR frames (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from vision.live.score import ScoreBoard

TEAM_COLOR = {0: (255, 140, 0), 1: (0, 0, 255), -1: (160, 160, 160)}  # BGR: blue, red, grey
BALL_COLOR = (0, 165, 255)
HOOP_COLOR = (0, 230, 255)
FLASH_MADE = (60, 200, 60)
FLASH_MISS = (60, 60, 200)


def clock(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def draw_tracks(frame: np.ndarray, record: dict | None, holder: int | None) -> None:
    if not record:
        return
    for p in record.get("players") or []:
        x1, y1, x2, y2 = (int(v) for v in p["bbox"])
        color = TEAM_COLOR.get(p.get("team", -1), TEAM_COLOR[-1])
        thick = 4 if p["id"] == holder else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        tag = f"{p['id']}" + ("  ball" if p["id"] == holder else "")
        cv2.putText(frame, tag, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    for h in record.get("hoops") or []:
        x1, y1, x2, y2 = (int(v) for v in h["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), HOOP_COLOR, 2)
    b = record.get("ball")
    if b:
        cx, cy = (int(v) for v in b["center"])
        cv2.circle(frame, (cx, cy), 14, BALL_COLOR, 3)


def draw_score_bar(frame: np.ndarray, board: ScoreBoard, t: float, info: str = "") -> None:
    h, w = frame.shape[:2]
    bar_h = 64
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, board.line(), (24, 44), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
    cv2.putText(frame, clock(t), (w // 2 - 50, 44), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    fg = board.fg_line()
    (tw, _), _ = cv2.getTextSize(fg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, fg, (w - tw - 24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
    if info:
        cv2.putText(frame, info, (24, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def draw_flash(frame: np.ndarray, label: str, made: bool, age_s: float, duration_s: float) -> None:
    if age_s < 0 or age_s > duration_s:
        return
    h, w = frame.shape[:2]
    alpha = 0.35 * max(0.0, 1 - age_s / duration_s)
    color = FLASH_MADE if made else FLASH_MISS
    tint = np.full_like(frame, color)
    cv2.addWeighted(tint, alpha, frame, 1 - alpha, 0, frame)
    scale = 2.2 if made else 1.2
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
    x, y = (w - tw) // 2, h // 2 + th // 2
    cv2.putText(frame, label, (x + 3, y + 3), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 6)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 3)
