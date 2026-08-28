"""Annotated video: id + team-colored boxes, ball trail, hoops, shot flashes.

Written with cv2.VideoWriter (mp4v) at the source fps divided by the frame
stride, so the overlay plays in real time next to the original.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

TEAM_COLORS = {  # BGR. Fixed and high-contrast on purpose: the real jersey
    -1: (210, 210, 210),  # colors (blue vs black) are too close to tell apart.
    0: (255, 140, 0),  # team 0 = the bluer jersey (see teams.py)
    1: (0, 0, 255),
}
BALL_COLOR = (0, 220, 255)
HOOP_COLOR = (0, 255, 120)
TRAIL_LEN = 25
FLASH_S = 1.0


def _palette(team_bgr: dict[int, tuple[int, int, int]]) -> sv.ColorPalette:
    # class_id = team + 1 → index 0 unknown, 1 team 0, 2 team 1
    return sv.ColorPalette(
        [sv.Color(r=c[2], g=c[1], b=c[0]) for c in (team_bgr[-1], team_bgr[0], team_bgr[1])]
    )


class OverlayWriter:
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        events: Path | None = None,
        team_bgr: dict[int, tuple[int, int, int]] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open VideoWriter for {path}")
        self.fps = fps
        self.trail: deque[tuple[int, int]] = deque(maxlen=TRAIL_LEN)
        self.frames = 0
        self.set_team_colors(team_bgr or TEAM_COLORS)
        self.shots = _load_shots(events)

    def set_team_colors(self, team_bgr: dict[int, tuple[int, int, int]]) -> None:
        self.team_bgr = {**TEAM_COLORS, **team_bgr}
        pal = _palette(self.team_bgr)
        self.box = sv.BoxAnnotator(color=pal, thickness=2, color_lookup=sv.ColorLookup.CLASS)
        self.label = sv.LabelAnnotator(
            color=pal,
            color_lookup=sv.ColorLookup.CLASS,
            text_color=sv.Color.BLACK,
            text_scale=0.6,
            text_thickness=2,
            text_padding=4,
        )

    def write(self, frame: np.ndarray, record: dict) -> None:
        img = frame.copy()
        players = record["players"]
        if players:
            det = sv.Detections(
                xyxy=np.array([p["bbox"] for p in players], dtype=np.float32),
                confidence=np.array([p["conf"] for p in players], dtype=np.float32),
                class_id=np.array([p["team"] + 1 for p in players], dtype=int),
                tracker_id=np.array([p["id"] for p in players], dtype=int),
            )
            img = self.box.annotate(img, det)
            img = self.label.annotate(img, det, labels=[f"#{p['id']}" for p in players])
            for p in players:
                fx, fy = (int(v) for v in p["foot"])
                cv2.circle(img, (fx, fy), 3, self.team_bgr[p["team"]], -1)

        for h in record["hoops"]:
            x1, y1, x2, y2 = (int(v) for v in h["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), HOOP_COLOR, 2)

        ball = record["ball"]
        if ball:
            cx, cy = (int(v) for v in ball["center"])
            self.trail.append((cx, cy))
        elif self.trail:
            # No ball this frame: let the trail fade instead of freezing it.
            self.trail.popleft()
        pts = list(self.trail)
        for i in range(1, len(pts)):
            a = (i + 1) / len(pts)
            cv2.line(img, pts[i - 1], pts[i], BALL_COLOR, max(1, int(4 * a)))
        if ball:
            cv2.circle(img, pts[-1], 8, BALL_COLOR, 2)

        self._flash(img, record["t"])
        self._hud(img, record)
        self.writer.write(img)
        self.frames += 1

    def _flash(self, img: np.ndarray, t: float) -> None:
        for s in self.shots:
            if 0 <= t - s["t"] <= FLASH_S:
                text = "MADE" if s.get("made") else "MISS"
                color = (0, 220, 0) if s.get("made") else (0, 0, 230)
                hb = s.get("hoop_bbox")
                x, y = (int(hb[0]), int(hb[1]) - 20) if hb else (img.shape[1] // 2 - 80, 80)
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 6)
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 2)

    def _hud(self, img: np.ndarray, r: dict) -> None:
        n0 = sum(p["team"] == 0 for p in r["players"])
        n1 = sum(p["team"] == 1 for p in r["players"])
        txt = f"f{r['frame']}  {r['t']:6.2f}s  team0 {n0}  team1 {n1}  ball {'y' if r['ball'] else '-'}"
        cv2.putText(img, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    def close(self) -> None:
        self.writer.release()


def _load_shots(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("shots", [])
    except (json.JSONDecodeError, AttributeError):
        return []
