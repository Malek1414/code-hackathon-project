"""Shared helpers for the QA sheets: tolerant readers, frame grabbing, drawing, tiling.

Colors mirror vision/track/overlay.py (TRACK owns them; copied so QA never
imports TRACK's module while it is being edited).
"""

from __future__ import annotations

import bisect
import fcntl
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"
QA_DIR = OUT / "qa"
TRACKS = OUT / "tracks.jsonl"
EVENTS = OUT / "events.json"
META = OUT / "tracks_meta.json"

# BGR, same as vision/track/overlay.py
TEAM_COLORS = {-1: (210, 210, 210), 0: (255, 140, 0), 1: (0, 0, 255)}
TEAM_NAMES = {-1: "unknown", 0: "team 0", 1: "team 1"}
BALL_COLOR = (0, 220, 255)
HOOP_COLOR = (0, 255, 120)
BG = (24, 24, 24)
FG = (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --- readers -----------------------------------------------------------------


def read_tracks(path: Path = TRACKS) -> list[dict]:
    """tracks.jsonl, tolerant of a half-written last line (TRACK streams it)."""
    frames: list[dict] = []
    if not path.exists():
        return frames
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return frames


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def meta_for(tracks: Path) -> dict:
    """tracks_meta.json next to the tracks file (archive dirs like out/dev60_v4/), else out/tracks_meta.json."""
    local = tracks.with_name("tracks_meta.json")
    return read_json(local) or (read_json(META) if tracks.resolve() == TRACKS.resolve() else None) or {}


def resolve_clip(*candidates: str | None) -> Path:
    """First existing clip among the given paths (events.json, tracks_meta.json, default)."""
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
    raise FileNotFoundError(f"no clip found among {candidates}")


class TrackIndex:
    """Nearest processed frame for any source frame number."""

    def __init__(self, frames: list[dict]) -> None:
        self.frames = sorted(frames, key=lambda f: f["frame"])
        self.keys = [f["frame"] for f in self.frames]
        gaps = np.diff(self.keys) if len(self.keys) > 1 else np.array([1])
        self.stride = int(np.median(gaps)) if len(gaps) else 1

    def nearest(self, frame: int, max_gap: int | None = None) -> dict | None:
        if not self.keys:
            return None
        if max_gap is None:
            max_gap = self.stride
        i = bisect.bisect_left(self.keys, frame)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.keys):
                d = abs(self.keys[j] - frame)
                if d <= max_gap and (best is None or d < abs(self.keys[best] - frame)):
                    best = j
        return self.frames[best] if best is not None else None


# --- frames --------------------------------------------------------------------


class FrameGrabber:
    def __init__(self, clip: Path) -> None:
        self.cap = cv2.VideoCapture(str(clip))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {clip}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 50.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next = 0

    SEEK_AHEAD = 150  # a cv2 seek costs ~2 s on these clips, grabbing forward ~100 fps

    def get(self, frame: int) -> np.ndarray | None:
        """Frame by source index. Callers should ask in ascending order: short
        forward gaps are skipped with grab(), only backwards or far jumps seek."""
        frame = max(0, min(frame, self.n - 1)) if self.n > 0 else max(0, frame)
        gap = frame - self._next
        if gap < 0 or gap > self.SEEK_AHEAD:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        else:
            for _ in range(gap):
                if not self.cap.grab():
                    break
        ok, img = self.cap.read()
        self._next = frame + 1
        return img if ok else None

    def close(self) -> None:
        self.cap.release()


# --- drawing -------------------------------------------------------------------


def put_text(img, text: str, org: tuple[int, int], scale: float = 0.55, color=FG, thick: int = 1) -> None:
    """Text with a 1 px dark halo. The halo is drawn by offset copies, not by a
    thicker stroke: OpenCV 5's font renderer changes glyph advance with the
    thickness, so a thicker outline would stick out past the fill."""
    x, y = org
    for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1), (0, 1), (1, 0)):
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_box(img, bbox, color, label: str | None = None, thick: int = 2, scale: float = 1.0) -> None:
    x1, y1, x2, y2 = (int(round(v * scale)) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if label:
        put_text(img, label, (x1, max(14, y1 - 6)), 0.5, color, 1)


def fit_height(img, h: int) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


def fit_width(img, w: int) -> np.ndarray:
    s = w / img.shape[1]
    return cv2.resize(img, (w, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)


def tile(images: list[np.ndarray], cols: int, pad: int = 6, bg=BG) -> np.ndarray:
    """Grid of equally sized tiles (first image sets the size, others are padded)."""
    if not images:
        return np.full((60, 400, 3), bg, np.uint8)
    th = max(i.shape[0] for i in images)
    tw = max(i.shape[1] for i in images)
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * (th + pad) + pad, cols * (tw + pad) + pad, 3), bg, np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        y = pad + r * (th + pad)
        x = pad + c * (tw + pad)
        canvas[y : y + im.shape[0], x : x + im.shape[1]] = im
    return canvas


def with_header(body: np.ndarray, lines: list[str], scale: float = 0.8, bg=BG) -> np.ndarray:
    lh = int(34 * scale / 0.8)
    h = lh * len(lines) + 14
    head = np.full((h, body.shape[1], 3), bg, np.uint8)
    for i, ln in enumerate(lines):
        put_text(head, ln, (12, lh * (i + 1)), scale, FG, 2 if i == 0 else 1)
    return np.vstack([head, body])


def band_label(width: int, text: str, color=FG, h: int = 36, bg=BG) -> np.ndarray:
    band = np.full((h, width, 3), bg, np.uint8)
    put_text(band, text, (12, h - 12), 0.7, color, 2)
    return band


def save_jpg(path: Path, img: np.ndarray, quality: int = 85) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.jpg")
    cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    tmp.replace(path)


def fmt_t(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


@dataclass
class Sample:
    frame: int
    t: float
    line: dict


@contextmanager
def qa_lock(out: Path = QA_DIR):
    """One QA writer at a time per output dir (the watcher and a manual run
    would otherwise delete each other's half-written files)."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
