"""Cheap image renders for the board, cached by (path, mtime, size). No model
is ever loaded here; cv2 only draws boxes and grabs the last frame of a video."""
from __future__ import annotations

import os
import threading
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # a half-written mp4 would spam "moov atom not found"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:  # noqa: BLE001
    pass

from vision.monitor import status

MAX_W = 960
JPG_QUALITY = 80
BOX_COLORS = {  # BGR
    0: (255, 200, 90),  # player: blue-ish
    1: (90, 200, 255),  # ball: orange
    2: (120, 230, 120),  # hoop: green
    3: (200, 140, 255),  # referee: violet
}

_lock = threading.Lock()
_cache: dict[str, tuple[tuple, bytes | None]] = {}


def _encode(img) -> bytes | None:
    h, w = img.shape[:2]
    if w > MAX_W:
        scale = MAX_W / w
        img = cv2.resize(img, (MAX_W, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
    return buf.tobytes() if ok else None


def _key_of(*paths: Path) -> tuple | None:
    parts = []
    for p in paths:
        st = status._stat(p)
        if st is None:
            return None
        parts.append((str(p), st[0], st[1]))
    return tuple(parts)


def _cached(name: str, key: tuple | None, render) -> bytes | None:
    if key is None:
        _cache.pop(name, None)
        return None
    hit = _cache.get(name)
    if hit and hit[0] == key:
        return hit[1]
    try:
        data = render()
    except Exception:  # noqa: BLE001
        data = None
    _cache[name] = (key, data)
    return data


# ---------------------------------------------------------------- LABEL ----


def _render_label(txt: Path, img_path: Path) -> bytes | None:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    with open(txt, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            color = BOX_COLORS.get(cls, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            name = status.CLASSES.get(cls, str(cls))
            cv2.putText(img, name, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(img, txt.name, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return _encode(img)


def label_image() -> bytes | None:
    txt = status.newest_label()
    if txt is None:
        return _cached("label", None, None)
    img_path = status.image_for_label(txt)
    if img_path is None:
        return _cached("label", None, None)
    key = _key_of(txt, img_path)
    return _cached("label", key, lambda: _render_label(txt, img_path))


# ---------------------------------------------------------------- VIDEO ----


def _has_moov(path: Path, window: int = 256_000) -> bool:
    """An mp4 still open in cv2.VideoWriter has no moov atom yet (it is written on
    release). Checking head and tail bytes is far cheaper than a failed open."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            head = fh.read(min(size, window))
            if b"moov" in head:
                return True
            if size > window:
                fh.seek(max(0, size - window))
                return b"moov" in fh.read(window)
    except OSError:
        return False
    return False


def _last_frame(path: Path):
    if path.suffix.lower() in (".mp4", ".mov", ".m4v") and not _has_moov(path):
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for back in (1, 3, 10, 30, 100):
            idx = n - back
            if idx < 0:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        # unknown frame count or seek failed: read forward, keep the last one
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        last = None
        for _ in range(3000):
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
        return last
    finally:
        cap.release()


def _render_video(path: Path) -> bytes | None:
    frame = _last_frame(path)
    if frame is None:
        return None
    return _encode(np.ascontiguousarray(frame))


def video_image(name: str, path: Path) -> bytes | None:
    return _cached(name, _key_of(path), lambda: _render_video(path))


def _render_file(path: Path) -> bytes | None:
    img = cv2.imread(str(path))
    return _encode(img) if img is not None else None


def file_image(name: str, path: Path) -> bytes | None:
    return _cached(name, _key_of(path), lambda: _render_file(path))


def track_image() -> bytes | None:
    """TRACK writes overlay_latest.jpg while the mp4 is still open; prefer it."""
    latest = status.OUT_DIR / "overlay_latest.jpg"
    if latest.exists():
        data = file_image("track", latest)
        if data:
            return data
    return video_image("track", status.OUT_DIR / "overlay.mp4")


# ------------------------------------------------------------------ API ----

SOURCES = {
    "label": label_image,
    "track": track_image,
    "numbers": lambda: file_image("numbers", status.OUT_DIR / "numbers_preview.jpg"),
    "court": lambda: video_image("court", status.OUT_DIR / "court_propagate_preview.mp4"),
}


def get(name: str) -> bytes | None:
    fn = SOURCES.get(name)
    if fn is None:
        return None
    with _lock:
        return fn()


def tokens() -> dict:
    """Version token per image (changes when the source changes), None if missing."""
    out = {}
    for name in SOURCES:
        data = get(name)
        hit = _cache.get(name)
        out[name] = (str(abs(hash(hit[0]))) if hit and data else None)
    return out
