"""Short H.264 clip per shot, cut from out/overlay.mp4 (boxes drawn) or, when
the overlay is missing, still being written, or belongs to another clip, from
the raw source clip without boxes. Uses the imageio_ffmpeg binary.

Two files per shot: shot_<n>.mp4 (normal speed) and shot_<n>_half.mp4
(setpts=2.0*PTS). Both 720 px high, yuv420p, faststart, no audio.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg

from .common import META, OUT, ROOT, read_json

OVERLAY = OUT / "overlay.mp4"
BEFORE_S, AFTER_S = 2.0, 1.5
SETTLE_S = 8.0  # an overlay written less than this ago is treated as in progress
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: Path) -> tuple[float, float] | None:
    """(fps, duration_s) via cv2, None if the file cannot be opened (no moov atom yet)."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps <= 0 or n <= 0:
            return None
        return fps, n / fps
    finally:
        cap.release()


def pick_source(clip: Path, t_end: float, overlay: Path = OVERLAY) -> tuple[Path, str]:
    """Overlay if usable for this clip and time span, else the raw clip. Returns (path, reason)."""
    if not overlay.exists():
        return clip, "overlay.mp4 missing"
    meta = read_json(META) or {}
    meta_clip = meta.get("clip")
    if meta_clip and (ROOT / meta_clip).resolve() != clip.resolve():
        return clip, f"overlay.mp4 belongs to {meta_clip}"
    if time.time() - overlay.stat().st_mtime < SETTLE_S:
        return clip, "overlay.mp4 still being written"
    info = probe(overlay)
    if info is None:
        return clip, "overlay.mp4 not readable yet"
    if info[1] + 0.5 < t_end:
        return clip, f"overlay.mp4 ends at {info[1]:.1f} s"
    return overlay, "overlay"


def cut(src: Path, start: float, dur: float, dst: Path, half: bool) -> None:
    vf = "scale=-2:720"
    rate: list[str] = []
    if half:
        vf = "setpts=2.0*PTS," + vf
        rate = ["-r", "25"]  # setpts alone halves the frame rate; duplicate frames so it stays smooth
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    cmd = [
        FFMPEG, "-y", "-v", "error", "-nostdin",
        "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}", "-i", str(src),
        "-an", "-filter:v", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", *rate,
        "-threads", "2",  # ORCH 13:41: renders must not starve the training / tracking runs
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"ffmpeg failed for {dst.name}: {proc.stderr.strip()[-300:]}")
    tmp.replace(dst)


def render_clip(n: int, t: float, clip: Path, out: Path) -> dict:
    """Both speed variants for one shot. Returns the manifest entry."""
    src, reason = pick_source(clip, t + AFTER_S)
    start = t - BEFORE_S
    if start < 0:
        start = 0.0
    dur = t + AFTER_S - start
    normal, half = out / f"shot_{n}.mp4", out / f"shot_{n}_half.mp4"
    cut(src, start, dur, normal, half=False)
    cut(src, start, dur, half, half=True)
    boxes = src == OVERLAY or src.name == OVERLAY.name
    return {
        "video": normal.name,
        "video_half": half.name,
        "video_source": "overlay" if boxes else "raw",
        "video_reason": reason,
        "video_start": round(start, 3),
        "video_end": round(t + AFTER_S, 3),
        "video_caption": (
            f"video: overlay.mp4 with boxes, {start - t:+.1f} s to {AFTER_S:+.1f} s"
            if boxes
            else f"video: raw clip without boxes ({reason}), {start - t:+.1f} s to {AFTER_S:+.1f} s"
        ),
    }
