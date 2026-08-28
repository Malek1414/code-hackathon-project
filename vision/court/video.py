"""Frame writer that pipes raw BGR frames into ffmpeg (H.264, yuv420p), so the
result plays in a browser <video> tag. cv2.VideoWriter's mp4v does not."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class FfmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 20, threads: int | None = None):
        self.width, self.height = width - width % 2, height - height % 2
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{self.width}x{self.height}", "-r", f"{fps:.3f}", "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", *(["-threads", str(threads)] if threads else []), str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = frame[: self.height, : self.width]
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
