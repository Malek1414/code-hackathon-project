"""Outputs: MJPEG over HTTP (OBS / phone browser) and RTMP push via ffmpeg."""

from __future__ import annotations

import logging
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from vision.live.env import redact

log = logging.getLogger(__name__)

INDEX_HTML = b"""<!doctype html><title>FollowCam live</title>
<body style="margin:0;background:#111"><img src="/stream" style="width:100vw;height:auto;display:block"></body>"""


class MjpegServer:
    """Serves the latest JPEG at /stream (multipart/x-mixed-replace) and a page at /."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8501) -> None:
        self.host, self.port = host, port
        self._jpeg: bytes | None = None
        self._cond = threading.Condition()
        self._seq = 0
        self._server: ThreadingHTTPServer | None = None
        self.state_json: bytes = b"{}"  # Big Ball Baller live state, served at /state.json

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # quiet
                pass

            def do_GET(self) -> None:
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    seen = -1
                    try:
                        while True:
                            with outer._cond:
                                outer._cond.wait_for(lambda: outer._seq != seen, timeout=2.0)
                                jpeg, seen = outer._jpeg, outer._seq
                            if jpeg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                            self.wfile.write(str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                elif self.path.startswith("/state.json"):
                    body = outer.state_json
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(INDEX_HTML)))
                    self.end_headers()
                    self.wfile.write(INDEX_HTML)
                else:
                    self.send_response(404)
                    self.end_headers()

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True, name="mjpeg").start()
        log.info("MJPEG at http://%s:%d/stream", self.host, self.port)

    def update(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()


class RtmpPusher:
    """Feeds raw BGR frames to an ffmpeg process that pushes H.264/FLV to an
    RTMP URL. The URL comes from the environment and is never logged."""

    def __init__(self, url: str, width: int, height: int, fps: float, bitrate: str = "2500k") -> None:
        import imageio_ffmpeg

        self.url = url
        self.width, self.height, self.fps = width, height, fps
        gop = max(int(round(fps * 2)), 1)
        self.cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
            "-g", str(gop), "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", "5000k",
            "-c:a", "aac", "-b:a", "96k", "-shortest",
            "-f", "flv", url,
        ]
        self.proc: subprocess.Popen | None = None
        self.frames = 0
        self.error: str | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        log.info("RTMP push to %s (%dx%d @ %g fps)", redact(self.url), self.width, self.height, self.fps)

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.error is None

    def write(self, frame: np.ndarray) -> None:
        if not self.alive:
            return
        try:
            self.proc.stdin.write(frame.tobytes())
            self.frames += 1
        except (BrokenPipeError, OSError) as exc:
            self.error = f"ffmpeg pipe closed ({exc.__class__.__name__})"
            log.error("RTMP push stopped: %s", self.error)

    def close(self) -> str:
        if self.proc is None:
            return ""
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        err = self.proc.stderr.read().decode(errors="replace").strip() if self.proc.stderr else ""
        if err:
            err = err.replace(self.url, redact(self.url))
            log.error("ffmpeg: %s", err[-500:])
        return err
