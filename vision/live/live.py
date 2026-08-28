"""LIVE mode: camera or video in, running score + stats overlay out.

    .venv/bin/python -m vision.live.live --source data/clips/dev60.mp4 --realtime
    .venv/bin/python -m vision.live.live --source 0            # camera index (Continuity Camera)

Detection runs in a worker thread at whatever rate the models allow (~10 fps
on MPS) using TRACK's per-frame API (vision.track.tracker.Tracker.step);
every source frame is rendered with the last known boxes. STATS's
StatsEngine gets one tracks line per processed frame and reports made /
missed shots at most 0.5 s after the ball dropped; the scoreboard auto-calls
+2 for the shooter's team, a human vetoes with hotkeys.

Hotkeys (window focused): 1/2 = +2 team A/B, 3/4 = +3, z = undo, q = quit.
Outputs: preview window, MJPEG at http://127.0.0.1:8501/stream, and an RTMP
push when FOLLOWCAM_RTMP_URL is set (read from .env; never logged).
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path

import cv2

from vision.live.env import load_dotenv, rtmp_url
from vision.live.overlay import draw_flash, draw_score_bar, draw_tracks
from vision.live.score import ScoreBoard
from vision.live.stream import MjpegServer, RtmpPusher
from vision.stats.engine import StatsEngine

log = logging.getLogger("live")

FLASH_S = 1.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="camera index (0, 1, ...) or video path")
    ap.add_argument("--realtime", action="store_true", help="pace a video file like a live camera")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--process-fps", type=float, default=10.0, help="target detection rate")
    ap.add_argument("--out-width", type=int, default=1280, help="MJPEG / RTMP frame width")
    ap.add_argument("--mjpeg-port", type=int, default=8501)
    ap.add_argument("--no-mjpeg", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this much source time (tests)")
    ap.add_argument("--events-out", default="out/live_events.json")
    ap.add_argument("--weights-players", default="models/yolo11s.pt")
    ap.add_argument("--weights-ballhoop", default="models/ball_hoop_avishah.pt")
    ap.add_argument("--weights", default=None, help="single contract model (LABEL's best.pt)")
    return ap.parse_args(argv)


def open_source(src: str) -> tuple[cv2.VideoCapture, bool, float]:
    is_cam = src.isdigit()
    cap = cv2.VideoCapture(int(src)) if is_cam else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {src!r}")
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0
    return cap, is_cam, fps


class Worker(threading.Thread):
    """Takes the newest frame, runs the tracker + engine, publishes the result."""

    def __init__(self, tracker, engine: StatsEngine, events: queue.Queue) -> None:
        super().__init__(daemon=True, name="detect")
        self.tracker, self.engine, self.events = tracker, engine, events
        self._slot: tuple | None = None
        self._cond = threading.Condition()
        self.latest: dict | None = None
        self.holder: int | None = None
        self.processed = 0
        self.proc_dt = 0.1
        self.stop = False

    def offer(self, frame, index: int, t: float) -> None:
        with self._cond:
            self._slot = (frame, index, t)
            self._cond.notify()

    def run(self) -> None:
        last_t = None
        while not self.stop:
            with self._cond:
                self._cond.wait_for(lambda: self._slot is not None or self.stop, timeout=0.5)
                if self._slot is None:
                    continue
                frame, index, t = self._slot
                self._slot = None
            record = self.tracker.step(frame, index, t)
            if last_t is not None and t > last_t:
                self.proc_dt = 0.8 * self.proc_dt + 0.2 * (t - last_t)
                self.engine.possession.dt = self.proc_dt
            last_t = t
            for ev in self.engine.push(record):
                self.events.put(ev)
            self.latest = record
            self.holder = self.engine.holder
            self.processed += 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    load_dotenv(".env")

    from vision.track.tracker import Tracker  # TRACK's per-frame API (loads torch/ultralytics)

    cap, is_cam, src_fps = open_source(args.source)
    tracker = Tracker(args.weights_players, args.weights_ballhoop, args.device, weights=args.weights, fps=src_fps)
    engine = StatsEngine(dt=1.0 / args.process_fps, fps=src_fps)
    events: queue.Queue = queue.Queue()
    worker = Worker(tracker, engine, events)
    worker.start()
    board = ScoreBoard()

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("no frames from source")
    h, w = frame.shape[:2]
    out_w = min(args.out_width, w)
    out_h = int(round(h * out_w / w / 2) * 2)

    mjpeg = None
    if not args.no_mjpeg:
        mjpeg = MjpegServer(port=args.mjpeg_port)
        mjpeg.start()
    pusher = None
    url = rtmp_url()
    if url:
        pusher = RtmpPusher(url, out_w, out_h, min(src_fps, 30.0))
        pusher.start()
    else:
        log.info("no FOLLOWCAM_RTMP_URL set: RTMP push off")

    window = "FollowCam LIVE"
    if not args.no_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, out_w, out_h)

    realtime = is_cam or args.realtime
    stride = max(1, int(round(src_fps / args.process_fps)))
    t_start = time.monotonic()
    idx = 0
    flash: tuple[str, bool, float] | None = None  # label, made, t_shown
    shots_log: list[dict] = []
    frames_rendered = 0
    rtmp_next = 0.0
    log.info("source %s: %dx%d @ %.1f fps, %s, detection stride %d", args.source, w, h, src_fps,
             "realtime" if realtime else "offline", stride)

    try:
        while True:
            if idx > 0:
                if realtime and not is_cam:
                    # pace like a camera; drop frames if rendering falls behind
                    target = t_start + idx / src_fps
                    lag = time.monotonic() - target
                    if lag > 2.0 / src_fps:
                        cap.grab()
                        idx += 1
                        continue
                    if lag < 0:
                        time.sleep(-lag)
                ok, frame = cap.read()
                if not ok:
                    break
            t = (time.monotonic() - t_start) if is_cam else idx / src_fps
            if args.max_seconds and t > args.max_seconds:
                break

            if realtime:
                worker.offer(frame, idx, t)  # worker takes it when free
            elif idx % stride == 0:
                worker.offer(frame, idx, t)
                while worker.processed < idx // stride + 1 and worker.is_alive():
                    time.sleep(0.001)

            while True:
                try:
                    ev = events.get_nowait()
                except queue.Empty:
                    break
                act = board.auto_shot(ev, t)
                flash = (act.label, ev.made, t)
                shots_log.append(ev.to_dict())
                log.info("shot %s at %.1fs: %s", "MADE" if ev.made else "miss", ev.t, act.label)

            view = frame.copy()
            draw_tracks(view, worker.latest, worker.holder)
            info = f"det {1 / max(worker.proc_dt, 1e-3):.1f} fps   1/2 +2  3/4 +3  z undo  q quit"
            draw_score_bar(view, board, t, info)
            if flash:
                draw_flash(view, flash[0], flash[1], t - flash[2], FLASH_S)
            small = cv2.resize(view, (out_w, out_h)) if out_w != w else view
            if mjpeg:
                ok_j, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok_j:
                    mjpeg.update(jpeg.tobytes())
            if pusher and pusher.alive and t >= rtmp_next:
                pusher.write(small)
                rtmp_next = t + 1.0 / pusher.fps
            frames_rendered += 1

            if not args.no_window:
                cv2.imshow(window, small)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                    team = 0 if key in (ord("1"), ord("3")) else 1
                    pts = 2 if key in (ord("1"), ord("2")) else 3
                    act = board.manual(team, pts, t)
                    flash = (act.label, True, t)
                elif key == ord("z"):
                    act = board.undo(t)
                    if act:
                        flash = (act.label, False, t)
            idx += 1
    finally:
        worker.stop = True
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        for ev in engine.finish():
            board.auto_shot(ev, t)
            shots_log.append(ev.to_dict())
        if pusher:
            pusher.close()
        if mjpeg:
            mjpeg.stop()
        summary = {
            "fps": src_fps,
            "clip": args.source,
            "shots": shots_log,
            "score": {str(k): vars(v) for k, v in board.teams.items()},
            "unassigned_baskets": board.unassigned,
            "frames_rendered": frames_rendered,
            "frames_processed": worker.processed,
            "rtmp_frames": pusher.frames if pusher else 0,
        }
        Path(args.events_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.events_out).write_text(json.dumps(summary, indent=1))
        log.info("done: rendered %d, processed %d, shots %d, score %s -> %s", frames_rendered,
                 worker.processed, len(shots_log), board.line(), args.events_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
