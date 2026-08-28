"""LIVE mode: camera or video in, running score + stats overlay out.

    .venv/bin/python -m vision.live.live --source data/clips/dev60.mp4 --realtime
    .venv/bin/python -m vision.live.live --source 0            # camera index (Continuity Camera)
    .venv/bin/python -m vision.live.live --source data/clips/dev60.mp4 --realtime --replay out/tracks.jsonl

Detection runs in a worker thread at whatever rate the models allow (~10 fps
on MPS) using TRACK's per-frame API (vision.track.tracker.Tracker.step);
every source frame is rendered with the last known boxes. STATS's
StatsEngine gets one tracks line per processed frame and reports made /
missed shots at most 0.5 s after the ball dropped; the scoreboard auto-calls
+2 for the shooter's team, a human vetoes with hotkeys.

Hotkeys (window focused): 1/2 = +2 team A/B, 3/4 = +3, z = undo, q = quit.
2D court (--minimap panel|window|off): COURT's minimap renderer on the last
tracks line, projected with out/court_calib.json (static calibration from one
frame: a pan breaks it, the panel says so), jersey numbers from
out/identities.json when known, possession holder ringed. Team per track is
TRACK's vote-smoothed value as it arrives in the tracks line.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `python vision/live/live.py` and `-m` both work

from vision.live.env import load_dotenv, rtmp_url
from vision.live.minimap import MiniMap, compose_side_by_side, load_numbers, try_load_calibration
from vision.live.overlay import draw_flash, draw_score_bar, draw_tracks

try:  # COURT's court overlay (vision/court/draw.py); optional so live never depends on it
    from vision.court.draw import court_lines
except Exception:  # noqa: BLE001
    court_lines = None
from vision.live.score import ScoreBoard
from vision.live.stream import MjpegServer, RtmpPusher
from vision.stats.engine import StatsEngine

log = logging.getLogger("live")

FLASH_S = 1.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None,
                    help="camera index (0, 1, ...), 'auto' = first camera that delivers frames, or a video path")
    ap.add_argument("--list-sources", action="store_true",
                    help="probe camera indices 0-4 (one frame each, 2 s timeout), print ok/no-frame, exit")
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
    ap.add_argument("--replay", default=None,
                    help="tracks.jsonl to replay instead of running the models; must be the tracks of the SAME clip "
                         "as --source (e.g. --source data/clips/dev60.mp4 --replay out/dev60/tracks.jsonl)")
    ap.add_argument("--minimap", choices=["panel", "window", "off"], default="panel",
                    help="2D court: right third of the output (panel), its own window, or off")
    ap.add_argument("--calib", default=None,
                    help="default: out/court_calib_<source stem>.json, else out/court_calib.json")
    ap.add_argument("--identities", default="out/identities.json")
    ap.add_argument("--no-court-lines", action="store_true", help="do not draw the court on the video")
    ap.add_argument("--panel-every", type=int, default=3,
                    help="render the court panel every Nth frame and reuse it in between (render rate)")
    return ap.parse_args(argv)


def _read_with_timeout(cap: cv2.VideoCapture, timeout_s: float):
    """cap.read() in a thread: a camera that opens but never delivers must not hang us."""
    box: list = []

    def run() -> None:
        box.append(cap.read())

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout_s)
    if not box:
        return False, None
    return box[0]


def probe_source(index: int, timeout_s: float = 2.0) -> tuple[str, int, int, float]:
    """('ok' | 'no-frame' | 'closed', width, height, fps) for one camera index.
    Reads are retried until `timeout_s` is used up: the iPhone (Continuity
    Camera) opens at once but delivers its first frames only after seconds."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return "closed", 0, 0, 0.0
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "no-frame", 0, 0, cap.get(cv2.CAP_PROP_FPS) or 0.0
            ok, frame = _read_with_timeout(cap, min(2.0, remaining))
            if ok and frame is not None:
                h, w = frame.shape[:2]
                return "ok", w, h, cap.get(cv2.CAP_PROP_FPS) or 0.0
            time.sleep(0.3)
    finally:
        cap.release()


def list_sources(max_index: int = 4, timeout_s: float = 2.0) -> list[tuple[int, str, int, int, float]]:
    """(index, status, width, height, fps) for indices 0..max_index. On macOS the
    iPhone shows up as an extra index when Continuity Camera is active; it can
    open and still deliver nothing until the phone wakes up."""
    return [(i, *probe_source(i, timeout_s)) for i in range(max_index + 1)]


def auto_source(max_index: int = 4, timeout_s: float = 15.0) -> int | None:
    """First index that delivers a frame, waiting up to `timeout_s` per index
    (the phone needs a few seconds after waking)."""
    for i in range(max_index + 1):
        status, _w, _h, _fps = probe_source(i, timeout_s)
        log.info("camera %d: %s", i, status)
        if status == "ok":
            return i
    return None


class Capture:
    """VideoCapture that never gives up: on a read failure it reports None,
    retries every `retry_s`, and reopens the device after `reopen_s` of
    silence. A file reports `eof` at its end instead."""

    def __init__(self, src: str) -> None:
        self.src = src
        self.is_cam = src.isdigit()
        self.cap: cv2.VideoCapture | None = None
        self.fps = 30.0
        self.eof = False
        self.failing_since: float | None = None
        self.last_attempt = 0.0
        self.retry_s, self.reopen_s = 0.5, 3.0
        self.reopens = 0
        self.open()
        if self.cap is None or not self.cap.isOpened():
            raise SystemExit(f"cannot open source {src!r}")

    def open(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(int(self.src)) if self.is_cam else cv2.VideoCapture(self.src)
        if self.is_cam and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.isOpened() else 0.0
        self.fps = fps if 1 < fps <= 240 else 30.0

    @property
    def healthy(self) -> bool:
        return self.failing_since is None

    def read(self):
        """Next frame or None (camera silent / file ended)."""
        now = time.monotonic()
        if not self.healthy and now - self.last_attempt < self.retry_s:
            return None
        self.last_attempt = now
        if not self.healthy and now - self.failing_since >= self.reopen_s:
            log.warning("source %s silent for %.0f s: reopening", self.src, now - self.failing_since)
            self.open()
            self.reopens += 1
            self.failing_since = now
        ok, frame = (self.cap.read() if self.cap is not None and self.cap.isOpened() else (False, None))
        if ok and frame is not None:
            if not self.healthy:
                log.info("source %s delivers frames again", self.src)
            self.failing_since = None
            return frame
        if not self.is_cam:
            self.eof = True
            return None
        if self.failing_since is None:
            self.failing_since = now
            log.warning("source %s: no frame", self.src)
        return None

    def grab(self) -> None:
        if self.cap is not None:
            self.cap.grab()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


HOTKEYS = {ord("1"): (0, 2), ord("2"): (1, 2), ord("3"): (0, 3), ord("4"): (1, 3)}


def handle_key(key: int, board: ScoreBoard, t: float) -> tuple[str, bool] | None:
    """Apply a hotkey to the board; returns (flash label, made-style) or None.
    1/2 = +2 team A/B, 3/4 = +3, z = undo."""
    if key in HOTKEYS:
        team, pts = HOTKEYS[key]
        return board.manual(team, pts, t).label, True
    if key == ord("z"):
        act = board.undo(t)
        return (act.label, False) if act else None
    return None


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
    if args.list_sources:
        for i, status, w, h, fps in list_sources():
            detail = f"{w}x{h} @ {fps:g} fps" if status == "ok" else status
            print(f"--source {i}: {detail}")
        return 0
    if args.source is None:
        print("--source is required (camera index, auto, or video path); --list-sources probes cameras", file=sys.stderr)
        return 2
    if args.source == "auto":
        idx = auto_source()
        if idx is None:
            print("no camera delivers frames (try --list-sources, wake the phone for Continuity Camera)", file=sys.stderr)
            return 2
        log.info("auto source: camera %d", idx)
        args.source = str(idx)
    load_dotenv(".env")

    cap = Capture(args.source)
    is_cam, src_fps = cap.is_cam, cap.fps
    if args.replay:
        from vision.live.replay import ReplayTracker

        tracker = ReplayTracker(args.replay, fps=src_fps)
        log.info("replaying tracks from %s (no models)", args.replay)
    else:
        from vision.track.tracker import Tracker  # TRACK's per-frame API (loads torch/ultralytics)

        tracker = Tracker(args.weights_players, args.weights_ballhoop, args.device, weights=args.weights, fps=src_fps)
    engine = StatsEngine(dt=1.0 / args.process_fps, fps=src_fps)
    events: queue.Queue = queue.Queue()
    worker = Worker(tracker, engine, events)
    worker.start()
    board = ScoreBoard()

    frame = None
    for _ in range(60 if is_cam else 20):  # the iPhone needs up to ~15 s for its first frame; a file delivers at once
        frame = cap.read()
        if frame is not None:
            break
        time.sleep(0.25)
    if frame is None:
        raise SystemExit(f"no frames from source {args.source!r} (try --list-sources)")
    h, w = frame.shape[:2]
    out_w = min(args.out_width, w)
    out_h = int(round(h * out_w / w / 2) * 2)
    minimap = None
    if args.minimap != "off":
        calib_path = args.calib
        if calib_path is None:
            per_clip = Path("out") / f"court_calib_{Path(args.source).stem}.json"
            calib_path = str(per_clip) if per_clip.exists() else "out/court_calib.json"
        minimap = MiniMap(try_load_calibration(calib_path), numbers=load_numbers(args.identities))
        if minimap.cal is None:
            log.info("no court calibration at %s: minimap shows 'uncalibrated'", calib_path)
    if args.minimap == "panel":  # video keeps its width, the panel adds a third on the right
        out_w = int(round(out_w * 1.5 / 2) * 2)

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
    map_window = "FollowCam court"
    if not args.no_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, out_w, out_h)
        if args.minimap == "window":
            cv2.namedWindow(map_window, cv2.WINDOW_NORMAL)

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

    status = ""  # shown in the score bar when the source misbehaves
    panel = None
    try:
        while True:
            fresh = True
            if idx > 0:
                if realtime and not is_cam and not cap.eof:
                    # pace like a camera; drop frames if rendering falls behind
                    target = t_start + idx / src_fps
                    lag = time.monotonic() - target
                    if lag > 2.0 / src_fps:
                        cap.grab()
                        idx += 1
                        continue
                    if lag < 0:
                        time.sleep(-lag)
                nxt = None if cap.eof else cap.read()
                if nxt is None:
                    fresh = False  # keep showing the last frame; never leave the stage dark
                    if cap.eof:
                        status = "Ende der Datei"
                        if args.no_window or not realtime:
                            break
                    else:
                        status = "Kamera: kein Bild"
                    time.sleep(0.05)
                else:
                    frame, status = nxt, ""
            t = (time.monotonic() - t_start) if is_cam else idx / src_fps
            if args.max_seconds and t > args.max_seconds:
                break

            if fresh and realtime:
                worker.offer(frame, idx, t)  # worker takes it when free
            elif fresh and idx % stride == 0:
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
            if court_lines is not None and minimap is not None and minimap.cal is not None and not args.no_court_lines:
                court_lines(view, None if is_cam else idx, minimap.cal)
            draw_tracks(view, worker.latest, worker.holder)
            info = f"det {1 / max(worker.proc_dt, 1e-3):.1f} fps   1/2 +2  3/4 +3  z undo  q quit"
            if status:
                info = f"{status}   |   {info}"
            draw_score_bar(view, board, t, info, warning=bool(status))
            if flash:
                draw_flash(view, flash[0], flash[1], t - flash[2], FLASH_S)
            if minimap and (panel is None or frames_rendered % max(args.panel_every, 1) == 0):
                panel = minimap.render(worker.latest, worker.holder)
            if args.minimap == "panel":
                small = compose_side_by_side(view, panel, out_w, out_h)
            else:
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
                if args.minimap == "window":
                    cv2.imshow(map_window, panel)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                hit = handle_key(key, board, t)
                if hit:
                    flash = (hit[0], hit[1], t)
            if fresh:
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
            "source_reopens": cap.reopens,
        }
        Path(args.events_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.events_out).write_text(json.dumps(summary, indent=1))
        log.info("done: rendered %d, processed %d, shots %d, score %s -> %s", frames_rendered,
                 worker.processed, len(shots_log), board.line(), args.events_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
