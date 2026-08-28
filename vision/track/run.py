"""TRACK batch wrapper: video → out/tracks.jsonl, out/tracks_meta.json, out/overlay.mp4.

    .venv/bin/python vision/track/run.py --video data/clips/dev60.mp4 --stride 2

The per-frame logic is vision/track/tracker.py (Tracker.step); this file only
decodes the video, samples frames for the kmeans team fit, writes the files
and the overlay (vision/track/overlay.py) and prints progress.

Consumers (STATS, NUMBERS, COURT, the dashboard) watch the contract paths in
out/, so a 20-minute run must not truncate them in place: everything is
written to out/<clip>/ (kept as an archive) and copied onto the contract
paths atomically (os.replace) only when the run is complete, meta last. Live
progress for the monitor board is out/overlay_latest.jpg plus the log.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.track.overlay import OverlayWriter  # noqa: E402
from vision.track.tracker import TRACKERS, Tracker  # noqa: E402,F401

log = logging.getLogger("track")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--weights", type=Path, default=None,
                   help="single model with contract classes (LABEL's best.pt); "
                        "default is the two-model setup below")
    p.add_argument("--person-weights", type=Path, default=Path("models/yolo11s.pt"))
    p.add_argument("--ball-weights", type=Path, default=Path("models/ball_hoop_avishah.pt"))
    p.add_argument("--person-imgsz", type=int, default=960)
    p.add_argument("--ball-imgsz", type=int, default=1280)
    p.add_argument("--imgsz", type=int, default=1280, help="imgsz for --weights mode")
    p.add_argument("--out", type=Path, default=Path("out/tracks.jsonl"),
                   help="contract path, written atomically at the end")
    p.add_argument("--overlay", type=Path, default=Path("out/overlay.mp4"),
                   help="contract path, written atomically at the end")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="where the run writes while running (default out/<clip>_vN/, N = next "
                        "free number; older runs are never overwritten)")
    p.add_argument("--no-publish", action="store_true",
                   help="leave the results in the work dir, do not touch the contract paths")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--events", type=Path, default=Path("out/events.json"),
                   help="STATS output; shot flashes + off_court ids are used if it exists")
    p.add_argument("--identities", type=Path, default=Path("out/identities.json"),
                   help="NUMBERS output; jersey numbers as labels if it exists")
    p.add_argument("--calib", type=Path, default=Path("out/court_calib.json"),
                   help="COURT output; only players on the court are drawn if it exists")
    p.add_argument("--device", default="mps")
    p.add_argument("--conf-player", type=float, default=0.3)
    p.add_argument("--conf-ball", type=float, default=0.45)
    p.add_argument("--conf-hoop", type=float, default=0.3)
    p.add_argument("--ball-max-px", type=int, default=80,
                   help="ball boxes wider/taller than this are not the ball")
    p.add_argument("--hoop-hold", type=int, default=50,
                   help="source frames a hoop is carried forward when not detected")
    p.add_argument("--tracker", choices=sorted(TRACKERS), default="bytetrack",
                   help="botsort = camera motion compensation, measured 10x slower "
                        "and no fewer id switches")
    p.add_argument("--team-mode", choices=["rules", "kmeans"], default="rules",
                   help="rules = blue vs black/red vs grey referee (this game); "
                        "kmeans = generic two-color split")
    p.add_argument("--team-samples", type=int, default=24,
                   help="kmeans mode: frames sampled across the clip for the fit")
    p.add_argument("--cuts", type=Path, default=None,
                   help="COURT's cut list (default out/cuts_<clip>.json if present); "
                        "tracker state is reset at every cut frame")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    return p.parse_args()


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, n, w, h


def sample_frames(path: Path, first: int, last: int, n: int):
    cap = cv2.VideoCapture(str(path))
    for i in np.linspace(first, max(first, last - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            yield frame
    cap.release()


def load_cuts(path: Path | None, clip: Path) -> list[int]:
    """Cut frames from COURT's file. Accepts [f, ...], [{"frame": f}, ...],
    {"cuts": [...]} or {"segments": [{"start": f}, ...]}."""
    if path is None:
        path = Path("out") / f"cuts_{clip.stem}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("cuts") or data.get("segments") or []
    frames = []
    for c in data:
        if isinstance(c, dict):
            v = c.get("frame", c.get("start"))
        else:
            v = c
        if v is not None:
            frames.append(int(v))
    log.info("%d cuts from %s", len(frames), path)
    return sorted(set(frames))


def publish(src: Path, dst: Path) -> None:
    """Copy src next to dst, then rename over dst: readers never see a partial file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def next_work_dir(clip: Path) -> Path:
    """out/<clip>_vN with the next free N; every run keeps its own archive."""
    out = Path("out")
    used = [1] if (out / clip.stem).exists() else [0]
    for d in out.glob(f"{clip.stem}_v*"):
        m = re.fullmatch(r"v(\d+)", d.name[len(clip.stem) + 1:])
        if m:
            used.append(int(m.group(1)))
    return out / f"{clip.stem}_v{max(used) + 1}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    a = parse_args()
    work = a.work_dir or next_work_dir(a.video)
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"{work} exists and is not empty; runs never overwrite each other")
    work.mkdir(parents=True, exist_ok=True)
    w_tracks, w_meta = work / "tracks.jsonl", work / "tracks_meta.json"
    w_summary, w_overlay = work / "track_summary.json", work / "overlay.mp4"
    w_rejects = work / "ball_rejects.jsonl"
    cap, fps, n_frames, width, height = open_video(a.video)
    first = a.start_frame
    last = n_frames if a.max_frames == 0 else min(n_frames, first + a.max_frames * a.stride)
    log.info("%s: %dx%d @ %.2f fps, %d frames, processing %d..%d stride %d",
             a.video, width, height, fps, n_frames, first, last, a.stride)

    tr = Tracker(a.person_weights, a.ball_weights, a.device, weights=a.weights,
                 person_imgsz=a.person_imgsz, ball_imgsz=a.ball_imgsz, imgsz=a.imgsz,
                 conf_player=a.conf_player, conf_ball=a.conf_ball, conf_hoop=a.conf_hoop,
                 ball_max_px=a.ball_max_px, hoop_hold=a.hoop_hold, tracker=a.tracker,
                 team_mode=a.team_mode, fps=fps)
    if a.team_mode == "kmeans":
        tr.fit_teams(sample_frames(a.video, first, last, a.team_samples))
        log.info("jersey centroids LAB %s", tr.teams.centroids_lab)
    else:
        log.info("team mode: rules (blue → 0, black/red → 1, grey/white → -1)")

    writer = None
    if not a.no_overlay:
        writer = OverlayWriter(w_overlay, width=width, height=height, fps=fps / a.stride,
                               events=a.events, identities=a.identities, calib=a.calib,
                               cuts=a.cuts or Path("out") / f"cuts_{a.video.stem}.json",
                               source_fps=fps, latest_path=a.overlay.with_name("overlay_latest.jpg"))

    meta = {"clip": str(a.video), "source_fps": fps, "stride": a.stride, "fps": fps / a.stride,
            "width": width, "height": height, "first_frame": first, "last_frame": last,
            "tracks": str(a.out), "weights": tr.weights_info,
            "cuts": load_cuts(a.cuts, a.video), "work_dir": str(work)}
    w_meta.write_text(json.dumps(meta, indent=1))
    log.info("writing to %s, publishing to %s when complete", work, a.out.parent)

    cuts = load_cuts(a.cuts, a.video)
    # Graceful stop: `touch out/<clip>/STOP` or SIGTERM ends the loop and still
    # publishes what was processed (a kill leaves an unreadable mp4).
    stop_file = work / "STOP"
    stop_file.unlink(missing_ok=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    t0 = time.time()
    done = 0
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    with w_tracks.open("w") as out, w_rejects.open("w") as rej:
        idx = first
        while idx < last:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - first) % a.stride == 0:
                if stop["flag"] or (done % 25 == 0 and stop_file.exists()):
                    log.warning("stopping early at frame %d (STOP file or SIGTERM)", idx)
                    break
                if cuts and cuts[0] <= idx:
                    while cuts and cuts[0] <= idx:
                        cuts.pop(0)
                    tr.reset()
                record = tr.step(frame, idx, idx / fps)
                out.write(json.dumps(record) + "\n")
                for x in tr.last_rejects:
                    rej.write(json.dumps({"frame": idx, "t": record["t"], **x}) + "\n")
                if writer:
                    writer.write(frame, record)
                done += 1
                if done % 25 == 0:
                    out.flush()  # NUMBERS' watcher and the monitor board read while we run
                if done % 250 == 0:
                    s = tr.summary()
                    log.info("frame %d/%d  %.3f s/frame  players/frame %.1f  ball %.0f%%  "
                             "hoop %.0f%%  ids %d", idx, last, (time.time() - t0) / done,
                             s["players_per_frame"], 100 * s["ball_frame_share"],
                             100 * s["hoop_frame_share"], s["track_ids"])
            idx += 1
    cap.release()
    if writer:
        writer.close()

    el = time.time() - t0
    summary = {**meta, **tr.summary(), "seconds": round(el, 1),
               "s_per_frame": round(el / max(done, 1), 3), "tracker": a.tracker,
               "stopped_early_at": idx if (stop["flag"] or stop_file.exists()) else None,
               "overlay": None if a.no_overlay else str(a.overlay)}
    w_summary.write_text(json.dumps(summary, indent=1))
    if not a.no_publish:
        publish(w_tracks, a.out)
        publish(w_summary, a.out.parent / "track_summary.json")
        if writer:
            publish(w_overlay, a.overlay)
        publish(w_meta, a.out.parent / "tracks_meta.json")  # meta last = "complete"
        log.info("published %s, %s, %s", a.out, a.overlay, a.out.parent / "tracks_meta.json")
    log.info("done: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
