"""Annotated video from tracks.jsonl: team-colored boxes with jersey numbers,
ball trail, hoop, shot flashes. Runs standalone on CPU, no re-detection:

    .venv/bin/python vision/track/overlay.py --video data/clips/dev60.mp4 \
        --tracks out/tracks.jsonl --out out/overlay.mp4 \
        [--identities out/identities.json] [--calib out/court_calib.json] \
        [--events out/events.json] [--cuts out/cuts_dev60.json]

What is drawn, and why:
* players: only those on the court. With out/court_calib.json the foot point
  is projected (vision.court.draw.on_court_px, court plus 1 m tolerance);
  without it, STATS's "off_court_track_ids" from events.json are hidden.
  Bench, coaches and spectators otherwise get boxes too. `--court-lines`
  additionally draws COURT's calibrated court lines.
* label: jersey number from out/identities.json as "#12" in the team color;
  tracks without a number get a small grey track id.
* ball: circle only when a ball is in this line, never a stale point. Trail =
  last 25 positions, broken at gaps > 0.5 s and at cuts, so it never draws a
  straight line across the hall.
* hoop: green box; shot flashes ("MADE"/"MISS") from events.json for 1 s.
Written with cv2.VideoWriter (mp4v) to a temp file, then transcoded to H.264
(yuv420p, faststart) so the dashboard can embed it. Every 100 frames the
current frame is saved atomically as <out dir>/overlay_latest.jpg.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger("track")

TEAM_COLORS = {  # BGR. Fixed and high-contrast on purpose: the real jersey
    -1: (210, 210, 210),  # colors (blue vs black) are too close to tell apart.
    0: (255, 140, 0),  # team 0 = the bluer jersey (see teams.py)
    1: (0, 0, 255),
}
BALL_COLOR = (0, 220, 255)
HOOP_COLOR = (0, 255, 120)
TRAIL_LEN = 25
TRAIL_GAP_S = 0.5
FLASH_S = 1.0


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("%s is not valid JSON, ignored", path)
        return {}


def load_identities(path: Path | None) -> dict[int, dict]:
    data = load_json(path)
    return {int(k): v for k, v in data.get("tracks", {}).items()}


def load_cut_times(path: Path | None, fps: float) -> list[float]:
    data = load_json(path)
    cuts = data.get("cuts") or []
    return sorted(float(c) / fps if not isinstance(c, dict) else float(c.get("frame", 0)) / fps
                  for c in cuts)


class OverlayWriter:
    def __init__(self, path: Path, *, width: int, height: int, fps: float,
                 events: Path | None = None, identities: Path | None = None,
                 calib: Path | None = None, cuts: Path | None = None,
                 source_fps: float = 50.0, court_lines: bool = False,
                 latest_path: Path | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.raw_path = path.with_name(path.stem + "_raw.mp4")
        self.latest_path = latest_path or path.with_name("overlay_latest.jpg")
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_every = 100
        self.writer = cv2.VideoWriter(
            str(self.raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open VideoWriter for {self.raw_path}")
        self.fps = fps
        self.frames = 0
        self.trail: deque[tuple[float, tuple[int, int]]] = deque(maxlen=TRAIL_LEN)

        ev = load_json(events)
        self.shots = ev.get("shots", [])
        self.off_court = {int(i) for i in ev.get("off_court_track_ids", [])}
        self.identities = load_identities(identities)
        self.cut_times = load_cut_times(cuts, source_fps)
        self.calib = None
        self.court_lines = court_lines
        if calib is not None and calib.exists():
            try:
                from vision.court import draw as court_draw
                from vision.court.project import load_calibration

                self.calib = load_calibration(calib)
                self._court = court_draw
                log.info("court filter from %s (%s)", calib, self.calib.mode)
            except Exception as e:  # noqa: BLE001
                log.warning("calibration %s unusable (%s), falling back to off_court ids", calib, e)
        log.info("overlay: %d identities, %d off-court ids, %d shots, %d cuts",
                 len(self.identities), len(self.off_court), len(self.shots), len(self.cut_times))

    # ----- filters -----------------------------------------------------------
    def visible_players(self, record: dict) -> list[dict]:
        players = record["players"]
        if self.calib is not None and players:
            feet = np.array([p["foot"] for p in players], np.float64)
            try:
                ok = self._court.on_court_px(self.calib, record["frame"], feet)
                return [p for p, keep in zip(players, ok) if keep]
            except Exception as e:  # noqa: BLE001
                log.debug("projection failed at frame %d: %s", record["frame"], e)
        return [p for p in players if p["id"] not in self.off_court]

    def label_for(self, p: dict) -> tuple[str, int, bool]:
        """(text, team, has_number)."""
        ident = self.identities.get(p["id"])
        if ident and ident.get("number") is not None:
            team = ident.get("team", p["team"])
            return f"#{ident['number']}", team if team is not None else p["team"], True
        return str(p["id"]), p["team"], False

    # ----- drawing -----------------------------------------------------------
    def write(self, frame: np.ndarray, record: dict) -> None:
        img = frame.copy()
        t = record["t"]
        if self.court_lines and self.calib is not None:
            try:
                img = self._court.court_lines(img, record["frame"], self.calib)
            except Exception as e:  # noqa: BLE001
                log.debug("court lines failed at frame %d: %s", record["frame"], e)
        for p in self.visible_players(record):
            text, team, has_number = self.label_for(p)
            color = TEAM_COLORS.get(team, TEAM_COLORS[-1])
            x1, y1, x2, y2 = (int(v) for v in p["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            fx, fy = (int(v) for v in p["foot"])
            cv2.circle(img, (fx, fy), 3, color, -1)
            if has_number:
                self._tag(img, text, x1, y1, color, scale=0.7, thick=2)
            else:
                self._tag(img, text, x1, y1, (140, 140, 140), scale=0.45, thick=1)

        for h in record["hoops"]:
            x1, y1, x2, y2 = (int(v) for v in h["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), HOOP_COLOR, 2)

        ball = record["ball"]
        if any(self.trail) and self.trail[-1][0] < t and self._cut_between(self.trail[-1][0], t):
            self.trail.clear()
        if ball:
            cx, cy = (int(v) for v in ball["center"])
            self.trail.append((t, (cx, cy)))
        elif self.trail:
            self.trail.popleft()  # fade instead of freezing on a stale point
        pts = list(self.trail)
        for i in range(1, len(pts)):
            if pts[i][0] - pts[i - 1][0] > TRAIL_GAP_S:
                continue
            a = (i + 1) / len(pts)
            cv2.line(img, pts[i - 1][1], pts[i][1], BALL_COLOR, max(1, int(4 * a)))
        if ball:
            cv2.circle(img, pts[-1][1], 8, BALL_COLOR, 2)

        self._flash(img, t)
        self._hud(img, record)
        self.writer.write(img)
        self.frames += 1
        if self.frames % self.latest_every == 1:
            tmp = self.latest_path.with_suffix(".tmp.jpg")  # never a half-written file
            cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            tmp.replace(self.latest_path)

    def _cut_between(self, t0: float, t1: float) -> bool:
        return any(t0 < c <= t1 for c in self.cut_times)

    @staticmethod
    def _tag(img, text, x, y, color, *, scale, thick) -> None:
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        y0 = max(y - h - base - 4, 0)
        cv2.rectangle(img, (x, y0), (x + w + 6, y0 + h + base + 4), color, -1)
        cv2.putText(img, text, (x + 3, y0 + h + 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thick, cv2.LINE_AA)

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
        self._transcode()

    def _transcode(self) -> None:
        """mp4v → H.264 for the browser. Falls back to the raw file on failure."""
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:  # noqa: BLE001
            log.warning("no ffmpeg (%s), keeping mp4v overlay", e)
            shutil.move(self.raw_path, self.path)
            return
        tmp = self.path.with_name(self.path.stem + "_h264.tmp.mp4")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(self.raw_path), "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-an", str(tmp)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not tmp.exists():
            log.warning("transcode failed: %s; keeping mp4v overlay", res.stderr.strip()[:300])
            shutil.move(self.raw_path, self.path)
            return
        tmp.replace(self.path)
        self.raw_path.unlink(missing_ok=True)
        log.info("overlay H.264 written: %s", self.path)


# ----- standalone re-render ---------------------------------------------------
def render(video: Path, tracks: Path, out: Path, *, identities: Path | None, calib: Path | None,
           events: Path | None, cuts: Path | None, max_frames: int = 0,
           court_lines: bool = False) -> int:
    records = [json.loads(l) for l in tracks.read_text().splitlines() if l.strip()]
    if not records:
        raise SystemExit(f"{tracks} is empty")
    meta = load_json(tracks.with_name("tracks_meta.json"))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    source_fps = float(meta.get("source_fps") or cap.get(cv2.CAP_PROP_FPS) or 50.0)
    stride = int(meta.get("stride") or (records[1]["frame"] - records[0]["frame"] if len(records) > 1 else 1))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if cuts is None:
        cuts = Path("out") / f"cuts_{video.stem}.json"
    w = OverlayWriter(out, width=width, height=height, fps=source_fps / stride, events=events,
                      identities=identities, calib=calib, cuts=cuts, source_fps=source_fps,
                      court_lines=court_lines)
    by_frame = {r["frame"]: r for r in records}
    first, last = records[0]["frame"], records[-1]["frame"]
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    idx, done = first, 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        r = by_frame.get(idx)
        if r is not None:
            w.write(frame, r)
            done += 1
            if max_frames and done >= max_frames:
                break
        idx += 1
    cap.release()
    w.close()
    log.info("rendered %d frames to %s", done, out)
    return done


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="re-render overlay.mp4 from tracks.jsonl (CPU only)")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--tracks", type=Path, default=Path("out/tracks.jsonl"))
    p.add_argument("--out", type=Path, default=Path("out/overlay.mp4"))
    p.add_argument("--identities", type=Path, default=Path("out/identities.json"))
    p.add_argument("--calib", type=Path, default=Path("out/court_calib.json"))
    p.add_argument("--events", type=Path, default=Path("out/events.json"))
    p.add_argument("--cuts", type=Path, default=None, help="default out/cuts_<clip>.json")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--court-lines", action="store_true", help="draw COURT's calibrated lines")
    a = p.parse_args()
    render(a.video, a.tracks, a.out, identities=a.identities, calib=a.calib, events=a.events,
           cuts=a.cuts, max_frames=a.max_frames, court_lines=a.court_lines)


if __name__ == "__main__":
    main()
