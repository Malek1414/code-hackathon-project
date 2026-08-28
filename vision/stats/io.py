"""Read / write the `out/tracks.jsonl` contract and build synthetic fixtures.

Contract (docs/ORCHESTRATION.md): one JSON object per processed frame::

    {"frame": 1250, "t": 25.0,
     "players": [{"id": 7, "bbox": [x1,y1,x2,y2], "foot": [x,y], "team": 0, "conf": 0.91}],
     "ball": {"bbox": [x1,y1,x2,y2], "center": [x,y], "conf": 0.6} | null,
     "hoops": [{"bbox": [x1,y1,x2,y2]}]}

All coordinates are pixels, y grows downward. The reader is tolerant: a
missing `foot` is derived from the bbox, a missing `t` from `frame / fps`,
and a truncated last line (TRACK writes while we read) is skipped.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class Player:
    id: int
    bbox: BBox
    foot: Point
    team: int = -1
    conf: float = 0.0
    on_court: bool | None = None  # TRACK's verdict if it has one; None = decide here

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> Point:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def edge_distance(self, p: Point) -> float:
        """Distance from `p` to the nearest point of the bbox (0 inside)."""
        dx = max(self.bbox[0] - p[0], 0.0, p[0] - self.bbox[2])
        dy = max(self.bbox[1] - p[1], 0.0, p[1] - self.bbox[3])
        return math.hypot(dx, dy)


@dataclass(frozen=True)
class Ball:
    center: Point
    bbox: BBox | None = None
    conf: float = 0.0


@dataclass
class Frame:
    frame: int
    t: float
    players: list[Player] = field(default_factory=list)
    ball: Ball | None = None
    hoops: list[BBox] = field(default_factory=list)

    def player(self, player_id: int) -> Player | None:
        for p in self.players:
            if p.id == player_id:
                return p
        return None


# --- reading -----------------------------------------------------------------


def _bbox(v) -> BBox:
    x1, y1, x2, y2 = (float(a) for a in v)
    return (x1, y1, x2, y2)


def frame_from_dict(d: dict, fps: float | None = None) -> Frame:
    frame_no = int(d["frame"])
    t = d.get("t")
    if t is None:
        if not fps:
            raise ValueError(f"frame {frame_no} has no 't' and no fps was given")
        t = frame_no / fps

    players = []
    for p in d.get("players") or []:
        bbox = _bbox(p["bbox"])
        foot = p.get("foot")
        foot_pt = (float(foot[0]), float(foot[1])) if foot else ((bbox[0] + bbox[2]) / 2, bbox[3])
        players.append(
            Player(
                id=int(p["id"]),
                bbox=bbox,
                foot=foot_pt,
                team=int(p.get("team", -1)),
                conf=float(p.get("conf", 0.0)),
                on_court=p.get("on_court"),
            )
        )

    ball = None
    b = d.get("ball")
    if b:
        bbox = _bbox(b["bbox"]) if b.get("bbox") else None
        c = b.get("center")
        if c is None and bbox is not None:
            c = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        if c is not None:
            ball = Ball(center=(float(c[0]), float(c[1])), bbox=bbox, conf=float(b.get("conf", 0.0)))

    hoops = [_bbox(h["bbox"] if isinstance(h, dict) else h) for h in d.get("hoops") or []]
    # Each end has a small folded side hoop on the wall; the real one is the largest box.
    hoops.sort(key=lambda h: (h[2] - h[0]) * (h[3] - h[1]), reverse=True)
    return Frame(frame=frame_no, t=float(t), players=players, ball=ball, hoops=hoops[:1])


def read_tracks(path: str | Path, fps: float | None = None) -> list[Frame]:
    """Read tracks.jsonl. Sorted by frame, duplicates keep the last record."""
    by_frame: dict[int, Frame] = {}
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # partial last line: writer is still busy
            raise
        fr = frame_from_dict(d, fps)
        by_frame[fr.frame] = fr
    return [by_frame[k] for k in sorted(by_frame)]


MAX_BALL_ASPECT = 1.8  # a ball box is roughly square (motion blur stretches it to ~1.5); signs are longer
MAX_BALL_SPEED_DIAMETERS_S = 45.0  # ~10 m/s for a 0.24 m ball
BALL_DIAMETER_FALLBACK_PX = 20.0


def is_round_ball(bbox: BBox) -> bool:
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return min(w, h) <= 0 or max(w, h) / min(w, h) <= MAX_BALL_ASPECT


def plausible_move(a: Frame, b: Frame, max_gap_s: float = 0.6) -> bool:
    """Could the ball have travelled from a to b? (After a long gap: yes.)"""
    dt = b.t - a.t
    if dt <= 0 or dt > max_gap_s:
        return True
    d = math.hypot(b.ball.center[0] - a.ball.center[0], b.ball.center[1] - a.ball.center[1])
    diam = BALL_DIAMETER_FALLBACK_PX
    if a.ball.bbox is not None:
        diam = max((a.ball.bbox[2] - a.ball.bbox[0] + a.ball.bbox[3] - a.ball.bbox[1]) / 2, 4.0)
    return d / dt <= MAX_BALL_SPEED_DIAMETERS_S * diam


def clean_ball(frames: list[Frame], *, max_gap_s: float = 0.6) -> int:
    """Drop stray ball detections in place, returns how many were dropped.

    Two filters, both from reference/thirdparty/utils.py (clean_ball_pos):
    boxes that are clearly not round, and isolated points that would need an
    impossible speed from the previous kept detection *and* to the next raw
    detection. A jump that the next detection confirms is kept (the ball
    really was somewhere else, e.g. after a long occlusion). StatsEngine
    applies the same rules incrementally.
    """
    dropped = 0
    for fr in frames:
        if fr.ball is not None and fr.ball.bbox is not None and not is_round_ball(fr.ball.bbox):
            fr.ball = None
            dropped += 1
    idx = [i for i, fr in enumerate(frames) if fr.ball is not None]
    last_kept: int | None = None
    for n, i in enumerate(idx):
        fr = frames[i]
        nxt = frames[idx[n + 1]] if n + 1 < len(idx) else None
        ok_prev = last_kept is None or plausible_move(frames[last_kept], fr, max_gap_s)
        ok_next = nxt is None or plausible_move(fr, nxt, max_gap_s)
        if ok_prev or ok_next:
            last_kept = i
        else:
            fr.ball = None
            dropped += 1
    return dropped


def infer_fps(frames: list[Frame]) -> float | None:
    """Frames per second from frame/t pairs (works on subsampled tracks too)."""
    if len(frames) < 2:
        return None
    rates = []
    for a, b in zip(frames, frames[1:]):
        if b.t > a.t and b.frame > a.frame:
            rates.append((b.frame - a.frame) / (b.t - a.t))
    return round(statistics.median(rates), 3) if rates else None  # 50.0, 29.97, ...


def median_dt(frames: list[Frame]) -> float:
    """Typical spacing between processed frames in seconds."""
    gaps = [b.t - a.t for a, b in zip(frames, frames[1:]) if b.t > a.t]
    return statistics.median(gaps) if gaps else 0.0


# --- writing -----------------------------------------------------------------


def frame_to_dict(fr: Frame) -> dict:
    return {
        "frame": fr.frame,
        "t": round(fr.t, 4),
        "players": [
            {
                "id": p.id,
                "bbox": [round(v, 1) for v in p.bbox],
                "foot": [round(v, 1) for v in p.foot],
                "team": p.team,
                "conf": round(p.conf, 3),
                "on_court": p.on_court,
            }
            for p in fr.players
        ],
        "ball": (
            None
            if fr.ball is None
            else {
                "bbox": [round(v, 1) for v in fr.ball.bbox] if fr.ball.bbox else None,
                "center": [round(v, 1) for v in fr.ball.center],
                "conf": round(fr.ball.conf, 3),
            }
        ),
        "hoops": [{"bbox": [round(v, 1) for v in h]} for h in fr.hoops],
    }


def write_tracks(frames: Iterable[Frame], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for fr in frames:
            f.write(json.dumps(frame_to_dict(fr)) + "\n")


# --- synthetic fixture -------------------------------------------------------

FIXTURE_HOOP: BBox = (1470.0, 280.0, 1530.0, 320.0)  # rim line = y 280, x-range 1470..1530
FIXTURE_PLAYERS = {
    # id: (team, foot x, foot y, width, height)
    1: (0, 400.0, 800.0, 80.0, 200.0),  # passer
    2: (0, 900.0, 820.0, 80.0, 200.0),  # shooter
    3: (1, 1300.0, 780.0, 80.0, 200.0),  # defender under the basket
}
FIXTURE_SHOOTER_ID = 2
FIXTURE_SHOT_START_S = 2.5
FIXTURE_SHOT_END_S = 3.7  # ball reaches the rim


def _lerp(a: float, b: float, s: float) -> float:
    return a + (b - a) * s


def _ball_path(kind: str, t: float) -> Point | None:
    """Ball center at time t for scenario `kind` (made | miss | pass)."""
    a_hand = (430.0, 700.0)  # in player 1's hands
    b_hand = (880.0, 720.0)  # in player 2's hands
    if t < 1.0:
        return a_hand
    if t < 1.5:  # pass 1 -> 2
        s = (t - 1.0) / 0.5
        return (_lerp(a_hand[0], b_hand[0], s), _lerp(a_hand[1], b_hand[1], s))
    if kind == "pass" or t < FIXTURE_SHOT_START_S:
        return b_hand
    if t < FIXTURE_SHOT_END_S:  # arc from player 2 to the rim, apex well above it
        s = (t - FIXTURE_SHOT_START_S) / (FIXTURE_SHOT_END_S - FIXTURE_SHOT_START_S)
        x = _lerp(880.0, 1500.0, s)
        y = _lerp(700.0, 270.0, s) - 4 * 400.0 * s * (1 - s)
        return (x, y)
    if kind == "made":
        if t < 4.0:  # through the net: straight down inside the hoop x-range
            s = (t - FIXTURE_SHOT_END_S) / 0.3
            return (1500.0, _lerp(275.0, 400.0, s))
        if t < 5.0:  # drops to the floor near the defender
            s = (t - 4.0) / 1.0
            return (_lerp(1500.0, 1350.0, s), _lerp(400.0, 850.0, s))
        return (1350.0, 850.0)
    # miss: rim bounce up and away, then to the floor
    if t < 3.9:
        s = (t - FIXTURE_SHOT_END_S) / 0.2
        return (_lerp(1500.0, 1400.0, s), _lerp(270.0, 200.0, s))
    if t < 4.7:
        s = (t - 3.9) / 0.8
        return (_lerp(1400.0, 1250.0, s), _lerp(200.0, 850.0, s))
    return (1250.0, 850.0)


def synthetic_scenario(
    kind: str = "made",
    *,
    fps: float = 50.0,
    duration_s: float = 6.0,
    ball_dropout: float = 0.0,
    hoop_dropout: float = 0.0,
    jitter_px: float = 0.0,
    seed: int = 0,
) -> list[Frame]:
    """Two team-0 players and a defender. Player 1 passes to player 2, who
    shoots at the hoop; the ball drops through (`made`), rattles out (`miss`)
    or is never shot (`pass`). `ball_dropout` / `hoop_dropout` randomly hide
    detections, `jitter_px` adds detector noise."""
    if kind not in ("made", "miss", "pass"):
        raise ValueError(f"unknown scenario {kind!r}")
    rng = random.Random(seed)
    frames: list[Frame] = []
    for n in range(int(round(duration_s * fps))):
        t = n / fps

        def jit() -> float:
            return rng.uniform(-jitter_px, jitter_px) if jitter_px else 0.0

        players = []
        for pid, (team, fx, fy, w, h) in FIXTURE_PLAYERS.items():
            fx, fy = fx + jit(), fy + jit()
            bbox = (fx - w / 2, fy - h, fx + w / 2, fy)
            players.append(Player(id=pid, bbox=bbox, foot=(fx, fy), team=team, conf=0.9))

        ball = None
        c = _ball_path(kind, t)
        if c is not None and rng.random() >= ball_dropout:
            cx, cy = c[0] + jit(), c[1] + jit()
            ball = Ball(center=(cx, cy), bbox=(cx - 12, cy - 12, cx + 12, cy + 12), conf=0.6)

        hoops = [] if rng.random() < hoop_dropout else [tuple(v + jit() for v in FIXTURE_HOOP)]
        frames.append(Frame(frame=n, t=t, players=players, ball=ball, hoops=hoops))
    return frames
