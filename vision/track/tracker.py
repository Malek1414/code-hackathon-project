"""Per-frame tracking API: one Tracker instance, one tracks.jsonl line per step.

    from vision.track.tracker import Tracker
    tr = Tracker()                       # yolo11s persons + ball/hoop model, MPS
    line = tr.step(frame_bgr, frame_index, t_seconds)

All state (ByteTrack persistence, last hoop box, ball gate, team votes and,
in kmeans mode, the centroids) lives in the instance. run.py is the batch
wrapper; vision/live uses the same object on camera frames.

Detectors (measured on MPS, 1080p50 game footage, see docs/ORCHESTRATION.md):
  * persons: yolo11s (COCO) at imgsz 960 with ByteTrack — 0.04 s/frame.
  * ball + hoop: models/ball_hoop_avishah.pt (0 Basketball, 1 Basketball Hoop)
    at imgsz 1280. Ball conf ≥ 0.45 (wall fixtures come through at 0.3-0.4),
    single best ball per frame after the gate in ball.py. Each court end shows
    the game hoop and a folded wall hoop: only the largest hoop box counts,
    carried forward at most `hoop_hold` frames (the camera pans).
  * `weights=` (LABEL's best.pt with contract classes 0 player, 1 ball,
    2 hoop, 3 referee) switches to a single model for everything.
Teams: vision/track/teams.py (rules by default, kmeans optional).

The footage is an edited production (close-ups, inserts, dissolves): call
`reset()` at every cut (COURT writes out/cuts_<clip>.json) so ByteTrack does
not glue a close-up face to a wide-shot player. Track ids stay unique across
resets (offset), so a player after a cut is a new id, never a reused one.
A hoop box must sit in the upper `hoop_max_y` of the frame and above the feet
of every player it overlaps (a false hoop was found on the floor between
players).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vision.track.ball import BallGate
from vision.track.teams import TeamClassifier, torso_color

log = logging.getLogger("track")

PLAYER, BALL, HOOP, REFEREE = 0, 1, 2, 3
CONTRACT_NAMES = {"player": PLAYER, "ball": BALL, "hoop": HOOP, "referee": REFEREE}
_HERE = Path(__file__).resolve().parent
TRACKERS = {"bytetrack": _HERE / "bytetrack.yaml", "botsort": _HERE / "botsort.yaml"}


def _round(box) -> list[float]:
    return [round(float(v), 1) for v in box]


def _iter_boxes(r):
    if not len(r.boxes):
        return
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None
    for k in range(len(xyxy)):
        yield xyxy[k], float(conf[k]), int(cls[k]), (int(ids[k]) if ids is not None else None)


class Tracker:
    def __init__(self, weights_players: str | Path = "models/yolo11s.pt",
                 weights_ballhoop: str | Path = "models/ball_hoop_avishah.pt",
                 device: str = "mps", *, weights: str | Path | None = None,
                 person_imgsz: int = 960, ball_imgsz: int = 1280, imgsz: int = 1280,
                 conf_player: float = 0.3, conf_ball: float = 0.45, conf_hoop: float = 0.5,
                 ball_max_px: int = 80, hoop_hold: int = 50, hoop_max_y: float = 0.55,
                 hoop_min_streak: int = 3,
                 tracker: str = "bytetrack", team_mode: str = "rules", fps: float = 50.0) -> None:
        from ultralytics import YOLO

        self.device = device
        self.person_imgsz, self.ball_imgsz, self.imgsz = person_imgsz, ball_imgsz, imgsz
        self.conf_player, self.conf_ball, self.conf_hoop = conf_player, conf_ball, conf_hoop
        self.ball_max_px, self.hoop_hold, self.hoop_max_y = ball_max_px, hoop_hold, hoop_max_y
        self.hoop_min_streak = hoop_min_streak
        self.hoop_streak = 0
        self.hoop_pending: list[float] | None = None
        self.tracker_yaml = str(TRACKERS[tracker])
        self.fps = fps

        self.single = weights is not None
        if self.single:
            self.model = YOLO(str(weights))
            names = self.model.names
            self.cmap = {i: CONTRACT_NAMES[n] for i, n in names.items() if n in CONTRACT_NAMES}
            if "sports ball" in names.values():  # plain COCO weights passed by hand
                self.cmap = {0: PLAYER, 32: BALL}
            if PLAYER not in self.cmap.values():
                raise ValueError(f"{weights}: classes {names} match neither contract nor COCO")
            self.person_ids = [k for k, v in self.cmap.items() if v in (PLAYER, REFEREE)]
            self.person_model = self.model
            self.weights_info = str(weights)
            log.info("single model %s, class map %s", weights, self.cmap)
        else:
            self.person_model = YOLO(str(weights_players))
            self.ball_model = YOLO(str(weights_ballhoop))
            bn = {n.lower(): i for i, n in self.ball_model.names.items()}
            self.ball_cls = bn.get("basketball", bn.get("ball", 0))
            self.hoop_cls = bn.get("basketball hoop", bn.get("hoop", 1))
            self.person_ids = [0]
            self.weights_info = {"persons": str(weights_players), "ball_hoop": str(weights_ballhoop)}
            log.info("persons %s @%d, ball/hoop %s @%d %s", weights_players, person_imgsz,
                     weights_ballhoop, ball_imgsz, self.ball_model.names)

        self.teams = TeamClassifier(team_mode)
        self.gate = BallGate()
        self.last_hoop: list[float] | None = None
        self.last_hoop_frame = -10**9
        self._auto_index = 0
        self.id_offset = 0
        self.max_raw_id = 0
        self.max_emitted_id = 0
        self.id_remap: dict[int, int] = {}  # raw tracker id → id after a color switch
        self.switches = 0
        self.resets = 0
        self.stats = {"frames": 0, "ball_frames": 0, "hoop_frames": 0, "hoop_fresh": 0,
                      "hoop_rejected": 0, "player_dets": 0, "ids": set()}

    def reset(self) -> None:
        """Cut in the footage: forget tracks, ball history and the hoop."""
        self.resets += 1
        self.id_offset = self.max_emitted_id
        self.max_raw_id = 0
        self.id_remap.clear()
        for m in {id(self.person_model): self.person_model}.values():
            pred = getattr(m, "predictor", None)
            if pred is not None and getattr(pred, "trackers", None):
                for t in pred.trackers:
                    t.reset()
        self.gate = BallGate(blacklist_rel=self.gate.blacklist_rel)
        self.last_hoop, self.last_hoop_frame = None, -10**9
        self.hoop_streak, self.hoop_pending = 0, None
        self.teams.reset_votes()

    # ----- detection ---------------------------------------------------------
    def person_boxes(self, frame: np.ndarray) -> np.ndarray:
        """Boxes only, no tracking: for the team-color pre-pass in kmeans mode."""
        r = self.person_model.predict(
            frame, imgsz=self.imgsz if self.single else self.person_imgsz,
            conf=self.conf_player, classes=self.person_ids, device=self.device, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy()

    def fit_teams(self, frames) -> None:
        """kmeans mode: fit on torso crops from the given frames. No-op for rules."""
        if self.teams.mode != "kmeans":
            return
        feats = [f for fr in frames for f in
                 (torso_color(fr, b) for b in self.person_boxes(fr)) if f is not None]
        log.info("team fit on %d torso crops", len(feats))
        self.teams.fit(np.array(feats, dtype=np.float32))

    def _detect(self, frame: np.ndarray):
        persons, balls, hoops = [], [], []  # (box, conf, id, is_ref), (conf, box), (conf, box)
        if self.single:
            r = self.model.track(frame, persist=True, tracker=self.tracker_yaml, imgsz=self.imgsz,
                                 conf=min(self.conf_player, self.conf_ball, self.conf_hoop),
                                 classes=sorted(self.cmap), device=self.device, verbose=False)[0]
            for box, conf, cls, tid in _iter_boxes(r):
                c = self.cmap.get(cls)
                if c in (PLAYER, REFEREE):
                    if tid is not None and conf >= self.conf_player:
                        persons.append((box, conf, tid, c == REFEREE))
                elif c == BALL and conf >= self.conf_ball:
                    balls.append((conf, box.tolist()))
                elif c == HOOP and conf >= self.conf_hoop:
                    hoops.append((conf, box.tolist()))
            return persons, balls, hoops

        r = self.person_model.track(frame, persist=True, tracker=self.tracker_yaml,
                                    imgsz=self.person_imgsz, conf=self.conf_player, classes=[0],
                                    device=self.device, verbose=False)[0]
        for box, conf, _cls, tid in _iter_boxes(r):
            if tid is not None:
                persons.append((box, conf, tid, False))
        r = self.ball_model.predict(frame, imgsz=self.ball_imgsz,
                                    conf=min(self.conf_ball, self.conf_hoop),
                                    classes=[self.ball_cls, self.hoop_cls], device=self.device,
                                    verbose=False)[0]
        for box, conf, cls, _tid in _iter_boxes(r):
            if cls == self.ball_cls and conf >= self.conf_ball:
                balls.append((conf, box.tolist()))
            elif cls == self.hoop_cls and conf >= self.conf_hoop:
                hoops.append((conf, box.tolist()))
        return persons, balls, hoops

    # ----- one frame → one tracks.jsonl line --------------------------------
    def step(self, frame: np.ndarray, frame_index: int | None = None,
             t: float | None = None) -> dict:
        if frame_index is None:
            frame_index = self._auto_index
        self._auto_index = frame_index + 1
        if t is None:
            t = frame_index / self.fps

        persons, balls, hoops = self._detect(frame)

        players = []
        for box, conf, raw_id, is_ref in persons:
            self.max_raw_id = max(self.max_raw_id, raw_id)
            tid = self.id_remap.get(raw_id, raw_id + self.id_offset)
            b = _round(box)
            team = -1
            if not is_ref:
                team, switched = self.teams.assign(tid, torso_color(frame, box))
                if switched:
                    # The box now follows another player: close this id, open a new one.
                    new_id = self.max_emitted_id + 1
                    self.teams.move_votes(tid, new_id)
                    self.id_remap[raw_id] = new_id
                    tid = new_id
                    self.switches += 1
            self.max_emitted_id = max(self.max_emitted_id, tid)
            players.append({"id": tid, "bbox": b, "foot": [round((b[0] + b[2]) / 2, 1), b[3]],
                            "team": team, "conf": round(conf, 3),
                            "on_court": None})  # STATS fills this from the calibration

        # Largest hoop box = the game hoop; the folded wall hoop is smaller.
        # Sanity: upper part of the frame, above the feet of overlapping players.
        max_y = frame.shape[0] * self.hoop_max_y
        ok_hoops = []
        for c, hb in hoops:
            cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
            if cy > max_y:
                continue
            # Hoop bottom inside the lowest quarter of an overlapping player box =
            # it sits at someone's feet. A real rim is above a jumper's box top.
            below_feet = any(p["bbox"][0] < hb[2] and p["bbox"][2] > hb[0]
                             and p["bbox"][1] < hb[3] and p["bbox"][3] > hb[1]
                             and hb[3] > p["bbox"][3] - 0.25 * (p["bbox"][3] - p["bbox"][1])
                             for p in players)
            if not below_feet:
                ok_hoops.append((c, hb))
        self.stats["hoop_rejected"] += len(hoops) - len(ok_hoops)
        hoops = ok_hoops
        # A hoop counts only after `hoop_min_streak` consecutive detections near
        # the same spot (wall items flash up for a frame or two, a hoop stays).
        if hoops:
            _c, hb = max(hoops, key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))
            hb = _round(hb)
            near_prev = self.hoop_pending is not None and abs(hb[0] - self.hoop_pending[0]) < 60 \
                and abs(hb[1] - self.hoop_pending[1]) < 60
            self.hoop_streak = self.hoop_streak + 1 if near_prev else 1
            self.hoop_pending = hb
            if self.hoop_streak >= self.hoop_min_streak:
                self.last_hoop, self.last_hoop_frame = hb, frame_index
                self.stats["hoop_fresh"] += 1
        else:
            self.hoop_streak, self.hoop_pending = 0, None
        if self.last_hoop is not None and frame_index - self.last_hoop_frame > self.hoop_hold:
            self.last_hoop = None
        hoop_list = [{"bbox": self.last_hoop}] if self.last_hoop else []

        ball = None
        balls = [(c, b) for c, b in balls
                 if b[2] - b[0] <= self.ball_max_px and b[3] - b[1] <= self.ball_max_px]
        picked = self.gate.pick(balls, self.last_hoop, [p["bbox"] for p in players])
        if picked:
            bconf, b = picked
            b = _round(b)
            ball = {"bbox": b, "center": [round((b[0] + b[2]) / 2, 1), round((b[1] + b[3]) / 2, 1)],
                    "conf": round(bconf, 3)}

        st = self.stats
        st["frames"] += 1
        st["ball_frames"] += ball is not None
        st["hoop_frames"] += bool(hoop_list)
        st["player_dets"] += len(players)
        st["ids"].update(p["id"] for p in players)
        return {"frame": frame_index, "t": round(t, 3), "players": players,
                "ball": ball, "hoops": hoop_list}

    def summary(self) -> dict:
        st, n = self.stats, max(self.stats["frames"], 1)
        return {"weights": self.weights_info, "frames_processed": st["frames"],
                "players_per_frame": round(st["player_dets"] / n, 2),
                "ball_frame_share": round(st["ball_frames"] / n, 4),
                "hoop_frame_share": round(st["hoop_frames"] / n, 4),
                "hoop_detected_share": round(st["hoop_fresh"] / n, 4),
                "hoop_rejected": st["hoop_rejected"], "resets": self.resets,
                "id_switches": self.switches,
                "track_ids": len(st["ids"]), "team_mode": self.teams.mode,
                "team_centroids_lab": self.teams.centroids_lab, **self.gate.summary()}
