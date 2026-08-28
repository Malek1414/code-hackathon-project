"""Big Ball Baller widgets: timing rules and compositing on the 1920x1080 frame.

Designs are PNGs with alpha in broadcast/assets/<id>.png (FRONTEND); until
they exist a dark glass placeholder of the same size sits at the same spot.
Dynamic text (scores, names, numbers) is drawn on top by this module either way.

Timing (docs/ORCHESTRATION.md, Broadcast package): score_bug always;
made_flash 1.5 s after a made basket; player_card 3 s after a made basket by
that player and the top scorer of each team every 3 min; team_overview every
5 min for 6 s and on hotkey t; lower_third the first 10 s and on hotkey b;
end_summary + heat_map on hotkey e or at the end of the file.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision.live.state import hex_to_bgr

CANVAS = (1920, 1080)
GEOMETRY = {  # id: (x, y, w, h) on the 1920x1080 canvas
    "score_bug": (700, 24, 520, 88),
    "made_flash": (700, 24, 520, 88),
    "player_card": (40, 880, 420, 160),
    "team_overview": (510, 330, 900, 420),
    "lower_third": (0, 960, 1920, 120),
    "end_summary": (0, 0, 1920, 1080),
    "heat_map": (0, 0, 1920, 1080),
}
GLASS = (28, 28, 32)
TEXT = (240, 240, 240)
DIM = (170, 170, 170)


class Assets:
    def __init__(self, folder: str | Path = "broadcast/assets", refresh_s: float = 30.0) -> None:
        self.folder = Path(folder)
        self._cache: dict[str, np.ndarray | None] = {}
        self._mtime: dict[str, float] = {}
        self._checked: dict[str, float] = {}
        self.refresh_s = refresh_s  # FRONTEND drops PNGs while the show runs: re-check the folder now and then

    def get(self, wid: str) -> np.ndarray | None:
        """BGRA image for the widget, resized to its geometry, or None (placeholder)."""
        import time as _time

        now = _time.monotonic()
        if wid in self._cache and now - self._checked.get(wid, 0.0) < self.refresh_s:
            return self._cache[wid]
        self._checked[wid] = now
        path = self.folder / f"{wid}.png"
        mtime = path.stat().st_mtime if path.exists() else -1.0
        if wid in self._cache and self._mtime.get(wid) == mtime:
            return self._cache[wid]
        self._mtime[wid] = mtime
        img = None
        for name in (f"{wid}.png",):
            p = self.folder / name
            if p.exists():
                raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    if raw.ndim == 2:
                        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
                    elif raw.shape[2] == 3:
                        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
                    x, y, w, h = GEOMETRY[wid]
                    img = cv2.resize(raw, (w, h)) if raw.shape[1] != w or raw.shape[0] != h else raw
        self._cache[wid] = img
        return img

    def external(self, name: str) -> np.ndarray | None:
        """COURT's renders (heat_map.png) from broadcast/assets or out/."""
        for p in (self.folder / name, Path("out") / name):
            if p.exists():
                raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    return raw
        return None


def _scale(frame: np.ndarray) -> float:
    return frame.shape[1] / CANVAS[0]


def blend(frame: np.ndarray, bgra: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite a BGRA image onto the BGR frame at (x, y), clipped."""
    h, w = bgra.shape[:2]
    H, W = frame.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    patch = bgra[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (patch[:, :, :3].astype(np.float32) * alpha + roi * (1 - alpha)).astype(np.uint8)


def glass(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.78, accent=None) -> None:
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    panel = np.full_like(roi, GLASS)
    frame[y0:y1, x0:x1] = cv2.addWeighted(panel, alpha, roi, 1 - alpha, 0)
    if accent is not None:
        cv2.rectangle(frame, (x0, y0), (x0 + max(int(6 * _scale(frame)), 2), y1), accent, -1)


def text(frame: np.ndarray, s: str, x: int, y: int, size: float = 1.0, color=TEXT, thick: int = 2, font=cv2.FONT_HERSHEY_DUPLEX) -> None:
    cv2.putText(frame, s, (x, y), font, size, color, thick, cv2.LINE_AA)


def _geom(frame: np.ndarray, wid: str) -> tuple[int, int, int, int]:
    k = _scale(frame)
    x, y, w, h = GEOMETRY[wid]
    return int(x * k), int(y * k), int(w * k), int(h * k)


def _place(frame: np.ndarray, assets: Assets, wid: str, accent=None) -> tuple[int, int, int, int, float]:
    x, y, w, h = _geom(frame, wid)
    img = assets.get(wid)
    if img is not None:
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h))
        blend(frame, img, x, y)
    else:
        glass(frame, x, y, w, h, accent=accent)
    return x, y, w, h, _scale(frame)


# --- individual widgets --------------------------------------------------------


def draw_score_bug(frame, assets: Assets, teams: list[dict], clock: str) -> None:
    x, y, w, h, k = _place(frame, assets, "score_bug")
    a, b = teams
    ca, cb = hex_to_bgr(a["color"]), hex_to_bgr(b["color"])
    cv2.rectangle(frame, (x, y), (x + int(10 * k), y + h), ca, -1)
    cv2.rectangle(frame, (x + w - int(10 * k), y), (x + w, y + h), cb, -1)
    text(frame, a["name"][:12], x + int(24 * k), y + int(36 * k), 0.8 * k, TEXT, 2)
    text(frame, f"{a['score']}", x + int(24 * k), y + int(76 * k), 1.2 * k, TEXT, 2)
    text(frame, clock, x + int(220 * k), y + int(58 * k), 1.0 * k, DIM, 2)
    (tw, _), _ = cv2.getTextSize(b["name"][:12], cv2.FONT_HERSHEY_DUPLEX, 0.8 * k, 2)
    text(frame, b["name"][:12], x + w - int(24 * k) - tw, y + int(36 * k), 0.8 * k, TEXT, 2)
    (tw, _), _ = cv2.getTextSize(f"{b['score']}", cv2.FONT_HERSHEY_DUPLEX, 1.2 * k, 2)
    text(frame, f"{b['score']}", x + w - int(24 * k) - tw, y + int(76 * k), 1.2 * k, TEXT, 2)


def draw_made_flash(frame, assets: Assets, team: dict | None, label: str, age_s: float, duration_s: float) -> None:
    x, y, w, h, k = _geom(frame, "made_flash") + (_scale(frame),)
    fade = max(0.0, 1 - age_s / duration_s)
    col = hex_to_bgr(team["color"]) if team else (60, 200, 60)
    img = assets.get("made_flash")
    if img is not None:
        img = img.copy()
        img[:, :, 3] = (img[:, :, 3].astype(np.float32) * fade).astype(np.uint8)
        blend(frame, img, x, y)
    else:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y + h + int(6 * k)), (x + w, y + h + int(46 * k)), col, -1)
        cv2.addWeighted(overlay, 0.85 * fade, frame, 1 - 0.85 * fade, 0, frame)
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9 * k, 2)
    text(frame, label, x + (w - tw) // 2, y + h + int(36 * k), 0.9 * k, TEXT, 2)


def draw_player_card(frame, assets: Assets, player: dict, team: dict) -> None:
    x, y, w, h, k = _place(frame, assets, "player_card", accent=hex_to_bgr(team["color"]))
    num = f"#{player['number']}" if player.get("number") else player["key"]
    text(frame, num, x + int(24 * k), y + int(58 * k), 1.4 * k, TEXT, 3)
    text(frame, team["name"][:16], x + int(24 * k), y + int(96 * k), 0.7 * k, DIM, 2)
    fg = f"{player['fgm']}/{player['fga']}"
    pct = f"{int(round(100 * player['fg_pct']))}%" if player.get("fg_pct") is not None else "-"
    text(frame, f"PTS {player['pts']}   FG {fg}   {pct}", x + int(24 * k), y + int(140 * k), 0.75 * k, TEXT, 2)


def draw_team_overview(frame, assets: Assets, teams: list[dict], players: list[dict]) -> None:
    x, y, w, h, k = _place(frame, assets, "team_overview")
    text(frame, "TEAM OVERVIEW", x + int(30 * k), y + int(50 * k), 0.9 * k, DIM, 2)
    for i, t in enumerate(teams):
        cx = x + int(30 * k) + i * (w // 2)
        col = hex_to_bgr(t["color"])
        cv2.rectangle(frame, (cx, y + int(70 * k)), (cx + int(12 * k), y + h - int(30 * k)), col, -1)
        cx += int(30 * k)
        text(frame, t["name"][:14], cx, y + int(110 * k), 0.9 * k, TEXT, 2)
        text(frame, f"{t['score']}", cx, y + int(190 * k), 2.0 * k, TEXT, 3)
        pct = f"{int(round(100 * t['fg_pct']))}%" if t.get("fg_pct") is not None else "-"
        text(frame, f"FG {t['fgm']}/{t['fga']}  {pct}", cx, y + int(250 * k), 0.8 * k, TEXT, 2)
        text(frame, f"possessions {t['possessions']}", cx, y + int(300 * k), 0.8 * k, TEXT, 2)
        top = next((p for p in players if p["team"] == t["id"] and p["pts"] > 0), None)
        if top:
            label = f"#{top['number']}" if top.get("number") else top["key"]
            text(frame, f"top {label}  {top['pts']} pts", cx, y + int(350 * k), 0.8 * k, TEXT, 2)


def draw_lower_third(frame, assets: Assets, title: str) -> None:
    x, y, w, h, k = _place(frame, assets, "lower_third")
    text(frame, "BIG BALL BALLER", x + int(40 * k), y + int(78 * k), 1.6 * k, TEXT, 3)
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.9 * k, 2)
    text(frame, title, x + w - int(40 * k) - tw, y + int(74 * k), 0.9 * k, DIM, 2)


def draw_end_summary(frame, assets: Assets, teams: list[dict], players: list[dict]) -> None:
    x, y, w, h, k = _place(frame, assets, "end_summary")
    text(frame, "BIG BALL BALLER   GAME SUMMARY", x + int(60 * k), y + int(90 * k), 1.5 * k, TEXT, 3)
    text(frame, f"{teams[0]['name']} {teams[0]['score']} : {teams[1]['score']} {teams[1]['name']}",
         x + int(60 * k), y + int(160 * k), 1.2 * k, DIM, 2)
    total_poss = sum(p["possession_s"] for p in players) or 1.0
    cols = [("PLAYER", 60), ("PTS", 520), ("FGA", 680), ("FGM", 840), ("FG%", 1000), ("POSS", 1180), ("DIST m", 1400)]
    for name, cx in cols:
        text(frame, name, x + int(cx * k), y + int(240 * k), 0.8 * k, DIM, 2)
    for i, p in enumerate(players[:14]):
        yy = y + int((290 + i * 48) * k)
        col = hex_to_bgr(teams[p["team"]]["color"]) if p["team"] in (0, 1) else DIM
        cv2.rectangle(frame, (x + int(36 * k), yy - int(28 * k)), (x + int(48 * k), yy + int(6 * k)), col, -1)
        label = f"#{p['number']}" if p.get("number") else p["key"]
        pct = f"{int(round(100 * p['fg_pct']))}%" if p.get("fg_pct") is not None else "-"
        vals = [label, str(p["pts"]), str(p["fga"]), str(p["fgm"]), pct, f"{int(round(100 * p['possession_s'] / total_poss))}%",
                "-" if p.get("distance_m") is None else str(p["distance_m"])]
        for (name, cx), v in zip(cols, vals):
            text(frame, v, x + int(cx * k), yy, 0.8 * k, TEXT, 2)


def draw_heat_map(frame, assets: Assets) -> bool:
    """COURT's heat_map.png full frame; False when it does not exist yet."""
    img = assets.external("heat_map.png")
    if img is None:
        x, y, w, h, k = _place(frame, assets, "heat_map")
        text(frame, "HEAT MAP (COURT render pending)", x + int(60 * k), y + int(90 * k), 1.3 * k, DIM, 3)
        return False
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.resize(img, (frame.shape[1], frame.shape[0]))
        blend(frame, img, 0, 0)
    else:
        frame[:] = cv2.resize(img[:, :, :3], (frame.shape[1], frame.shape[0]))
    return True


# --- scheduler ------------------------------------------------------------------


class WidgetScheduler:
    def __init__(self, assets: Assets, title: str = "") -> None:
        self.assets = assets
        self.title = title
        self.flash: tuple[str, int | None, float] | None = None  # label, team id, shown at t
        self.card: tuple[str, float] | None = None  # player key, shown at t
        self.overview_until = -1.0
        self.lower_third_until = 10.0
        self.end_since: float | None = None
        self.last_top_cards = -1.0
        self.last_overview = 0.0
        self.top_queue: list[str] = []

    # events
    def made(self, t: float, team: int | None, player_key: str | None, label: str) -> None:
        self.flash = (label, team, t)
        if player_key:
            self.card = (player_key, t)

    def manual(self, t: float, team: int, label: str) -> None:
        self.flash = (label, team, t)

    def hotkey(self, key: str, t: float) -> None:
        if key == "t":
            self.overview_until = t + 6.0
        elif key == "b":
            self.lower_third_until = t + 6.0
        elif key == "e":
            self.end_since = t if self.end_since is None else None

    def end_of_file(self, t: float) -> None:
        if self.end_since is None:
            self.end_since = t

    # per frame
    def render(self, frame, t: float, teams: list[dict], players: list[dict], clock: str) -> None:
        if self.end_since is not None:
            page = int((t - self.end_since) // 6) % 2
            if page == 0 or not draw_heat_map(frame, self.assets):
                draw_end_summary(frame, self.assets, teams, players)
            return
        draw_score_bug(frame, self.assets, teams, clock)
        if self.flash and t - self.flash[2] <= 1.5:
            team = teams[self.flash[1]] if self.flash[1] in (0, 1) else None
            draw_made_flash(frame, self.assets, team, self.flash[0], t - self.flash[2], 1.5)
        # top scorer cards every 3 min
        if t - self.last_top_cards >= 180.0 and t > 0:
            self.last_top_cards = t
            self.top_queue = [p["key"] for tid in (0, 1) for p in players if p["team"] == tid and p["pts"] > 0][:2]
        if (self.card is None or t - self.card[1] > 3.0) and self.top_queue:
            self.card = (self.top_queue.pop(0), t)
        if self.card and t - self.card[1] <= 3.0:
            p = next((p for p in players if p["key"] == self.card[0]), None)
            if p and p["team"] in (0, 1):
                draw_player_card(frame, self.assets, p, teams[p["team"]])
        if t - self.last_overview >= 300.0 and t > 0:
            self.last_overview = t
            self.overview_until = t + 6.0
        if t <= self.overview_until:
            draw_team_overview(frame, self.assets, teams, players)
        if t <= self.lower_third_until:
            draw_lower_third(frame, self.assets, self.title)
