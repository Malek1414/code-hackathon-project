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

import json
import re
from pathlib import Path

import cv2
import numpy as np

from vision.live.state import hex_to_bgr

CANVAS = (1920, 1080)
GEOMETRY = {  # id: (x, y, w, h) on the 1920x1080 canvas (placeholders; FRONTEND's PNGs are full-canvas)
    "score_bug": (700, 64, 520, 88),
    "made_flash": (700, 64, 520, 88),
    "player_card": (48, 872, 480, 160),
    "team_overview": (510, 330, 900, 420),
    "lower_third": (0, 960, 1920, 120),
    "end_summary": (0, 0, 1920, 1080),
    "heat_map": (0, 0, 1920, 1080),
    "heat_map_frame": (0, 0, 1920, 1080),
}
GLASS = (28, 28, 32)
TEXT = (240, 240, 240)
DIM = (170, 170, 170)


STATIC_WIDGETS = {"lower_third", "end_summary", "heat_map_frame", "heat_map"}
DYNAMIC_WIDGETS = {"score_bug", "made_flash", "player_card", "team_overview"}


class Assets:
    """FRONTEND's PNGs are full 1920x1080 BGRA canvases. Static widgets
    (lower_third, end_summary, heat_map_frame) are composited as rendered.
    Dynamic widgets carry numbers baked in by a 5 s Chrome render, which is too
    slow for a live score, so they are used only as `<id>_template.png` (chrome
    with empty text areas) plus broadcast/widgets/layout.json text boxes into
    which live.py writes the numbers every frame; without a template the
    placeholder panel is drawn. (ORCH decision 14:52.)"""

    def __init__(self, folder: str | Path = "broadcast/assets", refresh_s: float = 1.0,
                 layout_path: str | Path = "broadcast/widgets/layout.json") -> None:
        self.folder = Path(folder)
        self.layout_path = Path(layout_path)
        self._cache: dict[str, np.ndarray | None] = {}
        self._mtime: dict[str, float] = {}
        self._checked: dict[str, float] = {}
        self._layout: dict = {}
        self._layout_mtime = -1.0
        self.refresh_s = refresh_s  # FRONTEND drops PNGs while the show runs: re-check the folder now and then

    def layout(self, wid: str) -> list[dict]:
        """Text boxes for a dynamic widget from layout.json (re-read on change)."""
        mtime = self.layout_path.stat().st_mtime if self.layout_path.exists() else -1.0
        if mtime != self._layout_mtime:
            self._layout_mtime = mtime
            try:
                self._layout = json.loads(self.layout_path.read_text()) if mtime >= 0 else {}
            except json.JSONDecodeError:
                self._layout = {}
        boxes = self._layout.get(wid)
        if isinstance(boxes, dict):
            boxes = [{"id": k, **v} for k, v in boxes.items() if isinstance(v, dict)]
        return boxes if isinstance(boxes, list) else []

    def get(self, wid: str, variant: str = "") -> np.ndarray | None:
        """Sprite for the widget: the rendered PNG (static widgets) or the
        template PNG (dynamic widgets; `variant` "_b" = team B tint when that
        file exists), or None (placeholder)."""
        import time as _time

        key = wid + variant
        now = _time.monotonic()
        if key in self._cache and now - self._checked.get(key, 0.0) < self.refresh_s:
            return self._cache[key]
        self._checked[key] = now
        name = f"{wid}_template{variant}.png" if wid in DYNAMIC_WIDGETS else f"{wid}.png"
        path = self.folder / name
        if variant and not path.exists():
            return self.get(wid)  # no team-B file: the team-A template
        mtime = path.stat().st_mtime if path.exists() else -1.0
        if key in self._cache and self._mtime.get(key) == mtime:
            return self._cache[key]
        self._mtime[key] = mtime
        img = None
        if path.exists():
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is not None:
                if raw.ndim == 2:
                    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
                elif raw.shape[2] == 3:
                    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
                if (raw.shape[1], raw.shape[0]) != CANVAS:
                    x, y, w, h = GEOMETRY[wid]
                    full = np.zeros((CANVAS[1], CANVAS[0], 4), np.uint8)  # a cropped widget: place it at its slot
                    full[y:y + h, x:x + w] = cv2.resize(raw, (w, h))
                    raw = full
                img = _crop_to_alpha(raw)
        self._cache[key] = img
        return img

    def external(self, name: str) -> np.ndarray | None:
        """COURT's renders (heat_map.png) from broadcast/assets or out/ (cached)."""
        key = f"ext:{name}"
        if key in self._cache:
            return self._cache[key]
        raw = None
        for p in (self.folder / name, Path("out") / name):
            if p.exists():
                raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    break
        self._cache[key] = raw
        return raw


class Sprite:
    """A widget's opaque region only: blending the full 1920x1080 canvas per
    frame costs ~40 ms, the cropped region ~2 ms."""

    __slots__ = ("bgra", "x", "y")

    def __init__(self, bgra: np.ndarray, x: int, y: int) -> None:
        self.bgra, self.x, self.y = bgra, x, y


def _crop_to_alpha(raw: np.ndarray) -> Sprite | None:
    ys, xs = np.where(raw[:, :, 3] > 0)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    return Sprite(np.ascontiguousarray(raw[y0:y1, x0:x1]), x0, y0)


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


ROLE_COLORS = {"white": TEXT, "muted": DIM, "text": TEXT}
BOX_FONT = cv2.FONT_HERSHEY_SIMPLEX  # plain zero (DUPLEX draws a slashed one); 1.0 ~ 22 px cap height
HERSHEY_CAP_PX = 22.0
_PATH = re.compile(r"\{([^{}]+)\}")


def resolve_path(path: str, ctx: dict):
    """'teams[last_event.team].name' style paths against the state context."""
    cur = ctx
    for part in re.findall(r"[^.\[\]]+|\[[^\]]+\]", path):
        if part.startswith("["):
            idx = part[1:-1]
            if not idx.lstrip("-").isdigit():
                idx = resolve_path(idx, ctx)
            try:
                cur = cur[int(idx)]
            except (TypeError, ValueError, IndexError, KeyError):
                return None
        else:
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def render_format(fmt: str, ctx: dict) -> str | None:
    """FRONTEND's format strings, e.g. 'BASKET  {teams[last_event.team].name} +{last_event.points}'."""
    missing = False

    def sub(m):
        nonlocal missing
        v = resolve_path(m.group(1).strip(), ctx)
        if v is None:
            missing = True
            return ""
        if isinstance(v, float):
            return f"{int(round(v * 100))}%" if 0 <= v <= 1 and "pct" in m.group(1) else f"{v:g}"
        return str(v)

    out = _PATH.sub(sub, fmt)
    return None if missing else out


def draw_boxes(frame: np.ndarray, boxes: list[dict], fields: dict, teams: list[dict], ctx: dict | None = None) -> None:
    """Live text into FRONTEND's layout boxes (canvas px: x, y, w, h, baseline,
    font_px, align left|center|right, color_role team_a|team_b|white|muted;
    text from the box's `format` over the state context, else from `field`)."""
    k = _scale(frame)
    for b in boxes:
        val = None
        if ctx is not None and b.get("format"):
            val = render_format(b["format"], ctx)
        if val is None:
            val = fields.get(b.get("field", b.get("id")))
        if val is None:
            continue
        s = str(val).replace("0", "O")  # every Hershey face slashes its zero; the capital O is the plain oval
        size = float(b.get("font_px", 24)) / HERSHEY_CAP_PX * k
        thick = max(1, int(round(size * 1.6)))
        role = b.get("color_role", "white")
        if role == "team_a":
            color = hex_to_bgr(teams[0]["color"])
        elif role == "team_b":
            color = hex_to_bgr(teams[1]["color"])
        else:
            color = ROLE_COLORS.get(role, TEXT)
        (tw, th), _ = cv2.getTextSize(s, BOX_FONT, size, thick)
        x, y, w, h = (float(b.get("x", 0)) * k, float(b.get("y", 0)) * k, float(b.get("w", 0)) * k, float(b.get("h", 0)) * k)
        if w and tw > w:  # never spill out of the box: shrink to fit
            size *= w / tw
            thick = max(1, int(round(size * 1.6)))
            (tw, th), _ = cv2.getTextSize(s, BOX_FONT, size, thick)
        align = b.get("align", "left")
        if align == "center":
            x = x + (w - tw) / 2
        elif align == "right":
            x = x + w - tw
        if b.get("baseline") is not None:
            baseline = float(b["baseline"]) * k
        elif h:
            baseline = y + (h + th) / 2
        else:
            baseline = y
        cv2.putText(frame, s, (int(x), int(baseline)), BOX_FONT, size, color, thick, cv2.LINE_AA)


def composite(frame: np.ndarray, sprite, alpha_scale: float = 1.0) -> None:
    """A widget sprite (canvas coordinates) onto the frame, scaled if the frame is not 1080p."""
    if sprite is None:
        return
    if isinstance(sprite, np.ndarray):
        sprite = _crop_to_alpha(sprite)
        if sprite is None:
            return
    k = _scale(frame)
    img = sprite.bgra
    if alpha_scale != 1.0:
        img = img.copy()
        img[:, :, 3] = (img[:, :, 3].astype(np.float32) * alpha_scale).astype(np.uint8)
    if k != 1.0:
        img = cv2.resize(img, (max(int(img.shape[1] * k), 1), max(int(img.shape[0] * k), 1)))
    blend(frame, img, int(sprite.x * k), int(sprite.y * k))


def _place(frame: np.ndarray, assets: Assets, wid: str, accent=None, fields: dict | None = None,
           teams: list[dict] | None = None, ctx: dict | None = None, variant: str = "") -> tuple[int, int, int, int, float] | None:
    """Composite FRONTEND's widget (static: as rendered; dynamic: template +
    live text from layout.json) and return None, or draw the placeholder
    panel and return its geometry for the built-in text."""
    img = assets.get(wid, variant)
    if img is not None:
        composite(frame, img)
        if wid in DYNAMIC_WIDGETS and fields is not None:
            draw_boxes(frame, assets.layout(wid), fields, teams or [], ctx)
        return None
    x, y, w, h = _geom(frame, wid)
    glass(frame, x, y, w, h, accent=accent)
    return x, y, w, h, _scale(frame)


# --- individual widgets --------------------------------------------------------


def draw_score_bug(frame, assets: Assets, teams: list[dict], clock: str, ctx: dict | None = None) -> None:
    fields = {"team_a_name": teams[0]["name"], "team_b_name": teams[1]["name"], "score_a": teams[0]["score"],
              "score_b": teams[1]["score"], "clock": clock, "period": "P1"}
    placed = _place(frame, assets, "score_bug", fields=fields, teams=teams, ctx=ctx)
    if placed is None:
        return
    x, y, w, h, k = placed
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


def draw_made_flash(frame, assets: Assets, team: dict | None, label: str, age_s: float, duration_s: float,
                    ctx: dict | None = None, teams: list[dict] | None = None) -> None:
    x, y, w, h, k = _geom(frame, "made_flash") + (_scale(frame),)
    fade = max(0.0, 1 - age_s / duration_s)
    col = hex_to_bgr(team["color"]) if team else (60, 200, 60)
    img = assets.get("made_flash", "_b" if team and team.get("id") == 1 else "")
    if img is not None:
        composite(frame, img, alpha_scale=fade)
        draw_boxes(frame, assets.layout("made_flash"), {"label": label}, teams or [team or {"color": "#3cc83c"}] * 2, ctx)
        return
    else:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y + h + int(6 * k)), (x + w, y + h + int(46 * k)), col, -1)
        cv2.addWeighted(overlay, 0.85 * fade, frame, 1 - 0.85 * fade, 0, frame)
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9 * k, 2)
    text(frame, label, x + (w - tw) // 2, y + h + int(36 * k), 0.9 * k, TEXT, 2)


def draw_player_card(frame, assets: Assets, player: dict, team: dict, ctx: dict | None = None) -> None:
    pct = f"{int(round(100 * player['fg_pct']))}%" if player.get("fg_pct") is not None else "-"
    fields = {"number": f"#{player['number']}" if player.get("number") else player["key"], "key": player["key"],
              "team_name": team["name"], "pts": player["pts"], "fg": f"{player['fgm']}/{player['fga']}", "fg_pct": pct,
              "line": f"{player['fgm']}/{player['fga']} FG, {pct}"}
    pctx = dict(ctx or {}, player=player, team=team)
    placed = _place(frame, assets, "player_card", accent=hex_to_bgr(team["color"]), fields=fields,
                    teams=(ctx or {}).get("teams") or [team, team], ctx=pctx, variant="_b" if team.get("id") == 1 else "")
    if placed is None:
        return
    x, y, w, h, k = placed
    num = f"#{player['number']}" if player.get("number") else player["key"]
    text(frame, num, x + int(24 * k), y + int(58 * k), 1.4 * k, TEXT, 3)
    text(frame, team["name"][:16], x + int(24 * k), y + int(96 * k), 0.7 * k, DIM, 2)
    fg = f"{player['fgm']}/{player['fga']}"
    pct = f"{int(round(100 * player['fg_pct']))}%" if player.get("fg_pct") is not None else "-"
    text(frame, f"PTS {player['pts']}   FG {fg}   {pct}", x + int(24 * k), y + int(140 * k), 0.75 * k, TEXT, 2)


def draw_team_overview(frame, assets: Assets, teams: list[dict], players: list[dict], ctx: dict | None = None) -> None:
    fields = {}
    for i, suf in enumerate(("a", "b")):
        t = teams[i]
        top = next((p for p in players if p["team"] == t["id"] and p["pts"] > 0), None)
        pct = f"{int(round(100 * t['fg_pct']))}%" if t.get("fg_pct") is not None else "-"
        fields.update({f"team_{suf}_name": t["name"], f"score_{suf}": t["score"], f"fg_{suf}": f"{t['fgm']}/{t['fga']}",
                       f"fg_pct_{suf}": pct, f"poss_{suf}": t["possessions"],
                       f"top_{suf}": (f"#{top['number']}" if top and top.get("number") else (top["key"] if top else "-")),
                       f"top_pts_{suf}": top["pts"] if top else 0})
    placed = _place(frame, assets, "team_overview", fields=fields, teams=teams, ctx=ctx)
    if placed is None:
        return
    x, y, w, h, k = placed
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
    placed = _place(frame, assets, "lower_third")
    if placed is None:
        return
    x, y, w, h, k = placed
    text(frame, "BIG BALL BALLER", x + int(40 * k), y + int(78 * k), 1.6 * k, TEXT, 3)
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.9 * k, 2)
    text(frame, title, x + w - int(40 * k) - tw, y + int(74 * k), 0.9 * k, DIM, 2)


def draw_end_summary(frame, assets: Assets, teams: list[dict], players: list[dict]) -> None:
    placed = _place(frame, assets, "end_summary")
    if placed is None:
        return
    x, y, w, h, k = placed
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
    """FRONTEND's heat_map_frame.png (full canvas with alpha) over COURT's
    heat_map.png; False when neither exists yet."""
    base = assets.external("heat_map.png")
    frame_png = assets.get("heat_map_frame")
    if base is None and frame_png is None:
        placed = _place(frame, assets, "heat_map")
        if placed:
            x, y, w, h, k = placed
            text(frame, "HEAT MAP (COURT render pending)", x + int(60 * k), y + int(90 * k), 1.3 * k, DIM, 3)
        return False
    if base is not None:
        if base.ndim == 3 and base.shape[2] == 4:
            frame[:] = cv2.resize(base[:, :, :3], (frame.shape[1], frame.shape[0]))
        else:
            frame[:] = cv2.resize(base[:, :, :3], (frame.shape[1], frame.shape[0]))
    if frame_png is not None:
        composite(frame, frame_png)
    return True


# --- scheduler ------------------------------------------------------------------


class WidgetScheduler:
    def __init__(self, assets: Assets, title: str = "", overview_every_s: float = 300.0, top_cards_every_s: float = 180.0) -> None:
        self.assets = assets
        self.title = title
        self.overview_every_s = overview_every_s
        self.top_cards_every_s = top_cards_every_s
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
    def render(self, frame, t: float, teams: list[dict], players: list[dict], clock: str,
               last_event: dict | None = None, period: int = 1) -> None:
        if self.end_since is not None:
            page = int((t - self.end_since) // 6) % 2
            if page == 0 or not draw_heat_map(frame, self.assets):
                draw_end_summary(frame, self.assets, teams, players)
            return
        top = {}
        for tid in (0, 1):
            top[tid] = next((p for p in players if p["team"] == tid and p["pts"] > 0), None)
        ctx = {"teams": teams, "players": players, "clock": clock, "period": period, "last_event": last_event or {},
               "top": [top[0] or {}, top[1] or {}]}
        flashing = bool(self.flash and t - self.flash[2] <= 1.5)
        if flashing:  # the flash panel sits on the score bug: one of them at a time
            team = teams[self.flash[1]] if self.flash[1] in (0, 1) else None
            draw_made_flash(frame, self.assets, team, self.flash[0], t - self.flash[2], 1.5, ctx=ctx, teams=teams)
        else:
            draw_score_bug(frame, self.assets, teams, clock, ctx=ctx)
        # top scorer cards every 3 min
        if t - self.last_top_cards >= self.top_cards_every_s and t > 0:
            self.last_top_cards = t
            self.top_queue = [p["key"] for tid in (0, 1) for p in players if p["team"] == tid and p["pts"] > 0][:2]
        if (self.card is None or t - self.card[1] > 3.0) and self.top_queue:
            self.card = (self.top_queue.pop(0), t)
        if self.card and t - self.card[1] <= 3.0:
            p = next((p for p in players if p["key"] == self.card[0]), None)
            if p and p["team"] in (0, 1):
                draw_player_card(frame, self.assets, p, teams[p["team"]], ctx=ctx)
        if t - self.last_overview >= self.overview_every_s and t > 0:
            self.last_overview = t
            self.overview_until = t + 6.0
        if t <= self.overview_until:
            draw_team_overview(frame, self.assets, teams, players, ctx=ctx)
        if t <= self.lower_third_until:
            draw_lower_third(frame, self.assets, self.title)
