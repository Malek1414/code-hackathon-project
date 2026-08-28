"""Big Ball Baller heat maps: position density per team on the top-down court + shot chart.

    .venv/bin/python broadcast/heatmap.py [--tracks out/game10/tracks.jsonl] [--calib out/court_calib_game10.json]
        [--events out/game10/events.json] [--config broadcast/config.json] [--out broadcast/assets]

Writes heat_map_A.png, heat_map_B.png (one court each, 1200x680) and heat_map.png
(1920x1080 broadcast frame, both teams, BBB wordmark). Only calibrated, on-court
foot positions count; frames inside the calibration's uncertain stretches and the
close-up segments are skipped. Density = Gaussian kernel (sigma 1 m) on a 10 cm
grid, one hue per team (the team colour, light to dark on the dark surface).
Shots from events.json: made = filled dot, missed = hollow ring, number from
the player key when identified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

cv2.setNumThreads(2)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.court.geometry import FIBA, polylines  # noqa: E402
from vision.court.project import Calibration, iter_tracks, load_calibration  # noqa: E402

BRAND = "Big Ball Baller"
DEFAULT_TEAMS = {0: {"name": "Team A", "color": "#2f6fdb"}, 1: {"name": "Team B", "color": "#c8102e"}}
BG = (15, 17, 21)          # RGB, page background
PANEL = (24, 27, 34)       # glass panel
COURT_FILL = (34, 38, 46)
LINE = (205, 210, 220)
INK = (230, 232, 238)
MUTED = (140, 148, 167)
SCALE = 40.0               # px per metre on the court panels
MARGIN_M = 1.0
CELL_M = 0.1               # density grid
SIGMA_M = 1.0


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_teams(config: Path | None) -> dict[int, dict]:
    teams = {k: dict(v) for k, v in DEFAULT_TEAMS.items()}
    if config and config.exists():
        cfg = json.loads(config.read_text())
        for t in cfg.get("teams", []):
            teams.setdefault(int(t["id"]), {}).update({k: v for k, v in t.items() if k in ("name", "color")})
    return teams


# --- data ---------------------------------------------------------------------


def team_positions(tracks: Path, cal: Calibration) -> tuple[dict[int, np.ndarray], dict]:
    """Foot positions in metres per team, only calibrated + on-court + not uncertain."""
    per_team: dict[int, list] = {0: [], 1: []}
    used = skipped_uncal = skipped_unc = off = 0
    for rec in iter_tracks(tracks):
        frame = int(rec["frame"])
        players = rec.get("players") or []
        if not players:
            continue
        if cal.is_uncertain(frame):
            skipped_unc += 1
            continue
        H = cal.H_m_to_px(frame)
        if not np.isfinite(H).all():
            skipped_uncal += 1
            continue
        xy = cal.project(frame, [p["foot"] for p in players])
        ok = cal.on_court(xy, 0.5)
        for p, pt, keep in zip(players, xy, ok):
            t = int(p.get("team", -1))
            if keep and t in per_team:
                per_team[t].append(pt)
                used += 1
            elif not keep:
                off += 1
    info = {"positions_used": used, "positions_off_court": off, "frames_uncalibrated": skipped_uncal, "frames_uncertain": skipped_unc}
    return {t: np.array(v, np.float64).reshape(-1, 2) for t, v in per_team.items()}, info


def density(points_m: np.ndarray) -> np.ndarray:
    """Gaussian-smoothed count on a CELL_M grid over the court, normalised to 0..1."""
    nx, ny = int(FIBA.length_m / CELL_M), int(FIBA.width_m / CELL_M)
    grid = np.zeros((ny, nx), np.float32)
    if len(points_m):
        ix = np.clip((points_m[:, 0] / CELL_M).astype(int), 0, nx - 1)
        iy = np.clip((points_m[:, 1] / CELL_M).astype(int), 0, ny - 1)
        np.add.at(grid, (iy, ix), 1.0)
    sigma = SIGMA_M / CELL_M
    grid = cv2.GaussianBlur(grid, (0, 0), sigma)
    peak = np.percentile(grid, 99.5) if grid.max() > 0 else 1.0
    return np.clip(grid / max(peak, 1e-9), 0, 1)


# --- drawing --------------------------------------------------------------------


class CourtPanel:
    def __init__(self, scale: float = SCALE, margin_m: float = MARGIN_M):
        self.scale, self.margin = scale, margin_m
        self.w = int(round((FIBA.length_m + 2 * margin_m) * scale))
        self.h = int(round((FIBA.width_m + 2 * margin_m) * scale))

    def px(self, x_m: float, y_m: float) -> tuple[int, int]:
        return (int(round((x_m + self.margin) * self.scale)), int(round(self.h - (y_m + self.margin) * self.scale)))

    def render(self, dens: np.ndarray, color: tuple[int, int, int], shots: list[dict]) -> Image.Image:
        img = np.full((self.h, self.w, 3), PANEL, np.uint8)
        x0, y0 = self.px(0, FIBA.width_m)
        x1, y1 = self.px(FIBA.length_m, 0)
        img[y0:y1, x0:x1] = COURT_FILL
        # density: one hue, court fill -> team colour -> lighter team colour, alpha rises with density
        d = cv2.resize(dens, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)[::-1]  # court y up = image y down
        base = np.array(COURT_FILL, np.float32)
        col = np.array(color, np.float32)
        light = np.clip(col * 0.55 + 255 * 0.45, 0, 255)
        t = d[..., None]
        ramp = np.where(t < 0.5, base + (col - base) * (t / 0.5), col + (light - col) * ((t - 0.5) / 0.5))
        alpha = np.clip(t * 1.6, 0, 1)
        region = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = (region * (1 - alpha) + ramp * alpha).astype(np.uint8)
        # lines
        for poly in polylines(FIBA):
            pts = np.array([self.px(x, y) for x, y in poly], np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, LINE, 2, cv2.LINE_AA)
        for hx, hy in FIBA.hoops:
            cv2.circle(img, self.px(hx, hy), int(0.225 * self.scale), (245, 158, 11), 2, cv2.LINE_AA)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        f = font(22, bold=True)
        for s in shots:
            xy = s.get("court_m")
            if xy is None:
                continue
            cx, cy = self.px(xy[0], xy[1])
            r = 13
            if s.get("made"):
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(255, 255, 255), width=2)
            else:
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
            if s.get("number") is not None:
                label = str(s["number"])
                tw = draw.textlength(label, font=f)
                draw.text((cx - tw / 2, cy - 12), label, font=f, fill=(255, 255, 255) if s.get("made") else INK)
        return pil


def shots_for_team(events: dict | None, cal: Calibration, team: int) -> list[dict]:
    out = []
    for s in (events or {}).get("shots", []):
        if int(s.get("team", -1)) != team:
            continue
        s = dict(s)
        foot = s.get("shooter_foot")
        frame = s.get("release_frame") if s.get("release_frame") is not None else s.get("frame")
        xy = cal.project(int(frame), [foot])[0] if foot and frame is not None else np.array([np.nan, np.nan])
        s["court_m"] = [float(xy[0]), float(xy[1])] if np.isfinite(xy).all() and cal.on_court(xy, 1.5)[0] else None
        key = s.get("player_key") or ""
        s["number"] = int(key[1:]) if len(key) > 1 and key[1:].isdigit() else None
        out.append(s)
    return out


def broadcast_frame(panels: dict[int, Image.Image], teams: dict[int, dict], info: dict, shots: dict[int, list], clip: str) -> Image.Image:
    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    f_brand, f_title, f_sub, f_small = font(44, True), font(34, True), font(28), font(28)
    draw.text((60, 40), BRAND.upper(), font=f_brand, fill=INK)
    draw.text((60, 96), "Heat map  ·  where each team played, and where it shot from", font=f_sub, fill=MUTED)
    pw = 880
    for i, team in enumerate((0, 1)):
        x = 60 + i * (pw + 40)
        y = 170
        draw.rounded_rectangle([x, y, x + pw, y + 640], radius=18, fill=PANEL)
        col = hex_rgb(teams[team]["color"])
        draw.rectangle([x + 28, y + 30, x + 40, y + 62], fill=col)
        draw.text((x + 56, y + 26), teams[team]["name"], font=f_title, fill=INK)
        panel = panels[team].resize((pw - 56, int((pw - 56) * panels[team].height / panels[team].width)), Image.LANCZOS)
        img.paste(panel, (x + 28, y + 90))
        made = sum(1 for s in shots[team] if s.get("made"))
        placed = sum(1 for s in shots[team] if s.get("court_m") is not None)
        draw.text((x + 28, y + 90 + panel.height + 18),
                  f"{len(shots[team])} shots, {made} made, {placed} placed on the court", font=f_small, fill=MUTED)
    # legend
    ly = 860
    draw.ellipse([60, ly, 86, ly + 26], fill=hex_rgb(teams[0]["color"]), outline=(255, 255, 255), width=2)
    draw.text((100, ly - 4), "made", font=f_small, fill=INK)
    draw.ellipse([210, ly, 236, ly + 26], outline=hex_rgb(teams[0]["color"]), width=3)
    draw.text((250, ly - 4), "missed", font=f_small, fill=INK)
    draw.text((380, ly - 4), "brighter = more time spent there", font=f_small, fill=MUTED)
    draw.text((60, 1000), f"{clip}  ·  {info['positions_used']} player positions, camera calibrated per frame, "
              f"{info['frames_uncertain']} uncertain and {info['frames_uncalibrated']} close-up frames left out",
              font=f_small, fill=MUTED)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, default=ROOT / "out" / "game10" / "tracks.jsonl")
    ap.add_argument("--calib", type=Path, default=ROOT / "out" / "court_calib_game10.json")
    ap.add_argument("--events", type=Path, default=ROOT / "out" / "game10" / "events.json")
    ap.add_argument("--config", type=Path, default=ROOT / "broadcast" / "config.json")
    ap.add_argument("--out", type=Path, default=ROOT / "broadcast" / "assets")
    args = ap.parse_args(argv)

    teams = load_teams(args.config)
    cal = load_calibration(args.calib)
    events = json.loads(args.events.read_text()) if args.events.exists() else None
    positions, info = team_positions(args.tracks, cal)
    args.out.mkdir(parents=True, exist_ok=True)
    panel = CourtPanel()
    panels, shots = {}, {}
    for team, letter in ((0, "A"), (1, "B")):
        shots[team] = shots_for_team(events, cal, team)
        panels[team] = panel.render(density(positions[team]), hex_rgb(teams[team]["color"]), shots[team])
        panels[team].save(args.out / f"heat_map_{letter}.png")
        print(f"{teams[team]['name']}: {len(positions[team])} positions, {len(shots[team])} shots -> {args.out / f'heat_map_{letter}.png'}")
    frame = broadcast_frame(panels, teams, info, shots, cal.meta.get("clip", ""))
    frame.save(args.out / "heat_map.png")
    (args.out / "heat_map_info.json").write_text(json.dumps({**info, "teams": teams, "shots": {t: len(v) for t, v in shots.items()}}, indent=1))
    print(f"frame -> {args.out / 'heat_map.png'}  ({info})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
