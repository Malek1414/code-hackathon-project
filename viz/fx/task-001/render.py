#!/usr/bin/env python3
"""Render the FollowCam court-coverage panel as 240 antialiased PNG frames."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 960
HEIGHT = 1080
SCALE = 2
FPS = 30
FRAME_COUNT = 240
FRAMES_DIR = Path(__file__).resolve().parent / "frames"

BG = "#fafafa"
DARK = "#19191c"
ORANGE = "#eb6414"
ORANGE_DARK = "#b9440b"
BLUE = "#2661e6"
COURT = "#f3ead9"
COURT_LINE = "#b9ad97"
RED = "#e4312b"
WHITE = "#ffffff"

FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"
MONO_PATH = "/System/Library/Fonts/Menlo.ttc"

COURT_BOX = (150, 90, 890, 990)
ORIGIN = (150.0, 540.0)


def sp(value: float) -> int:
    """Scale a logical coordinate to the 2x working canvas."""
    return int(round(value * SCALE))


def points_scaled(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(sp(x), sp(y)) for x, y in points]


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = MONO_PATH if mono else FONT_PATH
    index = 0 if mono else (1 if bold else 0)
    return ImageFont.truetype(path, sp(size), index=index)


TITLE_FONT = font(40, bold=True)
READOUT_FONT = font(35, mono=True)
LABEL_FONT = font(20, bold=True)
GAUGE_FONT = font(17)
REC_FONT = font(28, bold=True)
CHIP_FONT = font(20, bold=True)


def position_at(t: float) -> tuple[float, float]:
    theta = 90.0 + 50.0 * math.sin(2.0 * math.pi * t / 8.0)
    aim = math.radians(theta - 90.0)
    depth = 380.0 + 190.0 * math.sin(2.0 * math.pi * t / 4.0 + 1.1)
    wobble = 18.0 * math.sin(2.0 * math.pi * t / 1.6)
    ux, uy = math.cos(aim), -math.sin(aim)
    px, py = math.sin(aim), math.cos(aim)
    return ORIGIN[0] + depth * ux + wobble * px, ORIGIN[1] + depth * uy + wobble * py


def clipped_layer(layer: Image.Image, mask: Image.Image) -> Image.Image:
    alpha = layer.getchannel("A")
    layer.putalpha(Image.composite(alpha, Image.new("L", layer.size, 0), mask))
    return layer


def draw_court_floor(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        tuple(sp(v) for v in COURT_BOX),
        radius=sp(6),
        fill=COURT,
    )


def draw_wedge(image: Image.Image, aim: float) -> None:
    ox, oy = ORIGIN
    radius = 820.0
    half_fov = math.radians(33.0)
    wedge_points = [(ox, oy)]
    for index in range(67):
        angle = aim - half_fov + (2.0 * half_fov * index / 66.0)
        wedge_points.append((ox + radius * math.cos(angle), oy - radius * math.sin(angle)))

    fill_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_layer)
    fill_draw.polygon(points_scaled(wedge_points), fill=(38, 97, 230, 46))

    edge_points = []
    for side in (-half_fov, half_fov):
        edge_points.append(
            [
                (ox, oy),
                (ox + radius * math.cos(aim + side), oy - radius * math.sin(aim + side)),
            ]
        )
    for edge in edge_points:
        fill_draw.line(points_scaled(edge), fill=BLUE, width=sp(2))

    dash_length = 13.0
    gap_length = 10.0
    cursor = 22.0
    while cursor < radius:
        end = min(radius, cursor + dash_length)
        segment = [
            (ox + cursor * math.cos(aim), oy - cursor * math.sin(aim)),
            (ox + end * math.cos(aim), oy - end * math.sin(aim)),
        ]
        fill_draw.line(points_scaled(segment), fill=BLUE, width=sp(1))
        cursor += dash_length + gap_length

    court_mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(court_mask).rectangle(tuple(sp(v) for v in COURT_BOX), fill=255)
    image.alpha_composite(clipped_layer(fill_layer, court_mask))


def draw_court_lines(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = (sp(v) for v in COURT_BOX)
    line_width = sp(3)
    cx = sp((COURT_BOX[0] + COURT_BOX[2]) / 2)
    cy = sp((COURT_BOX[1] + COURT_BOX[3]) / 2)

    draw.rounded_rectangle((x0, y0, x1, y1), radius=sp(6), outline=COURT_LINE, width=line_width)
    draw.line((x0, cy, x1, cy), fill=COURT_LINE, width=line_width)
    draw.ellipse((cx - sp(72), cy - sp(72), cx + sp(72), cy + sp(72)), outline=COURT_LINE, width=line_width)
    draw.ellipse((cx - sp(7), cy - sp(7), cx + sp(7), cy + sp(7)), fill=COURT_LINE)

    key_left, key_right = sp(414), sp(626)
    top_ft, bottom_ft = sp(280), sp(800)
    draw.rectangle((key_left, y0, key_right, top_ft), outline=COURT_LINE, width=line_width)
    draw.rectangle((key_left, bottom_ft, key_right, y1), outline=COURT_LINE, width=line_width)
    ft_radius = sp(76)
    draw.ellipse((cx - ft_radius, top_ft - ft_radius, cx + ft_radius, top_ft + ft_radius), outline=COURT_LINE, width=line_width)
    draw.ellipse((cx - ft_radius, bottom_ft - ft_radius, cx + ft_radius, bottom_ft + ft_radius), outline=COURT_LINE, width=line_width)

    corner_left, corner_right = sp(240), sp(800)
    top_join, bottom_join = sp(190), sp(890)
    draw.line((corner_left, y0, corner_left, top_join), fill=COURT_LINE, width=line_width)
    draw.line((corner_right, y0, corner_right, top_join), fill=COURT_LINE, width=line_width)
    draw.arc((sp(240), sp(-85), sp(800), sp(465)), 0, 180, fill=COURT_LINE, width=line_width)
    draw.line((corner_left, bottom_join, corner_left, y1), fill=COURT_LINE, width=line_width)
    draw.line((corner_right, bottom_join, corner_right, y1), fill=COURT_LINE, width=line_width)
    draw.arc((sp(240), sp(615), sp(800), sp(1165)), 180, 360, fill=COURT_LINE, width=line_width)

    backboard_half = sp(26)
    hoop_radius = sp(10)
    for board_y, hoop_y in ((116, 139), (964, 941)):
        by = sp(board_y)
        hy = sp(hoop_y)
        draw.line((cx - backboard_half, by, cx + backboard_half, by), fill=COURT_LINE, width=sp(4))
        draw.ellipse((cx - hoop_radius, hy - hoop_radius, cx + hoop_radius, hy + hoop_radius), outline=COURT_LINE, width=line_width)
        draw.line((cx, by, cx, hy - hoop_radius), fill=COURT_LINE, width=sp(2))


def draw_trail_and_ball(image: Image.Image, t: float) -> None:
    trail_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    trail_draw = ImageDraw.Draw(trail_layer)
    trail_frames = 36
    positions = [position_at(t - offset / FPS) for offset in range(trail_frames, -1, -1)]
    for index in range(1, len(positions)):
        progress = index / (len(positions) - 1)
        alpha = int(round(18 + 150 * progress**1.5))
        width = sp(2.0 + 2.0 * progress)
        trail_draw.line(
            points_scaled([positions[index - 1], positions[index]]),
            fill=(235, 100, 20, alpha),
            width=width,
        )
    image.alpha_composite(trail_layer)

    x, y = position_at(t)
    draw = ImageDraw.Draw(image)
    r = sp(13)
    draw.ellipse((sp(x) - r, sp(y) - r, sp(x) + r, sp(y) + r), fill=ORANGE, outline=ORANGE_DARK, width=sp(2))
    shine = sp(3)
    draw.ellipse((sp(x - 4) - shine, sp(y - 4) - shine, sp(x - 4) + shine, sp(y - 4) + shine), fill="#ffad67")


def draw_gauge_and_tripod(image: Image.Image, theta: float) -> None:
    draw = ImageDraw.Draw(image)
    ox, oy = ORIGIN
    gauge_radius = 60.0

    arc_points = []
    for index in range(61):
        phi = math.radians(-50.0 + 100.0 * index / 60.0)
        arc_points.append((ox + gauge_radius * math.cos(phi), oy - gauge_radius * math.sin(phi)))
    draw.line(points_scaled(arc_points), fill=COURT_LINE, width=sp(2))

    for value in range(40, 141, 10):
        phi = math.radians(value - 90.0)
        inner = gauge_radius - (7.0 if value % 20 == 0 else 4.0)
        outer = gauge_radius + 3.0
        draw.line(
            points_scaled(
                [
                    (ox + inner * math.cos(phi), oy - inner * math.sin(phi)),
                    (ox + outer * math.cos(phi), oy - outer * math.sin(phi)),
                ]
            ),
            fill=COURT_LINE,
            width=sp(2),
        )

    aim = math.radians(theta - 90.0)
    needle_start = 25.0
    needle_end = gauge_radius + 7.0
    draw.line(
        points_scaled(
            [
                (ox + needle_start * math.cos(aim), oy - needle_start * math.sin(aim)),
                (ox + needle_end * math.cos(aim), oy - needle_end * math.sin(aim)),
            ]
        ),
        fill=ORANGE,
        width=sp(3),
    )

    draw.text((sp(184), sp(592)), "40°", font=GAUGE_FONT, fill=DARK, anchor="mm")
    draw.text((sp(184), sp(488)), "140°", font=GAUGE_FONT, fill=DARK, anchor="mm")

    draw.ellipse((sp(ox - 21), sp(oy - 21), sp(ox + 21), sp(oy + 21)), fill=ORANGE)
    draw.ellipse((sp(ox - 14), sp(oy - 14), sp(ox + 14), sp(oy + 14)), fill=DARK)
    draw.ellipse((sp(ox - 4), sp(oy - 4), sp(ox + 4), sp(oy + 4)), fill="#34343a")
    draw.text((sp(132), sp(515)), "FollowCam", font=LABEL_FONT, fill=DARK, anchor="rs")


def draw_hud(image: Image.Image, t: float, theta: float) -> None:
    draw = ImageDraw.Draw(image)
    draw.text((sp(70), sp(25)), "court coverage — top view", font=TITLE_FONT, fill=DARK, anchor="la")

    chip = (sp(672), sp(20), sp(906), sp(74))
    draw.rounded_rectangle(chip, radius=sp(12), fill=WHITE, outline="#e9e6df", width=sp(1))
    readout = f"servo {int(round(theta)):3d}°"
    draw.text((sp(890), sp(47)), readout, font=READOUT_FONT, fill=DARK, anchor="rm")

    pulse = 0.35 + 0.65 * abs(math.sin(math.pi * t))
    rec_alpha = int(round(255 * pulse))
    rec_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    rec_draw = ImageDraw.Draw(rec_layer)
    rec_draw.ellipse((sp(70), sp(1022), sp(92), sp(1044)), fill=(228, 49, 43, rec_alpha))
    image.alpha_composite(rec_layer)
    draw = ImageDraw.Draw(image)
    draw.text((sp(104), sp(1033)), "REC", font=REC_FONT, fill=DARK, anchor="lm")
    tracking_box = (sp(180), sp(1015), sp(320), sp(1050))
    draw.rounded_rectangle(tracking_box, radius=sp(8), fill=DARK)
    draw.text((sp(250), sp(1033)), "TRACKING", font=CHIP_FONT, fill=WHITE, anchor="mm")


def render_frame(frame_number: int) -> Image.Image:
    t = frame_number / FPS
    theta = 90.0 + 50.0 * math.sin(2.0 * math.pi * t / 8.0)
    aim = math.radians(theta - 90.0)

    image = Image.new("RGBA", (WIDTH * SCALE, HEIGHT * SCALE), BG)
    draw_court_floor(image)
    draw_wedge(image, aim)
    draw_court_lines(image)
    draw_trail_and_ball(image, t)
    draw_gauge_and_tripod(image, theta)
    draw_hud(image, t, theta)

    return image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def main() -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for frame_number in range(FRAME_COUNT):
        frame = render_frame(frame_number)
        frame.save(FRAMES_DIR / f"{frame_number:05d}.png", compress_level=2)
        if frame_number % 30 == 0 or frame_number == FRAME_COUNT - 1:
            print(f"rendered {frame_number + 1:03d}/{FRAME_COUNT}", flush=True)


if __name__ == "__main__":
    main()
