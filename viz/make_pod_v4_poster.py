#!/usr/bin/env python3
"""Compose the FollowCam Pod v4 poster from the Blender walkthrough frames.

Replaces viz/followcam_assembled_poster.png, which showed the superseded column
clamp + fork arm rig. Every panel here is a still lifted straight out of
cad/blender/pod_v4/renders/, which build_v4.py writes from the same geometry
that exports the printable STLs -- so the poster cannot drift from the parts.

Run from the repo root:

    python3 viz/make_pod_v4_poster.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RENDERS = ROOT / "cad" / "blender" / "pod_v4" / "renders"
OUT = ROOT / "viz" / "followcam_pod_v4_poster.png"

# Canvas ---------------------------------------------------------------------
W, H = 2400, 1700
MARGIN = 60
HEADER_H = 158
FOOTER_H = 120
GUTTER = 40

PAPER = (247, 246, 243)
INK = (22, 24, 28)
MUTED = (108, 112, 120)
ACCENT = (223, 106, 68)      # the coral the v4 renders use for printed parts
CHIP_BG = (252, 252, 251)
PANEL_EDGE = (206, 204, 199)

FONT_DIRS = (
    Path("/System/Library/Fonts/Supplemental"),
    Path("/Library/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
)
BOLD_NAMES = ("Arial Bold.ttf", "DejaVuSans-Bold.ttf")
REG_NAMES = ("Arial.ttf", "DejaVuSans.ttf")


def font(names: tuple[str, ...], size: int) -> ImageFont.FreeTypeFont:
    for directory in FONT_DIRS:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size)


# Panels ---------------------------------------------------------------------
# Callout anchors are normalised to the source frame, so they follow the image
# when the panel is rescaled. `chip` is the label's top-left, `at` is the point
# on the model the leader line lands on.
PANELS = (
    {
        "frame": "01_hero/f0135.png",
        "caption": "Assembled — pod on the tripod, leg case on the leg",
        "callouts": (
            ("Phone in the portrait cradle", (0.06, 0.12), (0.49, 0.21)),
            ("Rotor deck — turns with the phone", (0.63, 0.33), (0.52, 0.43)),
            ("Stator shell — bolts to the tripod", (0.04, 0.62), (0.47, 0.55)),
            ("Leg case — Uno R3 + 4 × AA", (0.66, 0.90), (0.60, 0.79)),
        ),
    },
    {
        "frame": "04_exploded/f0112.png",
        "caption": "Exploded — the stack, bottom to top",
        "callouts": (
            ("06  portrait cradle", (0.03, 0.15), (0.47, 0.23)),
            ("04  rotor", (0.70, 0.52), (0.53, 0.60)),
            ("03  servo carrier + MG996R", (0.02, 0.66), (0.47, 0.71)),
            ("05  drive dog", (0.72, 0.84), (0.63, 0.78)),
            ("01  stator shell", (0.03, 0.89), (0.47, 0.86)),
        ),
    },
    {
        "frame": "05_landscape/f0080.png",
        "caption": "Landscape — same pod, one thumbscrew swaps the cradle",
        "callouts": (),
    },
    {
        "frame": "06_leg_case/f0080.png",
        "caption": "Leg case — controller and battery clip to a tripod leg",
        "callouts": (),
    },
)

TITLE = "FollowCam Pod v4"
SUBTITLE = "— direct-drive pan pod, replaces the tripod head"
FOOTER = (
    "Ø88 mm printed slew ring  ·  12 printed parts  ·  MG996R direct drive, 1:1  "
    "·  geometry exported from build_v4.py  ·  CODE Hackathon 2026"
)


def rounded(draw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_callout(draw, label, chip_xy, point_xy, f_chip, bounds):
    """White label chip plus a leader line to the point it names.

    The chip is clamped inside `bounds` so a long label can never hang off the
    panel edge -- the normalised anchor positions a chip, it does not size it.
    """
    pad_x, pad_y = 18, 11
    left, top = chip_xy
    bbox = draw.textbbox((0, 0), label, font=f_chip)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    chip_w, chip_h = tw + 2 * pad_x, th + 2 * pad_y

    inset = 10
    bx0, by0, bx1, by1 = bounds
    left = max(bx0 + inset, min(left, bx1 - inset - chip_w))
    top = max(by0 + inset, min(top, by1 - inset - chip_h))
    right, bottom = left + chip_w, top + chip_h

    px, py = point_xy
    # Leave from the chip edge that faces the point, so the line never crosses
    # its own label.
    start_x = right if px > right else (left if px < left else (left + right) // 2)
    start_y = max(top, min(bottom, py)) if left <= px <= right else (top + bottom) // 2

    draw.line((start_x, start_y, px, py), fill=ACCENT, width=3)
    draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=ACCENT)
    rounded(draw, (left, top, right, bottom), 8, fill=CHIP_BG, outline=PANEL_EDGE)
    draw.text((left + pad_x, top + pad_y - bbox[1]), label, font=f_chip, fill=INK)


def main() -> None:
    missing = [p["frame"] for p in PANELS if not (RENDERS / p["frame"]).exists()]
    if missing:
        raise SystemExit(
            "missing render frames: "
            + ", ".join(missing)
            + "\nregenerate with: blender --background --python "
            "cad/blender/pod_v4/build_v4.py"
        )

    f_title = font(BOLD_NAMES, 76)
    f_sub = font(REG_NAMES, 40)
    f_caption = font(BOLD_NAMES, 30)
    f_chip = font(BOLD_NAMES, 29)
    f_footer = font(REG_NAMES, 28)

    poster = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(poster)

    # Header
    draw.text((MARGIN, 44), TITLE, font=f_title, fill=INK)
    title_w = draw.textlength(TITLE, font=f_title)
    draw.text((MARGIN + title_w + 26, 70), SUBTITLE, font=f_sub, fill=MUTED)
    draw.rectangle((MARGIN, HEADER_H, W - MARGIN, HEADER_H + 6), fill=ACCENT)

    # 2 x 2 grid of panels
    caption_h = 54
    grid_top = HEADER_H + 46
    grid_bottom = H - FOOTER_H
    cell_w = (W - 2 * MARGIN - GUTTER) // 2
    cell_h = (grid_bottom - grid_top - GUTTER) // 2
    img_h = cell_h - caption_h
    img_w = cell_w

    for index, panel in enumerate(PANELS):
        col, row = index % 2, index // 2
        x0 = MARGIN + col * (cell_w + GUTTER)
        y0 = grid_top + row * (cell_h + GUTTER)

        with Image.open(RENDERS / panel["frame"]) as src:
            shot = src.convert("RGB")
        # Cover-fit the 16:9 frame into the cell, centre-cropped.
        scale = max(img_w / shot.width, img_h / shot.height)
        resized = shot.resize(
            (round(shot.width * scale), round(shot.height * scale)), Image.LANCZOS
        )
        off_x = (resized.width - img_w) // 2
        off_y = (resized.height - img_h) // 2
        poster.paste(resized.crop((off_x, off_y, off_x + img_w, off_y + img_h)), (x0, y0))
        draw.rectangle((x0, y0, x0 + img_w - 1, y0 + img_h - 1), outline=PANEL_EDGE)

        for label, chip, point in panel["callouts"]:
            chip_xy = (x0 + int(chip[0] * img_w), y0 + int(chip[1] * img_h))
            point_xy = (
                x0 + int(point[0] * resized.width) - off_x,
                y0 + int(point[1] * resized.height) - off_y,
            )
            draw_callout(
                draw,
                label,
                chip_xy,
                point_xy,
                f_chip,
                (x0, y0, x0 + img_w, y0 + img_h),
            )

        cap_y = y0 + img_h + 15
        draw.rectangle((x0, cap_y + 4, x0 + 6, cap_y + 32), fill=ACCENT)
        draw.text((x0 + 20, cap_y), panel["caption"], font=f_caption, fill=INK)

    # Footer
    draw.rectangle((MARGIN, H - FOOTER_H + 16, W - MARGIN, H - FOOTER_H + 18), fill=PANEL_EDGE)
    draw.text((MARGIN, H - FOOTER_H + 46), FOOTER, font=f_footer, fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    poster.save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)}  {poster.width}x{poster.height}")


if __name__ == "__main__":
    main()
