#!/usr/bin/env python3
"""Render the 11-page FollowCam LEGO-style assembly manual."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


WIDTH = 1920
HEIGHT = 1080
SCALE = 2

BG = "#FAFAF8"
WHITE = "#FFFFFF"
DARK = "#1B1C1E"
MUTED = "#6B6B6B"
FOOTER = "#9A9A9A"
ORANGE = "#EB6414"

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "manual_assets"
PAGES_DIR = Path(__file__).resolve().parent / "pages"

HELVETICA = "/System/Library/Fonts/Helvetica.ttc"
MENLO = "/System/Library/Fonts/Menlo.ttc"

SAFE_LEFT = 96
SAFE_RIGHT = 1824
SAFE_TOP = 60
SAFE_BOTTOM = 1026
BADGE_BOX = (SAFE_LEFT, SAFE_TOP, SAFE_LEFT + 110, SAFE_TOP + 110)
TITLE_X = 234
INSTRUCTION_Y = 121
MAIN_TOP = 270
MAIN_BOTTOM = 950
PARTS_LEFT = 1504


@dataclass(frozen=True)
class PageSpec:
    title: str
    instruction: str
    media: tuple[str, ...]
    parts: tuple[tuple[str, str], ...] = ()
    wiring: bool = False
    phone: bool = False
    note: str | None = None
    arrow: str | None = None


PAGES: tuple[PageSpec, ...] = (
    PageSpec(
        "FollowCam - build it",
        "Lay out the complete kit before you begin.",
        ("IMG_6614.jpg",),
        (
            ("printed column clamp", "x1"),
            ("printed fork arm", "x1"),
            ("metal-gear servo", "x1"),
            ("servo horn + screws", "x1 set"),
            ("Arduino Uno R3", "x1"),
            ("USB cable", "x1"),
            ("M3 bolts", "x2"),
            ("zip ties", "x6"),
            ("tripod", "x1"),
            ("phone", "x1"),
        ),
    ),
    PageSpec(
        "Flash the firmware",
        "Arduino IDE -> open software/servo_pan/servo_pan.ino -> board Arduino Uno -> Upload.",
        ("IMG_6615.jpg",),
    ),
    PageSpec(
        "Wire the servo",
        "orange->9, red->5V, brown->GND.",
        ("IMG_6615.jpg",),
        wiring=True,
    ),
    PageSpec(
        "Bench test before mounting",
        "python3 software/servo_test.py -> press s -> one smooth 40-140 sweep.",
        ("IMG_6615.jpg",),
    ),
    PageSpec(
        "Clamp onto the column",
        "Raise the column. Ring around it just below the head, shelf pointing the SAME side as the handle. 2x M3 through the slit bosses - snug.",
        ("IMG_6619.jpg", "IMG_6603.jpg"),
        (("column clamp", "x1"), ("M3 bolt", "x2")),
        arrow="clamp_to_column",
    ),
    PageSpec(
        "Seat the servo ON the rails",
        "Big servo does not drop in the pocket - it sits ON TOP of the two rails, output shaft UP and toward the column. 2 zip ties through the ledge slots, cinch tight.",
        ("IMG_6619.jpg", "IMG_6621.jpg"),
        (("metal-gear servo", "x1"), ("zip ties", "x2")),
        note="pocket was sized for SG90 - rails carry it, ties hold it",
    ),
    PageSpec(
        "Center, then fit the arm",
        "servo_test.py -> press c (90 deg). Only then press the fork arm onto the horn pointing straight at the pan handle; fix the horn screw.",
        ("IMG_6616.jpg",),
        (("fork arm", "x1"), ("horn + screw", "x1")),
    ),
    PageSpec(
        "Catch the handle shaft",
        "Drop the tall U-fork over the chrome shaft from below - the window is forgiving, +/-10mm everywhere.",
        ("IMG_6603_annotated.jpg",),
        arrow="chrome_shaft",
    ),
    PageSpec(
        "Friction + phone",
        "Loosen pan friction to a two-finger glide. Lock tilt HARD. Phone into the clamp, camera facing the court.",
        ("IMG_6604.jpg",),
    ),
    PageSpec(
        "Link and go",
        "Hotspot on -> laptop: python3 software/pan_bridge.py --port /dev/cu.usbmodem* -> app: settings, laptop IP, RIG turns blue -> tap a player -> record.",
        ("app_screen.png",),
        phone=True,
    ),
    PageSpec(
        "The goal",
        "Every clip you record trains the model. Film wide, vary distance and speed.",
        ("followcam_assembled_poster.png",),
    ),
)


def sp(value: float) -> int:
    return int(round(value * SCALE))


def sp_box(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(sp(value) for value in box)  # type: ignore[return-value]


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = MENLO if mono else HELVETICA
    index = 1 if bold else 0
    return ImageFont.truetype(path, sp(size), index=index)


TITLE_FONT = font(44, bold=True)
INSTRUCTION_FONT = font(28)
BADGE_FONT = font(58, bold=True)
BADGE_SMALL_FONT = font(25, bold=True)
PARTS_HEADING_FONT = font(19, bold=True, mono=True)
PART_FONT = font(20, mono=True)
PART_QTY_FONT = font(18, bold=True, mono=True)
FOOTER_FONT = font(20, mono=True)
NOTE_FONT = font(20, bold=True)


def text_width(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont) -> float:
    return draw.textlength(text, font=text_font) / SCALE


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
    max_width: float,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(draw, candidate, text_font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def load_asset(name: str) -> Image.Image:
    with Image.open(ASSETS / name) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def draw_header(draw: ImageDraw.ImageDraw, page_number: int, spec: PageSpec) -> None:
    draw.rectangle(sp_box(BADGE_BOX), fill=ORANGE)
    badge_text = "FC-01" if page_number == 0 else str(page_number)
    badge_font = BADGE_SMALL_FONT if page_number == 0 else BADGE_FONT
    draw.text(
        (sp((BADGE_BOX[0] + BADGE_BOX[2]) / 2), sp((BADGE_BOX[1] + BADGE_BOX[3]) / 2)),
        badge_text,
        font=badge_font,
        fill=WHITE,
        anchor="mm",
    )

    max_right = PARTS_LEFT - 32 if spec.parts else SAFE_RIGHT
    draw.text((sp(TITLE_X), sp(SAFE_TOP + 1)), spec.title, font=TITLE_FONT, fill=DARK)
    instruction_lines = wrap_text(draw, spec.instruction, INSTRUCTION_FONT, max_right - TITLE_X)
    line_height = 35
    for line_number, line in enumerate(instruction_lines):
        draw.text(
            (sp(TITLE_X), sp(INSTRUCTION_Y + line_number * line_height)),
            line,
            font=INSTRUCTION_FONT,
            fill=MUTED,
        )


def draw_parts_strip(draw: ImageDraw.ImageDraw, parts: tuple[tuple[str, str], ...]) -> None:
    strip_box = (PARTS_LEFT, SAFE_TOP, SAFE_RIGHT, MAIN_BOTTOM)
    draw.rectangle(sp_box(strip_box), outline=DARK, width=sp(1))
    draw.text(
        (sp(PARTS_LEFT + 18), sp(SAFE_TOP + 18)),
        "PARTS FOR THIS STEP",
        font=PARTS_HEADING_FONT,
        fill=DARK,
    )

    start_y = SAFE_TOP + 78
    row_height = 78 if len(parts) >= 8 else 92
    name_width = 210
    for row, (name, quantity) in enumerate(parts):
        y = start_y + row * row_height
        lines = wrap_text(draw, name, PART_FONT, name_width)
        for line_number, line in enumerate(lines[:2]):
            draw.text(
                (sp(PARTS_LEFT + 18), sp(y + line_number * 27)),
                line,
                font=PART_FONT,
                fill=DARK,
            )
        draw.text(
            (sp(SAFE_RIGHT - 18), sp(y + 1)),
            quantity,
            font=PART_QTY_FONT,
            fill=DARK,
            anchor="ra",
        )


def draw_media(
    page: Image.Image,
    media: Image.Image,
    slot: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Contain media in a white 12px mat with a thin dark outline."""
    pad = 12
    slot_w = slot[2] - slot[0]
    slot_h = slot[3] - slot[1]
    max_content_w = max(1, slot_w - 2 * pad)
    max_content_h = max(1, slot_h - 2 * pad)
    ratio = min(max_content_w / media.width, max_content_h / media.height)
    content_w = media.width * ratio
    content_h = media.height * ratio
    frame_w = content_w + 2 * pad
    frame_h = content_h + 2 * pad
    frame_x = slot[0] + (slot_w - frame_w) / 2
    frame_y = slot[1] + (slot_h - frame_h) / 2
    frame_box = (frame_x, frame_y, frame_x + frame_w, frame_y + frame_h)

    draw = ImageDraw.Draw(page)
    draw.rectangle(sp_box(frame_box), fill=WHITE, outline=DARK, width=sp(1))
    resized = media.resize((sp(content_w), sp(content_h)), Image.Resampling.LANCZOS)
    content_x = frame_x + pad
    content_y = frame_y + pad
    page.paste(resized, (sp(content_x), sp(content_y)))
    return (content_x, content_y, content_x + content_w, content_y + content_h)


def make_wiring_diagram() -> Image.Image:
    diagram = Image.new("RGB", (sp(900), sp(760)), WHITE)
    draw = ImageDraw.Draw(diagram)
    diagram_title = font(30, bold=True)
    label_font = font(22, bold=True, mono=True)
    small_font = font(19, mono=True)

    draw.text((sp(450), sp(42)), "SERVO WIRING", font=diagram_title, fill=DARK, anchor="mm")
    plug_box = (54, 205, 245, 570)
    uno_box = (620, 105, 845, 655)
    draw.rounded_rectangle(sp_box(plug_box), radius=sp(12), fill="#F1F1EF", outline=DARK, width=sp(3))
    draw.text((sp(150), sp(388)), "SERVO\nPLUG", font=label_font, fill=DARK, anchor="mm", align="center")
    draw.rounded_rectangle(sp_box(uno_box), radius=sp(15), fill="#E7F1F6", outline=DARK, width=sp(3))
    draw.text((sp(733), sp(590)), "UNO", font=label_font, fill=DARK, anchor="mm")

    wires = (
        ("orange", ORANGE, 265, 225, "PIN 9"),
        ("red", "#D62828", 385, 380, "5V"),
        ("brown", "#7A4B2A", 505, 535, "GND"),
    )
    for name, color, start_y, end_y, pin in wires:
        draw.ellipse(sp_box((220, start_y - 8, 236, start_y + 8)), fill=color)
        draw.line(
            [(sp(236), sp(start_y)), (sp(440), sp(start_y)), (sp(620), sp(end_y))],
            fill=color,
            width=sp(9),
            joint="curve",
        )
        draw.ellipse(sp_box((612, end_y - 8, 628, end_y + 8)), fill=color)
        draw.text((sp(285), sp(start_y - 38)), name, font=small_font, fill=color)
        draw.text((sp(650), sp(end_y)), pin, font=label_font, fill=DARK, anchor="lm")

    return diagram


def make_phone_asset() -> Image.Image:
    screenshot = load_asset("app_screen.png")
    asset = Image.new("RGB", (sp(560), sp(1220)), BG)
    draw = ImageDraw.Draw(asset)
    outer = (20, 10, 540, 1210)
    screen_slot = (43, 50, 517, 1166)
    draw.rounded_rectangle(sp_box(outer), radius=sp(62), fill=DARK, outline=DARK, width=sp(2))

    slot_w = screen_slot[2] - screen_slot[0]
    slot_h = screen_slot[3] - screen_slot[1]
    ratio = min(slot_w / screenshot.width, slot_h / screenshot.height)
    image_w = screenshot.width * ratio
    image_h = screenshot.height * ratio
    image_x = screen_slot[0] + (slot_w - image_w) / 2
    image_y = screen_slot[1] + (slot_h - image_h) / 2
    resized = screenshot.resize((sp(image_w), sp(image_h)), Image.Resampling.LANCZOS)
    mask = Image.new("L", resized.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, resized.width - 1, resized.height - 1), radius=sp(28), fill=255)
    asset.paste(resized, (sp(image_x), sp(image_y)), mask)
    draw.rounded_rectangle(sp_box((220, 25, 340, 39)), radius=sp(7), fill="#4B4C50")
    draw.rounded_rectangle(sp_box((225, 1183, 335, 1191)), radius=sp(4), fill="#4B4C50")
    return asset


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    width: float = 18,
    head_length: float = 46,
    head_width: float = 40,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    base = (end[0] - ux * head_length, end[1] - uy * head_length)
    draw.line(
        [(sp(start[0]), sp(start[1])), (sp(base[0]), sp(base[1]))],
        fill=ORANGE,
        width=sp(width),
    )
    head = [
        (sp(end[0]), sp(end[1])),
        (sp(base[0] + px * head_width / 2), sp(base[1] + py * head_width / 2)),
        (sp(base[0] - px * head_width / 2), sp(base[1] - py * head_width / 2)),
    ]
    draw.polygon(head, fill=ORANGE)


def draw_note_chip(draw: ImageDraw.ImageDraw, note: str) -> None:
    chip_x = SAFE_LEFT
    chip_y = 908
    padding_x = 18
    padding_y = 10
    chip_w = text_width(draw, note, NOTE_FONT) + 2 * padding_x
    chip_box = (chip_x, chip_y, chip_x + chip_w, chip_y + 45)
    draw.rounded_rectangle(sp_box(chip_box), radius=sp(8), fill=ORANGE)
    draw.text(
        (sp(chip_x + padding_x), sp(chip_y + padding_y - 1)),
        note,
        font=NOTE_FONT,
        fill=WHITE,
    )


def draw_footer(draw: ImageDraw.ImageDraw, page_number: int) -> None:
    footer = f"FollowCam assembly - page {page_number}/10"
    bbox = draw.textbbox((0, 0), footer, font=FOOTER_FONT)
    width = (bbox[2] - bbox[0]) / SCALE
    height = (bbox[3] - bbox[1]) / SCALE
    draw.text(
        (sp(SAFE_RIGHT - width), sp(SAFE_BOTTOM - height - 2)),
        footer,
        font=FOOTER_FONT,
        fill=FOOTER,
    )


def render_page(page_number: int, spec: PageSpec) -> Image.Image:
    page = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BG)
    draw = ImageDraw.Draw(page)
    draw_header(draw, page_number, spec)
    if spec.parts:
        draw_parts_strip(draw, spec.parts)

    main_right = PARTS_LEFT - 32 if spec.parts else SAFE_RIGHT
    media_bottom = 891 if spec.note else MAIN_BOTTOM
    main_box = (SAFE_LEFT, MAIN_TOP, main_right, media_bottom)
    placements: list[tuple[float, float, float, float]] = []

    if spec.phone:
        placements.append(draw_media(page, make_phone_asset(), main_box))
    elif spec.wiring:
        gap = 32
        slot_width = (main_right - SAFE_LEFT - gap) / 2
        left_slot = (SAFE_LEFT, MAIN_TOP, SAFE_LEFT + slot_width, media_bottom)
        right_slot = (SAFE_LEFT + slot_width + gap, MAIN_TOP, main_right, media_bottom)
        placements.append(draw_media(page, load_asset(spec.media[0]), left_slot))
        placements.append(draw_media(page, make_wiring_diagram(), right_slot))
    elif len(spec.media) == 2:
        gap = 32
        slot_width = (main_right - SAFE_LEFT - gap) / 2
        slots = (
            (SAFE_LEFT, MAIN_TOP, SAFE_LEFT + slot_width, media_bottom),
            (SAFE_LEFT + slot_width + gap, MAIN_TOP, main_right, media_bottom),
        )
        for name, slot in zip(spec.media, slots):
            placements.append(draw_media(page, load_asset(name), slot))
    else:
        placements.append(draw_media(page, load_asset(spec.media[0]), main_box))

    draw = ImageDraw.Draw(page)
    if spec.arrow == "clamp_to_column":
        left, right = placements
        start = (left[2] - 28, left[1] + (left[3] - left[1]) * 0.48)
        end = (right[0] + (right[2] - right[0]) * 0.48, right[1] + (right[3] - right[1]) * 0.66)
        draw_arrow(draw, start, end)
    elif spec.arrow == "chrome_shaft":
        photo = placements[0]
        target = (
            photo[0] + (photo[2] - photo[0]) * 0.45,
            photo[1] + (photo[3] - photo[1]) * 0.67,
        )
        start = (target[0] + 155, target[1] - 120)
        draw_arrow(draw, start, target)

    if spec.note:
        draw_note_chip(draw, spec.note)
    draw_footer(draw, page_number)
    return page.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def main() -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    for page_number, spec in enumerate(PAGES):
        output = PAGES_DIR / f"page_{page_number:02d}.png"
        render_page(page_number, spec).save(output, format="PNG", optimize=True)
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
