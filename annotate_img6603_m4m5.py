from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


SOURCE = Path("tripod-photos/IMG_6603.jpg")
OUTPUT = Path("annotated/IMG_6603_M4M5.jpg")

# Coordinates are on the upright 900 x 1200 working canvas.
A_X = 520  # center pole / pan axis
B_X = 443  # middle of the chrome rod at its head-side measurement point
C_Y = 884  # lower contour / bottom edge of the black head casting
D_Y = 842  # center height of the chrome rod

CYAN = "#00E5FF"
ORANGE = "#FF3D00"
DARK = "#1B1B1B"
WHITE = "#FFFFFF"


def load_font(size: int, *, bold: bool = True) -> ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise FileNotFoundError("No suitable TrueType font found")


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str,
    width: int = 4,
    dash: int = 18,
    gap: int = 12,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:
        direction = 1 if y2 >= y1 else -1
        cursor = y1
        while (cursor - y2) * direction <= 0:
            stop = cursor + direction * dash
            if (stop - y2) * direction > 0:
                stop = y2
            draw.line((x1, cursor, x2, stop), fill=fill, width=width)
            cursor += direction * (dash + gap)
    elif y1 == y2:
        direction = 1 if x2 >= x1 else -1
        cursor = x1
        while (cursor - x2) * direction <= 0:
            stop = cursor + direction * dash
            if (stop - x2) * direction > 0:
                stop = x2
            draw.line((cursor, y1, stop, y2), fill=fill, width=width)
            cursor += direction * (dash + gap)
    else:
        raise ValueError("Witness lines must be horizontal or vertical")


def chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    pad_x: int = 11,
    pad_y: int = 7,
    radius: int = 8,
) -> tuple[int, int, int, int]:
    x, y = xy
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    box = (
        x,
        y,
        x + right - left + pad_x * 2,
        y + bottom - top + pad_y * 2,
    )
    draw.rounded_rectangle(box, radius=radius, fill=DARK)
    draw.text(
        (x + pad_x - left, y + pad_y - top),
        text,
        font=font,
        fill=WHITE,
    )
    return box


def centered_letter_chip(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    letter: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), letter, font=font)
    text_w = right - left
    text_h = bottom - top
    box_w = max(38, text_w + 16)
    box_h = max(34, text_h + 12)
    x = round(center[0] - box_w / 2)
    y = round(center[1] - box_h / 2)
    draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=7, fill=DARK)
    draw.text(
        (center[0] - text_w / 2 - left, center[1] - text_h / 2 - top),
        letter,
        font=font,
        fill=WHITE,
    )


def horizontal_dimension_arrow(
    draw: ImageDraw.ImageDraw, x1: int, x2: int, y: int
) -> None:
    draw.line((x1, y, x2, y), fill=ORANGE, width=10)
    head_length = 22
    half_width = 15
    draw.polygon(
        ((x1, y), (x1 + head_length, y - half_width), (x1 + head_length, y + half_width)),
        fill=ORANGE,
    )
    draw.polygon(
        ((x2, y), (x2 - head_length, y - half_width), (x2 - head_length, y + half_width)),
        fill=ORANGE,
    )
    radius = 7
    for x in (x1, x2):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=ORANGE)


def vertical_dimension_arrow(
    draw: ImageDraw.ImageDraw, x: int, y1: int, y2: int
) -> None:
    y_top, y_bottom = sorted((y1, y2))
    draw.line((x, y_top, x, y_bottom), fill=ORANGE, width=10)
    head_length = 16
    half_width = 15
    draw.polygon(
        ((x, y_top), (x - half_width, y_top + head_length), (x + half_width, y_top + head_length)),
        fill=ORANGE,
    )
    draw.polygon(
        ((x, y_bottom), (x - half_width, y_bottom - head_length), (x + half_width, y_bottom - head_length)),
        fill=ORANGE,
    )
    radius = 7
    for y in (y_top, y_bottom):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=ORANGE)


def add_instruction_panel(image: Image.Image) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(overlay)
    panel_draw.rounded_rectangle(
        (14, 14, 886, 147),
        radius=12,
        fill=(27, 27, 27, round(255 * 0.88)),
    )
    image.alpha_composite(overlay)

    draw = ImageDraw.Draw(image)
    panel_font = load_font(26, bold=True)
    lines = (
        "M4: distance from center pole (A) to middle of chrome rod (B)",
        "M5: distance from bottom of head (C) down to chrome rod height (D)",
        "Both in mm. ±10mm is fine.",
    )
    y = 25
    for line in lines:
        draw.text((29, y), line, font=panel_font, fill=WHITE)
        y += 39


def main() -> None:
    raw = Image.open(SOURCE)
    source_size = raw.size
    source_exif = raw.getexif()
    source_icc = raw.info.get("icc_profile")

    upright = ImageOps.exif_transpose(raw).convert("RGBA")
    if upright.size != (900, 1200):
        raise ValueError(f"Unexpected upright source size: {upright.size}")

    draw = ImageDraw.Draw(upright)
    witness_top = 650
    witness_bottom = upright.height - 1
    horizontal_end = upright.width // 2

    dashed_line(draw, (A_X, witness_top), (A_X, witness_bottom), fill=CYAN)
    dashed_line(draw, (B_X, witness_top), (B_X, witness_bottom), fill=CYAN)
    dashed_line(draw, (0, C_Y), (horizontal_end, C_Y), fill=CYAN)
    dashed_line(draw, (0, D_Y), (horizontal_end, D_Y), fill=CYAN)

    horizontal_dimension_arrow(draw, B_X, A_X, 1138)
    vertical_dimension_arrow(draw, 68, D_Y, C_Y)

    letter_font = load_font(24, bold=True)
    centered_letter_chip(draw, (A_X, 1180), "A", letter_font)
    centered_letter_chip(draw, (B_X, 1180), "B", letter_font)
    centered_letter_chip(draw, (20, C_Y), "C", letter_font)
    centered_letter_chip(draw, (20, D_Y), "D", letter_font)

    dimension_font = load_font(26, bold=True)
    chip(draw, (18, 913), "M5 = C→D  (hold ruler UPRIGHT/vertical)", dimension_font)
    chip(draw, (18, 1064), "M4 = A→B  (hold ruler FLAT/horizontal)", dimension_font)
    add_instruction_panel(upright)

    # Return to the source's raw 1200 x 900 pixel orientation, then preserve its
    # EXIF Orientation=6 so normal viewers display the upright annotated image.
    output_raw = upright.convert("RGB").transpose(Image.Transpose.ROTATE_90)
    if output_raw.size != source_size:
        raise ValueError(f"Output size {output_raw.size} does not match {source_size}")
    source_exif[274] = 6
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "format": "JPEG",
        "quality": 95,
        "subsampling": 0,
        "exif": source_exif.tobytes(),
        "dpi": raw.info.get("dpi", (72, 72)),
    }
    if source_icc:
        save_kwargs["icc_profile"] = source_icc
    output_raw.save(OUTPUT, **save_kwargs)

    print(f"Saved {OUTPUT} at raw resolution {output_raw.size}; upright canvas {upright.size}")
    print(f"Line A: x={A_X}, y={witness_top}..{witness_bottom}")
    print(f"Line B: x={B_X}, y={witness_top}..{witness_bottom}")
    print(f"Line C: y={C_Y}, x=0..{horizontal_end}")
    print(f"Line D: y={D_Y}, x=0..{horizontal_end}")


if __name__ == "__main__":
    main()
