#!/usr/bin/env python3
"""Add measurement callouts to the three tripod reference photos."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "tripod-photos"
OUTPUT_DIR = ROOT / "annotated"

REFERENCE_SIZE = (900, 1200)  # Display orientation after applying EXIF rotation.
ORANGE = (255, 61, 0, 255)  # #FF3D00
CHIP = (27, 27, 27, 217)  # #1B1B1B at about 85% opacity.
WHITE = (255, 255, 255, 255)

FONT_PATHS = (
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError("No supported bold TrueType font was found")


FONT = load_font(30)


def scaled_point(point: tuple[int, int], size: tuple[int, int]) -> tuple[int, int]:
    """Scale a point authored against the 900x1200 display reference."""
    sx = size[0] / REFERENCE_SIZE[0]
    sy = size[1] / REFERENCE_SIZE[1]
    return round(point[0] * sx), round(point[1] * sy)


def draw_arrowhead(
    draw: ImageDraw.ImageDraw,
    tip: tuple[int, int],
    toward_tip: tuple[float, float],
    *,
    length: int = 20,
    width: int = 20,
) -> None:
    """Draw a filled triangular head whose point is exactly at ``tip``."""
    ux, uy = toward_tip
    base_x = tip[0] - ux * length
    base_y = tip[1] - uy * length
    px, py = -uy, ux
    half_width = width / 2
    draw.polygon(
        [
            tip,
            (round(base_x + px * half_width), round(base_y + py * half_width)),
            (round(base_x - px * half_width), round(base_y - py * half_width)),
        ],
        fill=ORANGE,
    )


def draw_arrow(
    layer: Image.Image,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    double: bool = False,
    stroke: int = 7,
    head_length: int = 20,
    head_width: int = 20,
) -> None:
    """Draw a bold single- or double-headed measurement arrow."""
    draw = ImageDraw.Draw(layer)
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        raise ValueError("Arrow endpoints must be different")
    ux, uy = dx / distance, dy / distance
    draw.line([start, end], fill=ORANGE, width=stroke)
    draw_arrowhead(
        draw,
        end,
        (ux, uy),
        length=head_length,
        width=head_width,
    )
    if double:
        draw_arrowhead(
            draw,
            start,
            (-ux, -uy),
            length=head_length,
            width=head_width,
        )


def draw_thin_leader(
    layer: Image.Image,
    points: list[tuple[int, int]],
    *,
    stroke: int = 3,
) -> None:
    """Draw the TIP note's thin leader and a compact arrowhead."""
    draw = ImageDraw.Draw(layer)
    draw.line(points, fill=ORANGE, width=stroke, joint="curve")
    dx = points[-1][0] - points[-2][0]
    dy = points[-1][1] - points[-2][1]
    distance = math.hypot(dx, dy)
    draw_arrowhead(
        draw,
        points[-1],
        (dx / distance, dy / distance),
        length=14,
        width=14,
    )


def draw_chip(
    layer: Image.Image,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont = FONT,
    padding_x: int = 12,
    padding_y: int = 9,
    spacing: int = 3,
) -> tuple[int, int, int, int]:
    """Draw a rounded, translucent dark label chip and return its bounds."""
    draw = ImageDraw.Draw(layer)
    text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    box = (
        xy[0],
        xy[1],
        xy[0] + text_w + 2 * padding_x,
        xy[1] + text_h + 2 * padding_y,
    )
    if box[0] < 0 or box[1] < 0 or box[2] > layer.width or box[3] > layer.height:
        raise ValueError(f"Label chip leaves the {layer.size} frame: {box!r} ({text!r})")
    draw.rounded_rectangle(box, radius=10, fill=CHIP)
    draw.multiline_text(
        (xy[0] + padding_x, xy[1] + padding_y - text_box[1]),
        text,
        font=font,
        fill=WHITE,
        spacing=spacing,
    )
    return box


def annotate_6597(image: Image.Image) -> tuple[Image.Image, str]:
    size = image.size
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    p = lambda point: scaled_point(point, size)

    # M1: center-column tube, above the brace hub.
    m1_tip = p((431, 558))
    draw_arrow(layer, p((493, 666)), m1_tip)

    # M2: exposed column section between the underside of the head and crown.
    m2_top, m2_bottom = p((481, 326)), p((481, 389))
    draw_arrow(layer, m2_top, m2_bottom, double=True)

    # TIP: a deliberately thinner leader to the same center-column tube.
    tip_target = p((434, 492))
    draw_thin_leader(layer, [p((473, 301)), p((452, 344)), tip_target])

    # Chips are rendered last so leader/arrow starts tuck cleanly under them.
    draw_chip(layer, p((20, 666)), "M1: column diameter (paper-strip trick)")
    draw_chip(layer, p((513, 337)), "M2: free column height")
    draw_chip(
        layer,
        p((458, 212)),
        "TIP: raise + lock the\nelevator column ~10cm first",
    )

    result = Image.alpha_composite(image.convert("RGBA"), layer)
    summary = (
        f"IMG_6597: M1 tip={m1_tip}; M2 endpoints={m2_top}->{m2_bottom}; "
        f"TIP target={tip_target}"
    )
    return result, summary


def annotate_6603(image: Image.Image) -> tuple[Image.Image, str]:
    size = image.size
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    p = lambda point: scaled_point(point, size)

    # M3 lands at the middle of the exposed chrome shaft between grip and head.
    m3_tip = p((443, 842))
    draw_arrow(layer, p((395, 718)), m3_tip)

    # M4 projects the shaft midpoint to the tripod/head vertical centerline.
    m4_outer, m4_center = p((443, 907)), p((520, 907))
    draw_arrow(layer, m4_outer, m4_center, double=True)

    # M5 compares chrome-shaft height with the bottom edge of the head casting.
    m5_shaft, m5_casting = p((602, 842)), p((602, 901))
    draw_arrow(layer, m5_shaft, m5_casting, double=True)

    draw_chip(layer, p((20, 666)), "M3: chrome shaft diameter")
    draw_chip(layer, p((577, 934)), "M4: arm reach\n(horizontal)")
    draw_chip(layer, p((592, 703)), "M5: fork drop\n(vertical)")

    result = Image.alpha_composite(image.convert("RGBA"), layer)
    summary = (
        f"IMG_6603: M3 tip={m3_tip}; M4 endpoints={m4_outer}->{m4_center}; "
        f"M5 endpoints={m5_shaft}->{m5_casting}"
    )
    return result, summary


def annotate_6604(image: Image.Image) -> tuple[Image.Image, str]:
    size = image.size
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    p = lambda point: scaled_point(point, size)

    # M6 targets the small opening in the handle's rounded butt face.
    m6_tip = p((282, 966))
    draw_arrow(layer, p((302, 1031)), m6_tip)

    # M3 targets the center of the exposed chrome rod.
    m3_tip = p((554, 651))
    draw_arrow(layer, p((402, 552)), m3_tip)

    # M2 spans the visible free black center-column section below the casting.
    m2_top, m2_bottom = p((476, 773)), p((476, 875))
    draw_arrow(layer, m2_top, m2_bottom, double=True)

    draw_chip(layer, p((20, 1028)), "M6: butt hole - through-hole? diameter?")
    draw_chip(layer, p((20, 503)), "M3: chrome shaft diameter")
    draw_chip(layer, p((535, 803)), "M2: free column height")

    result = Image.alpha_composite(image.convert("RGBA"), layer)
    summary = (
        f"IMG_6604: M6 tip={m6_tip}; M3 tip={m3_tip}; "
        f"M2 endpoints={m2_top}->{m2_bottom}"
    )
    return result, summary


ANNOTATORS = {
    "IMG_6597.jpg": annotate_6597,
    "IMG_6603.jpg": annotate_6603,
    "IMG_6604.jpg": annotate_6604,
}


def restore_storage_orientation(image: Image.Image, orientation: int) -> Image.Image:
    """Invert EXIF display orientation so output storage matches the source JPEG."""
    inverse_transpose = {
        1: None,
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_90,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_270,
    }
    operation = inverse_transpose.get(orientation)
    return image if operation is None else image.transpose(operation)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, annotator in ANNOTATORS.items():
        source_path = SOURCE_DIR / filename
        output_path = OUTPUT_DIR / f"{source_path.stem}_annotated.jpg"
        with Image.open(source_path) as raw:
            raw_size = raw.size
            exif = raw.getexif()
            orientation = exif.get(274, 1)
            exif_bytes = exif.tobytes()
            icc_profile = raw.info.get("icc_profile")
            oriented = ImageOps.exif_transpose(raw).convert("RGB")
        annotated, summary = annotator(oriented)
        stored = restore_storage_orientation(annotated.convert("RGB"), orientation)
        if stored.size != raw_size:
            raise ValueError(
                f"Output storage size {stored.size} does not match source {raw_size}"
            )
        save_options = {
            "quality": 95,
            "subsampling": 0,
            "exif": exif_bytes,
        }
        if icc_profile is not None:
            save_options["icc_profile"] = icc_profile
        stored.save(output_path, **save_options)
        print(summary)


if __name__ == "__main__":
    main()
