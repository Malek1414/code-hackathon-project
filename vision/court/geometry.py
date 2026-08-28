"""Basketball court geometry: the single source of truth for line positions.

Copied and reduced from ~/Desktop/APP/courtside/engine/court/geometry.py
(Courtside, Sami Magdouli). The pydantic models became dataclasses because the
followcam .venv has no pydantic; only the basketball court survived the cut.

Coordinate system (same as Courtside):
    origin at the bottom-left corner of the court, metres,
    x runs along the length (baseline to baseline, 0..28),
    y runs across the width (sideline to sideline, 0..15).
Angles are degrees, 0 pointing along +x, counter-clockwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str = "line"


@dataclass
class Circle:
    cx: float
    cy: float
    r: float
    kind: str = "circle"


@dataclass
class Arc:
    cx: float
    cy: float
    r: float
    start_deg: float
    end_deg: float
    kind: str = "arc"


@dataclass
class Spot:
    """A filled marking with no meaningful radius, here the hoop centre."""

    cx: float
    cy: float
    kind: str = "spot"


Shape = Line | Circle | Arc | Spot


def _rect(x1: float, y1: float, x2: float, y2: float) -> list[Shape]:
    return [
        Line(x1, y1, x2, y1),
        Line(x2, y1, x2, y2),
        Line(x2, y2, x1, y2),
        Line(x1, y2, x1, y1),
    ]


@dataclass
class Landmark:
    """A point on the court a person can find and click.

    Every one of these is the crossing of two painted lines, because those are
    the only things anyone can click accurately. The centre of the centre
    circle is deliberately absent: under perspective the middle of the drawn
    ellipse is not the image of the middle of the circle.
    """

    id: str
    label: str
    x: float
    y: float


@dataclass
class SurfaceSpec:
    label: str
    length_m: float
    width_m: float
    shapes: list[Shape] = field(default_factory=list)
    landmarks: list[Landmark] = field(default_factory=list)
    hoops: list[tuple[float, float]] = field(default_factory=list)

    def landmark(self, landmark_id: str) -> Landmark:
        for lm in self.landmarks:
            if lm.id == landmark_id:
                return lm
        raise KeyError(landmark_id)

    def contains(self, x: float, y: float, tolerance_m: float = 1.0) -> bool:
        return (
            -tolerance_m <= x <= self.length_m + tolerance_m
            and -tolerance_m <= y <= self.width_m + tolerance_m
        )


def basketball(length: float = 28.0, width: float = 15.0, label: str = "FIBA (28 x 15 m)") -> SurfaceSpec:
    """FIBA-proportioned full court."""
    mid_y = width / 2
    three_r = 6.75
    corner_offset = 0.9  # straight section of the three-point line, from the sideline
    hoop_x = 1.575  # centre of the ring, from the baseline
    paint_depth = 5.8
    paint_width = 4.9
    ft_circle_r = 1.8

    shapes: list[Shape] = [*_rect(0, 0, length, width)]
    shapes += [
        Line(length / 2, 0, length / 2, width),
        Circle(length / 2, mid_y, ft_circle_r),
    ]

    hoops: list[tuple[float, float]] = []
    for side in (0, 1):
        base_x = 0.0 if side == 0 else length
        sign = 1 if side == 0 else -1
        hx = base_x + sign * hoop_x
        hoops.append((hx, mid_y))

        shapes += _rect(base_x, mid_y - paint_width / 2, base_x + sign * paint_depth, mid_y + paint_width / 2)
        shapes.append(Circle(base_x + sign * paint_depth, mid_y, ft_circle_r))
        shapes.append(Spot(hx, mid_y))

        # Three-point line: two straights along the sidelines joined by an arc
        # around the hoop. The straight ends where it meets the arc.
        corner_y_low = corner_offset
        corner_y_high = width - corner_offset
        dy = mid_y - corner_y_low
        dx = math.sqrt(three_r**2 - dy**2)
        straight_end_x = hx + sign * dx
        shapes += [
            Line(base_x, corner_y_low, straight_end_x, corner_y_low),
            Line(base_x, corner_y_high, straight_end_x, corner_y_high),
        ]
        half = math.degrees(math.atan2(dy, dx))
        centre_angle = 0.0 if side == 0 else 180.0
        shapes.append(Arc(hx, mid_y, three_r, centre_angle - half, centre_angle + half))

    landmarks: list[Landmark] = [
        Landmark("corner_bl", "Ecke links unten", 0.0, 0.0),
        Landmark("corner_br", "Ecke rechts unten", length, 0.0),
        Landmark("corner_tr", "Ecke rechts oben", length, width),
        Landmark("corner_tl", "Ecke links oben", 0.0, width),
        Landmark("halfway_bottom", "Mittellinie unten", length / 2, 0.0),
        Landmark("halfway_top", "Mittellinie oben", length / 2, width),
    ]
    for side in (0, 1):
        base_x = 0.0 if side == 0 else length
        sign = 1 if side == 0 else -1
        seite = "links" if side == 0 else "rechts"
        landmarks += [
            Landmark(f"paint_{side}_front_bottom", f"Zone {seite}: Freiwurflinie unten",
                     base_x + sign * paint_depth, mid_y - paint_width / 2),
            Landmark(f"paint_{side}_front_top", f"Zone {seite}: Freiwurflinie oben",
                     base_x + sign * paint_depth, mid_y + paint_width / 2),
            Landmark(f"paint_{side}_base_bottom", f"Zone {seite}: Grundlinie unten",
                     base_x, mid_y - paint_width / 2),
            Landmark(f"paint_{side}_base_top", f"Zone {seite}: Grundlinie oben",
                     base_x, mid_y + paint_width / 2),
            Landmark(f"three_{side}_bottom", f"Dreier {seite}: Ende unten (Grundlinie)",
                     base_x, corner_offset),
            Landmark(f"three_{side}_top", f"Dreier {seite}: Ende oben (Grundlinie)",
                     base_x, width - corner_offset),
        ]

    return SurfaceSpec(label=label, length_m=length, width_m=width,
                       shapes=shapes, landmarks=landmarks, hoops=hoops)


FIBA = basketball()


def polylines(spec: SurfaceSpec, step_deg: float = 4.0) -> list[np.ndarray]:
    """Every marking as a polyline in metres, [N, 2]. Spots become tiny circles.

    Shared by the calibration overlay, the minimap and the dashboard SVG so
    all three draw exactly the same court.
    """
    out: list[np.ndarray] = []
    for shape in spec.shapes:
        if isinstance(shape, Line):
            out.append(np.array([[shape.x1, shape.y1], [shape.x2, shape.y2]], np.float64))
        elif isinstance(shape, Circle):
            angles = np.radians(np.arange(0.0, 360.0 + step_deg, step_deg))
            out.append(np.stack([shape.cx + shape.r * np.cos(angles),
                                 shape.cy + shape.r * np.sin(angles)], axis=1))
        elif isinstance(shape, Arc):
            angles = np.radians(np.linspace(shape.start_deg, shape.end_deg,
                                            max(2, int(abs(shape.end_deg - shape.start_deg) / step_deg) + 1)))
            out.append(np.stack([shape.cx + shape.r * np.cos(angles),
                                 shape.cy + shape.r * np.sin(angles)], axis=1))
        elif isinstance(shape, Spot):
            angles = np.radians(np.arange(0.0, 360.0 + 30, 30))
            out.append(np.stack([shape.cx + 0.225 * np.cos(angles),
                                 shape.cy + 0.225 * np.sin(angles)], axis=1))
    return out
