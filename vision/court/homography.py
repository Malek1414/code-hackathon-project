"""Image pixels <-> court metres.

Copied from ~/Desktop/APP/courtside/engine/court/homography.py (Courtside,
Sami Magdouli): Correspondence, HomographyFit, solve_homography and _apply.
The camera-propagation half (track_camera, measure_drift) stayed behind; the
hackathon rig is a fixed tripod, one H per clip is the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_CORRESPONDENCES = 4
"""Eight degrees of freedom, four point pairs is the minimum. Four exactly means
no redundancy and therefore no error estimate."""

RANSAC_PIXELS = 6.0
"""How far a clicked point may sit from the fitted mapping and still count.
Generous on purpose: a human clicking a line crossing is routinely three or
four pixels out."""


@dataclass
class Correspondence:
    """One named point, in both worlds."""

    name: str
    image_x: float
    image_y: float
    world_x: float
    world_y: float


@dataclass
class HomographyFit:
    """A solved image -> world mapping, with the evidence for how good it is."""

    matrix: np.ndarray  # 3x3, image pixels -> court metres
    inliers: int
    total: int
    reprojection_errors_m: np.ndarray
    reprojection_errors_px: np.ndarray = field(default_factory=lambda: np.zeros(0))
    warnings: list[str] = field(default_factory=list)

    @property
    def mean_error_m(self) -> float:
        return float(np.mean(self.reprojection_errors_m)) if len(self.reprojection_errors_m) else 0.0

    @property
    def mean_error_px(self) -> float:
        return float(np.mean(self.reprojection_errors_px)) if len(self.reprojection_errors_px) else 0.0

    @property
    def max_error_m(self) -> float:
        return float(np.max(self.reprojection_errors_m)) if len(self.reprojection_errors_m) else 0.0

    @property
    def usable(self) -> bool:
        """Half a metre is about the width of a player."""
        return self.mean_error_m <= 0.5 and self.inliers >= MIN_CORRESPONDENCES

    def to_image(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    def project(self, points) -> np.ndarray:
        """Image pixels -> court metres. Accepts [N, 2], returns [N, 2]."""
        return apply_h(self.matrix, points)

    def unproject(self, points) -> np.ndarray:
        """Court metres -> image pixels."""
        return apply_h(self.to_image(), points)


def apply_h(matrix: np.ndarray, points) -> np.ndarray:
    """Apply a 3x3 homography to [N, 2] points. Points at the horizon become NaN."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    projected = homogeneous @ np.asarray(matrix, np.float64).T
    w = projected[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(w) > 1e-9, projected[:, :2] / w, np.nan)


def solve_homography(correspondences: list[Correspondence]) -> HomographyFit:
    """Fit image -> world from named point pairs."""
    import cv2

    if len(correspondences) < MIN_CORRESPONDENCES:
        raise ValueError(f"{len(correspondences)} Punkte reichen nicht, mindestens {MIN_CORRESPONDENCES} nötig.")

    image = np.array([[c.image_x, c.image_y] for c in correspondences], np.float64)
    world = np.array([[c.world_x, c.world_y] for c in correspondences], np.float64)

    # Fitted world -> image and then inverted: the click error is a constant
    # number of pixels anywhere in the frame, so RANSAC has to judge in pixels.
    matrix_to_image, mask = cv2.findHomography(world, image, cv2.RANSAC, ransacReprojThreshold=RANSAC_PIXELS)
    if matrix_to_image is None:
        raise ValueError("Keine Abbildung gefunden. Meist liegen die Punkte auf einer Geraden oder eine Zuordnung ist vertauscht.")

    matrix = np.linalg.inv(matrix_to_image)
    errors = np.linalg.norm(apply_h(matrix, image) - world, axis=1)
    pixel_errors = np.linalg.norm(apply_h(matrix_to_image, world) - image, axis=1)

    warnings: list[str] = []
    if len(correspondences) == MIN_CORRESPONDENCES:
        warnings.append("Genau vier Punkte: Fehler ist zwangsläufig null und sagt nichts aus. Mehr Punkte setzen.")

    inliers = int(mask.sum()) if mask is not None else len(correspondences)
    if inliers < len(correspondences):
        rejected = [c.name for c, keep in zip(correspondences, mask.ravel(), strict=True) if not keep]
        warnings.append(f"Verworfen, passt nicht zu den anderen: {', '.join(rejected)}")

    fit = HomographyFit(matrix=matrix, inliers=inliers, total=len(correspondences),
                        reprojection_errors_m=errors, reprojection_errors_px=pixel_errors, warnings=warnings)
    if not fit.usable and len(correspondences) > MIN_CORRESPONDENCES:
        warnings.append(f"Mittlerer Fehler {fit.mean_error_m:.2f} m. Über 0,5 m sind Distanzen nicht belastbar.")
    return fit
