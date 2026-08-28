"""Team assignment by jersey color.

Crop = chest area (rows 15-45 % of the box, the middle 40 % of its width:
shirt, not head, shorts or floor). Feature = median LAB of the crop plus the
share of blue, red and dark pixels in it. Shares instead of one median color
because a big white number on a blue jersey pulls the median to neutral.

Two modes:

* `rules` (default, this game): Moabit plays blue, the Wiesel black with red
  panels, referees grey. KMeans k=2 fails here, measured: in LAB black sits
  closer to blue than to red, so k=2 splits light/dark or red/rest instead of
  team/team. The rule is explicit: strongly blue (b* below the threshold) →
  team 0; enough red or dark pixels (black, red) → team 1; the rest (grey,
  white: referee, spectators) → -1.
* `kmeans`: generic path for footage with two clearly different jersey colors.
  k=2 on (L/2, a, b), fitted on crops sampled across the clip, team 0 = the
  bluer centroid, far-from-both = -1.

Per track id the label is a majority vote over the track's last `vote_window`
labels, so a player does not flip team when he crosses a shadow, but a track
that started occluded (someone else's shirt in the crop) recovers within a
second instead of keeping the wrong team for its whole life.
"""

from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np
from sklearn.cluster import KMeans

L_WEIGHT = 0.5
BLUE_B_MAX = 118.0  # OpenCV LAB b channel (b* + 128); blue jerseys measured 99-117
RED_A_MIN = 150.0  # a channel; red panels measured 160+
DARK_L_MAX = 60.0  # black jerseys measured L 12-40, grey referee 107, white 198
BLUE_SHARE, RED_SHARE, DARK_SHARE = 0.15, 0.2, 0.4  # black jerseys measured blue share 0.00-0.01
VOTE_WINDOW = 30  # labels per track kept for the majority vote (~1.2 s at 25 fps)


def torso_color(frame_bgr: np.ndarray, bbox) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = y2 - y1, x2 - x1
    if h < 16 or w < 8:
        return None
    ty1, ty2 = y1 + int(h * 0.15), y1 + int(h * 0.45)
    tx1, tx2 = x1 + int(w * 0.30), x2 - int(w * 0.30)
    ty1, ty2 = max(ty1, 0), min(ty2, frame_bgr.shape[0])
    tx1, tx2 = max(tx1, 0), min(tx2, frame_bgr.shape[1])
    if ty2 - ty1 < 2 or tx2 - tx1 < 2:
        return None
    crop = frame_bgr[ty1:ty2, tx1:tx2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    med = np.median(lab, axis=0)  # raw L, a, b
    blue = float(np.mean(lab[:, 2] < BLUE_B_MAX))
    red = float(np.mean(lab[:, 1] > RED_A_MIN))
    dark = float(np.mean(lab[:, 0] < DARK_L_MAX))
    return np.array([med[0], med[1], med[2], blue, red, dark], dtype=np.float32)


def rule_label(feat: np.ndarray) -> int:
    blue, red, dark = float(feat[3]), float(feat[4]), float(feat[5])
    if blue >= BLUE_SHARE:
        return 0
    if red >= RED_SHARE or dark >= DARK_SHARE:
        return 1
    return -1


class TeamClassifier:
    def __init__(self, mode: str = "rules", *, min_samples: int = 200) -> None:
        self.mode = mode
        self.min_samples = min_samples
        self.samples: list[np.ndarray] = []
        self.kmeans: KMeans | None = None
        self.referee_dist: float = float("inf")
        self.votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=VOTE_WINDOW))
        self.centroids_lab: list[list[float]] = []

    @property
    def fitted(self) -> bool:
        return self.mode == "rules" or self.kmeans is not None

    def fit(self, feats: np.ndarray) -> None:
        """Only the kmeans mode learns anything; rules need no fit."""
        if self.mode == "rules" or len(feats) < 10:
            return
        x = feats[:, :3].copy()
        x[:, 0] *= L_WEIGHT
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(x)
        if km.cluster_centers_[0, 2] > km.cluster_centers_[1, 2]:  # team 0 = bluer
            km.cluster_centers_ = km.cluster_centers_[::-1].copy()
            km.labels_ = 1 - km.labels_
        self.kmeans = km
        d = np.linalg.norm(x - km.cluster_centers_[km.labels_], axis=1)
        self.referee_dist = float(np.percentile(d, 90) * 1.6 + 5.0)
        c = km.cluster_centers_.copy()
        c[:, 0] /= L_WEIGHT
        self.centroids_lab = c.round(1).tolist()

    def observe(self, feat: np.ndarray | None) -> None:
        """Online path (live): collect crops, fit once enough are in."""
        if self.mode != "kmeans" or self.kmeans is not None or feat is None:
            return
        self.samples.append(feat)
        if len(self.samples) >= self.min_samples:
            self.fit(np.array(self.samples, dtype=np.float32))
            self.samples.clear()

    def raw_label(self, feat: np.ndarray | None) -> int:
        if feat is None:
            return -1
        if self.mode == "rules":
            return rule_label(feat)
        if self.kmeans is None:
            self.observe(feat)
            return -1
        x = feat[:3].copy()
        x[0] *= L_WEIGHT
        d = np.linalg.norm(self.kmeans.cluster_centers_ - x, axis=1)
        k = int(np.argmin(d))
        return -1 if d[k] > self.referee_dist else k

    def assign(self, track_id: int, feat: np.ndarray | None) -> int:
        """Vote-smoothed team for this track: 0, 1 or -1."""
        label = self.raw_label(feat)
        hist = self.votes[track_id]
        hist.append(label)
        n0, n1, nu = hist.count(0), hist.count(1), hist.count(-1)
        if n0 == 0 and n1 == 0:
            return -1
        # Unknown wins only if it clearly dominates the recent history.
        if nu > 2 * max(n0, n1) and nu >= 5:
            return -1
        return 0 if n0 >= n1 else 1
