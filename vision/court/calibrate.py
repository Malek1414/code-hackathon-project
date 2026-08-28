"""Click court landmarks on keyframes, solve one homography per keyframe, write out/court_calib.json.

Usage (Sami clicks in this terminal's OpenCV window):
    .venv/bin/python vision/court/calibrate.py data/clips/dev60.mp4

Tasten:  Linksklick = Pixel für den markierten Punkt   n = überspringen   u = zurück
         Enter = lösen + Overlay + speichern            j/k = Punkt wechseln
         a/d = 1 s zurück/vor   A/D = 10 s   ,/. = 1 Frame   q/Esc = Ende
Jedes Bild, auf dem geklickt wird, ist ein eigener Keyframe (Kamera schwenkt).
Danach: vision/court/propagate.py rechnet die Frames dazwischen.

Headless (re-solve saved clicks, tests):
    .venv/bin/python vision/court/calibrate.py CLIP --no-gui
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision.court.geometry import FIBA, SurfaceSpec, polylines  # noqa: E402
from vision.court.homography import Correspondence, HomographyFit, apply_h, solve_homography  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

GREEN = (80, 220, 80)
YELLOW = (0, 220, 255)
RED = (60, 60, 240)
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
DARK = (28, 28, 28)

Clicks = dict[str, tuple[float, float]]


# --- solving + file format ---------------------------------------------------


def solve(spec: SurfaceSpec, clicks: Clicks) -> HomographyFit:
    corr = [Correspondence(lm.id, *clicks[lm.id], lm.x, lm.y) for lm in spec.landmarks if lm.id in clicks]
    return solve_homography(corr)


def keyframe_dict(spec: SurfaceSpec, clicks: Clicks, fit: HomographyFit) -> dict:
    points = [{"id": lm.id, "px": [round(float(clicks[lm.id][0]), 1), round(float(clicks[lm.id][1]), 1)],
               "m": [lm.x, lm.y]} for lm in spec.landmarks if lm.id in clicks]
    return {
        "points": points,
        "H_px_to_m": fit.matrix.tolist(),
        "H_m_to_px": fit.to_image().tolist(),
        "reproj_err_px": round(fit.mean_error_px, 2),
        "reproj_err_m": round(fit.mean_error_m, 3),
        "max_err_m": round(fit.max_error_m, 3),
        "inliers": fit.inliers,
        "total": fit.total,
        "warnings": fit.warnings,
    }


def geometry_warnings(spec: SurfaceSpec, clicks: Clicks) -> list[str]:
    """Why a fit that solved may still be worthless: the points span no area."""
    ids = [lm.id for lm in spec.landmarks if lm.id in clicks]
    if len(ids) < 4:
        return []
    px = np.array([clicks[i] for i in ids], np.float64)
    world = np.array([[spec.landmark(i).x, spec.landmark(i).y] for i in ids], np.float64)
    out = []
    sv = np.linalg.svd(px - px.mean(axis=0), compute_uv=False)
    if sv[0] > 0 and sv[1] / sv[0] < 0.08:
        out.append("Punkte liegen im Bild fast auf einer Linie. Punkte in einer anderen Tiefe setzen (Freiwurflinie, Mittellinie).")
    if np.ptp(world[:, 0]) < 1.0:
        out.append("Alle Punkte auf der Grundlinie, die Tiefe ist unbestimmt. Freiwurflinie oder Mittellinie dazunehmen.")
    if np.ptp(world[:, 1]) < 3.0:
        out.append("Alle Punkte in einem schmalen Streifen der Feldbreite. Punkte weiter auseinander setzen.")
    if len(ids) < 6:
        out.append(f"Nur {len(ids)} Punkte. Ab 6 wird der Fehler aussagekräftig.")
    return out


def calib_dict(spec: SurfaceSpec, keyframes: dict[int, Clicks], fits: dict[int, HomographyFit],
               clip: str, fps: float, image_size: tuple[int, int]) -> dict:
    """Contract format from docs/ORCHESTRATION.md: single H for the first keyframe
    at top level, every hand-clicked keyframe under "frames"."""
    solved = sorted(f for f in keyframes if f in fits)
    first = solved[0]
    top = keyframe_dict(spec, keyframes[first], fits[first])
    return {
        "clip": clip,
        "frame": first,
        "fps": fps,
        "image_size": list(image_size),
        "court_m": {"length": spec.length_m, "width": spec.width_m},
        **top,
        "frames": {str(f): keyframe_dict(spec, keyframes[f], fits[f]) for f in solved},
    }


def load_keyframes(path: Path) -> dict[int, Clicks]:
    data = json.loads(path.read_text())
    out: dict[int, Clicks] = {}
    frames = data.get("frames") or {str(data.get("frame", 0)): data}
    for key, kf in frames.items():
        out[int(key)] = {p["id"]: (float(p["px"][0]), float(p["px"][1])) for p in kf.get("points", [])}
    return out


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# --- drawing -----------------------------------------------------------------


def draw_court(frame: np.ndarray, H_m_to_px: np.ndarray, spec: SurfaceSpec,
               color=YELLOW, thickness: int = 2) -> None:
    """Reprojected court markings onto the frame, in place. Segments that land
    absurdly far outside the image (behind the camera, near the horizon) are dropped."""
    h, w = frame.shape[:2]
    limit = 4 * max(h, w)
    for poly in polylines(spec):
        px = apply_h(H_m_to_px, poly)
        for (x1, y1), (x2, y2) in zip(px[:-1], px[1:]):
            if not np.all(np.isfinite([x1, y1, x2, y2])):
                continue
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) > limit:
                continue
            cv2.line(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                     color, thickness, cv2.LINE_AA)


def overlay_image(frame: np.ndarray, spec: SurfaceSpec, clicks: Clicks, fit: HomographyFit | None) -> np.ndarray:
    out = frame.copy()
    if fit is not None:
        draw_court(out, fit.to_image(), spec)
        for lm in spec.landmarks:  # where the fit thinks each clicked landmark is
            if lm.id in clicks:
                rx, ry = apply_h(fit.to_image(), [[lm.x, lm.y]])[0]
                if np.isfinite(rx) and np.isfinite(ry):
                    cv2.drawMarker(out, (int(rx), int(ry)), RED, cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
    for lm in spec.landmarks:
        if lm.id in clicks:
            x, y = clicks[lm.id]
            cv2.circle(out, (int(x), int(y)), 6, GREEN, 2, cv2.LINE_AA)
            cv2.putText(out, lm.id, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    return out


# --- video access ------------------------------------------------------------


class Clip:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"Clip nicht lesbar: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cache: dict[int, np.ndarray] = {}
        self.last_read = 0

    def frame(self, index: int) -> np.ndarray:
        """Frame by index. CAP_PROP_FRAME_COUNT overstates the length (dev60 reports
        3121, 3001 are readable), so an unreadable index shrinks `count` and the
        last readable frame is returned instead of crashing the click session."""
        index = max(0, min(index, max(self.count - 1, 0)))
        while index not in self._cache:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, img = self.cap.read()
            if ok:
                if len(self._cache) > 12:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[index] = img
                break
            if index == 0:
                raise SystemExit(f"Frame 0 aus {self.path} nicht lesbar.")
            self.count = index
            index = max(0, index - 25)
        self.last_read = index
        return self._cache[index]


# --- interactive tool --------------------------------------------------------


class Calibrator:
    PANEL_W = 470

    def __init__(self, clip: Clip, spec: SurfaceSpec, keyframes: dict[int, Clicks], start_frame: int,
                 display_width: int):
        self.clip = clip
        self.spec = spec
        self.keyframes = {f: dict(c) for f, c in keyframes.items()}
        self.fits: dict[int, HomographyFit] = {}
        self.history: list[tuple[int, str]] = []
        self.frame_index = start_frame
        self.current = 0
        self.message = "Klick auf den markierten Punkt. Enter = lösen und speichern."
        self.cursor: tuple[int, int] | None = None
        self.scale = min(1.0, display_width / clip.width)
        self.disp_w, self.disp_h = int(clip.width * self.scale), int(clip.height * self.scale)
        for f, clicks in self.keyframes.items():
            if len(clicks) >= 4:
                try:
                    self.fits[f] = solve(spec, clicks)
                except ValueError:
                    pass
        self._advance_to_unset(0)

    @property
    def clicks(self) -> Clicks:
        return self.keyframes.setdefault(self.frame_index, {})

    @property
    def fit(self) -> HomographyFit | None:
        return self.fits.get(self.frame_index)

    @property
    def frame(self) -> np.ndarray:
        return self.clip.frame(self.frame_index)

    # navigation
    def _advance_to_unset(self, start: int) -> None:
        n = len(self.spec.landmarks)
        for k in range(n):
            i = (start + k) % n
            if self.spec.landmarks[i].id not in self.clicks:
                self.current = i
                return
        self.current = start % n

    def seek(self, delta_frames: int) -> None:
        if not self.keyframes.get(self.frame_index):
            self.keyframes.pop(self.frame_index, None)
        self.frame_index = max(0, min(self.frame_index + delta_frames, self.clip.count - 1))
        self.clip.frame(self.frame_index)
        if self.clip.last_read != self.frame_index:  # ran into the real end of the clip
            self.frame_index = self.clip.last_read
        t = self.frame_index / self.clip.fps
        state = "Keyframe" if self.keyframes.get(self.frame_index) else "neuer Keyframe, sobald geklickt wird"
        self.message = f"Frame {self.frame_index} ({t:.1f} s): {state}."
        self._advance_to_unset(0)

    def on_mouse(self, event, x, y, flags, param) -> None:
        if x >= self.disp_w or y >= self.disp_h:
            self.cursor = None
            return
        fx, fy = x / self.scale, y / self.scale
        self.cursor = (int(fx), int(fy))
        if event == cv2.EVENT_LBUTTONDOWN:
            lm = self.spec.landmarks[self.current]
            self.clicks[lm.id] = (fx, fy)
            self.history = [h for h in self.history if h != (self.frame_index, lm.id)]
            self.history.append((self.frame_index, lm.id))
            self.message = f"{lm.id} gesetzt ({int(fx)}, {int(fy)})"
            self._advance_to_unset(self.current + 1)

    def skip(self) -> None:
        self.current = (self.current + 1) % len(self.spec.landmarks)

    def back(self) -> None:
        self.current = (self.current - 1) % len(self.spec.landmarks)

    def undo(self) -> None:
        if not self.history:
            self.message = "Nichts zum Zurücknehmen."
            return
        frame, last = self.history.pop()
        self.frame_index = frame
        self.clicks.pop(last, None)
        self.fits.pop(frame, None)
        self.current = next(i for i, lm in enumerate(self.spec.landmarks) if lm.id == last)
        self.message = f"{last} entfernt."

    def try_solve(self) -> bool:
        try:
            self.fits[self.frame_index] = solve(self.spec, self.clicks)
        except ValueError as exc:
            self.fits.pop(self.frame_index, None)
            self.message = str(exc)
            return False
        fit = self.fits[self.frame_index]
        fit.warnings = geometry_warnings(self.spec, self.clicks) + fit.warnings
        verdict = "OK" if fit.mean_error_px < 6 and not fit.warnings else "PRUEFEN"
        self.message = (f"{verdict}: Fehler {fit.mean_error_px:.1f} px / {fit.mean_error_m:.2f} m "
                        f"({fit.inliers}/{fit.total} Punkte). Gespeichert.")
        return True

    # rendering
    def render(self) -> np.ndarray:
        img = overlay_image(self.frame, self.spec, self.clicks, self.fit)
        disp = cv2.resize(img, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((max(self.disp_h, 760), self.disp_w + self.PANEL_W, 3), DARK, np.uint8)
        canvas[: self.disp_h, : self.disp_w] = disp
        self._panel(canvas)
        return canvas

    def _panel(self, canvas: np.ndarray) -> None:
        x0 = self.disp_w + 16
        y = 28
        t = self.frame_index / self.clip.fps
        solved = sorted(self.fits)
        cv2.putText(canvas, f"Frame {self.frame_index}  ({t:.1f} s)", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        y += 22
        cv2.putText(canvas, "Klick=setzen  n=weiter  j/k=wechseln  u=zurueck", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
        y += 17
        cv2.putText(canvas, "Enter=loesen+speichern  a/d=1s  A/D=10s  ,/.=Frame  q=Ende", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
        y += 17
        kf_text = ", ".join(f"{f} ({f / self.clip.fps:.0f}s)" for f in solved) or "noch keiner"
        for line in _wrap("Keyframes: " + kf_text, 58):
            cv2.putText(canvas, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1, cv2.LINE_AA)
            y += 17
        y += 8
        for i, lm in enumerate(self.spec.landmarks):
            done = lm.id in self.clicks
            color = GREEN if done else GREY
            if i == self.current:
                cv2.rectangle(canvas, (x0 - 6, y - 15), (x0 + self.PANEL_W - 26, y + 6), (60, 60, 60), -1)
                color = YELLOW if not done else GREEN
            mark = "x" if done else " "
            cv2.putText(canvas, f"[{mark}] {lm.label}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
            y += 20
        y += 6
        for line in _wrap(self.message, 54):
            cv2.putText(canvas, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
            y += 18
        if self.fit is not None:
            for wtext in self.fit.warnings[:2]:
                for line in _wrap(wtext, 54):
                    cv2.putText(canvas, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, YELLOW, 1, cv2.LINE_AA)
                    y += 17
        self._magnifier(canvas, x0, canvas.shape[0] - 196)

    def _magnifier(self, canvas: np.ndarray, x0: int, y0: int) -> None:
        """4x crop around the cursor so a line crossing can be hit on a downscaled 1080p frame."""
        size, zoom = 180, 4
        if self.cursor is None:
            return
        cx, cy = self.cursor
        r = size // (2 * zoom)
        frame = self.frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        mag = cv2.resize(crop, ((x2 - x1) * zoom, (y2 - y1) * zoom), interpolation=cv2.INTER_NEAREST)
        mh, mw = mag.shape[:2]
        if y0 + mh > canvas.shape[0] or x0 + mw > canvas.shape[1] or y0 < 0:
            return
        canvas[y0:y0 + mh, x0:x0 + mw] = mag
        ox, oy = x0 + (cx - x1) * zoom, y0 + (cy - y1) * zoom
        cv2.line(canvas, (ox - 12, oy), (ox + 12, oy), RED, 1)
        cv2.line(canvas, (ox, oy - 12), (ox, oy + 12), RED, 1)
        cv2.rectangle(canvas, (x0, y0), (x0 + mw, y0 + mh), GREY, 1)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines


def write_outputs(out: Path, spec: SurfaceSpec, keyframes: dict[int, Clicks], fits: dict[int, HomographyFit],
                  clip: Clip) -> None:
    if not fits:
        print("Kein gelöster Keyframe, nichts geschrieben.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    data = calib_dict(spec, keyframes, fits, rel(clip.path), clip.fps, (clip.width, clip.height))
    out.write_text(json.dumps(data, indent=1))
    contract = out.with_name("court_calib.json")
    if out != contract:
        contract.write_text(json.dumps(data, indent=1))
    for f in sorted(fits):
        preview = out.with_name(f"{out.stem}_preview_{f}.jpg")
        cv2.imwrite(str(preview), overlay_image(clip.frame(f), spec, keyframes[f], fits[f]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        kf = data["frames"][str(f)]
        print(f"Keyframe {f}: {kf['reproj_err_px']} px / {kf['reproj_err_m']} m, {kf['inliers']}/{kf['total']} Punkte  {rel(preview)}")
        for wtext in kf["warnings"]:
            print("  Hinweis:", wtext)
    print(f"gespeichert: {rel(out)}  ({len(fits)} Keyframes)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--frame", type=int, default=0, help="Startframe im Fenster")
    ap.add_argument("--out", type=Path, default=None,
                    help="Standard: out/court_calib_<clip>.json, zusätzlich Kopie nach out/court_calib.json (Vertrag)")
    ap.add_argument("--points", type=Path, default=None,
                    help="vorhandene court_calib.json als Start laden (Standard: --out, falls vorhanden)")
    ap.add_argument("--fresh", action="store_true", help="keine gespeicherten Punkte laden")
    ap.add_argument("--no-gui", action="store_true", help="nur lösen und schreiben, kein Fenster")
    ap.add_argument("--width", type=int, default=1380, help="Anzeigebreite des Frames in Pixel")
    args = ap.parse_args(argv)

    spec = FIBA
    clip = Clip(args.clip)
    if args.out is None:
        args.out = ROOT / "out" / f"court_calib_{args.clip.stem}.json"
        contract = ROOT / "out" / "court_calib.json"
        if not args.out.exists() and contract.exists() and json.loads(contract.read_text()).get("clip") == rel(args.clip):
            args.out.write_text(contract.read_text())  # older run saved only the contract file
    keyframes: dict[int, Clicks] = {}
    source = args.points or (args.out if args.out.exists() and not args.fresh else None)
    if source and source.exists():
        keyframes = load_keyframes(source)
        print(f"{len(keyframes)} Keyframes aus {rel(source)} geladen: {sorted(keyframes)}")

    if args.no_gui:
        fits = {f: solve(spec, c) for f, c in keyframes.items() if len(c) >= 4}
        for f, fit in fits.items():
            fit.warnings = geometry_warnings(spec, keyframes[f]) + fit.warnings
        write_outputs(args.out, spec, keyframes, fits, clip)
        return 0

    start = args.frame if args.frame or not keyframes else sorted(keyframes)[0]
    cal = Calibrator(clip, spec, keyframes, start, args.width)
    win = "Court-Kalibrierung"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, cal.on_mouse)
    print("Fenster offen. Klicken, Enter = lösen + speichern, a/d = 1 s vor/zurück, q = Ende.")
    step_1s = max(1, int(round(clip.fps)))
    keys = {
        ord("n"): cal.skip, ord("j"): cal.skip, ord("k"): cal.back, ord("u"): cal.undo,
        ord("a"): lambda: cal.seek(-step_1s), ord("d"): lambda: cal.seek(step_1s),
        ord("A"): lambda: cal.seek(-10 * step_1s), ord("D"): lambda: cal.seek(10 * step_1s),
        ord(","): lambda: cal.seek(-1), ord("."): lambda: cal.seek(1),
    }
    while True:
        cv2.imshow(win, cal.render())
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in keys:
            keys[key]()
        elif key in (13, 10):
            if cal.try_solve():
                write_outputs(args.out, spec, cal.keyframes, cal.fits, clip)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    if cal.fit is None and len(cal.clicks) >= 4 and cal.try_solve():
        write_outputs(args.out, spec, cal.keyframes, cal.fits, clip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
