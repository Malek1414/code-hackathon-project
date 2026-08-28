"""Turn court_points.json (from out/court_click.html) into court_calib_<clip>.json files.

    .venv/bin/python vision/court/from_points.py ~/Downloads/court_points.json [--no-run]

For every clip and frame in the file the homography is solved, the reprojection
error and an OK/PRUEFEN verdict are printed, court_calib_<clip>.json plus the
contract copy out/court_calib.json are written, and (unless --no-run) propagate
and minimap are run for dev60. dev60 is game10 from frame 3000 on, so points
clicked on one clip are reused for the other with the frame offset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision.court.calibrate import Clicks, Clip, geometry_warnings, solve, write_outputs  # noqa: E402
from vision.court.geometry import FIBA  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OFFSET = {("dev60", "game10"): 3000, ("game10", "dev60"): -3000}  # dev60 = game10[3000:]
PY = ROOT / ".venv" / "bin" / "python"


def collect(data: dict) -> dict[str, dict[int, Clicks]]:
    clips: dict[str, dict[int, Clicks]] = {}
    for clip, frames in data.get("clips", {}).items():
        for fr, pts in frames.items():
            clicks = {k: (float(v[0]), float(v[1])) for k, v in pts.items()}
            if len(clicks) >= 4:
                clips.setdefault(clip, {})[int(fr)] = clicks
    # share between the two cuts of the same footage
    for (src, dst), off in OFFSET.items():
        for fr, clicks in clips.get(src, {}).items():
            tgt = fr + off
            if tgt >= 0 and tgt not in clips.setdefault(dst, {}):
                clips[dst][tgt] = dict(clicks)
    return clips


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("points", type=Path, nargs="?", default=Path.home() / "Downloads" / "court_points.json")
    ap.add_argument("--no-run", action="store_true", help="nur Kalibrierdateien schreiben, keine Pipeline")
    args = ap.parse_args(argv)
    data = json.loads(args.points.read_text())
    clips = collect(data)
    if not clips:
        raise SystemExit("keine Frames mit mindestens 4 Punkten in der Datei.")
    ok_all = True
    for clip_name, keyframes in clips.items():
        clip_path = ROOT / "data" / "clips" / f"{clip_name}.mp4"
        if not clip_path.exists():
            print(f"{clip_name}: Clip fehlt, übersprungen")
            continue
        clip = Clip(clip_path)
        fits = {}
        for fr, clicks in sorted(keyframes.items()):
            if fr >= clip.count:
                continue
            try:
                fit = solve(FIBA, clicks)
            except ValueError as exc:
                print(f"{clip_name} Frame {fr}: {exc}")
                ok_all = False
                continue
            fit.warnings = geometry_warnings(FIBA, clicks) + fit.warnings
            verdict = "OK" if fit.mean_error_px < 6 and not fit.warnings else "PRUEFEN"
            print(f"{clip_name} Frame {fr}: {verdict}, Fehler {fit.mean_error_px:.1f} px / {fit.mean_error_m:.2f} m, "
                  f"{fit.inliers}/{fit.total} Punkte")
            for w in fit.warnings:
                print("   Hinweis:", w)
            if verdict != "OK":
                ok_all = False
            fits[fr] = fit
        if fits:
            write_outputs(ROOT / "out" / f"court_calib_{clip_name}.json", FIBA, {f: keyframes[f] for f in fits}, fits, clip)
    if args.no_run or "dev60" not in clips:
        return 0 if ok_all else 1
    print("\n== propagate dev60 ==")
    subprocess.run([str(PY), str(ROOT / "vision/court/propagate.py"), "data/clips/dev60.mp4", "--preview", "--preview-every", "10"], cwd=ROOT, check=False)
    print("\n== minimap dev60 ==")
    subprocess.run([str(PY), str(ROOT / "vision/court/minimap.py"), "--clip", "data/clips/dev60.mp4", "--preview", "out/minimap_preview.png"], cwd=ROOT, check=False)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
