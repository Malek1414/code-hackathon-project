"""Render the Big Ball Baller widgets to PNG with alpha.

    .venv/bin/python broadcast/render_widgets.py [--state out/live_state.json] [--only score_bug,player_card]
        [--chrome "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"] [--out broadcast/assets]

Each broadcast/widgets/<id>.html is self-contained and carries example values in
<script id="bbb-state" type="application/json">. When --state (default
out/live_state.json) exists, its JSON replaces that block in a temporary copy, so
the PNGs show the live numbers. Rendering uses headless Chrome with a fully
transparent default background on a 1920x1080 window; the PNG keeps the alpha
channel (verified with Pillow after each render). Without Chrome the script
exits with code 2 and prints where to get one; the HTML sources still work in
any browser (open them with #<urlencoded json> in the hash to inject a state).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDGETS = ROOT / "broadcast" / "widgets"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome", "chromium", "chromium-browser",
]
STATE_RE = re.compile(r'(<script id="bbb-state" type="application/json">)(.*?)(</script>)', re.S)


def find_chrome(explicit: str | None) -> str | None:
    for c in ([explicit] if explicit else []) + CHROME_CANDIDATES:
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    return None


def inject(html: str, state: dict | None) -> str:
    if state is None:
        return html
    return STATE_RE.sub(lambda m: m.group(1) + json.dumps(state) + m.group(3), html, count=1)


def render(chrome: str, html_path: Path, out_png: Path, profile: Path, width=1920, height=1080) -> None:
    cmd = [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
           f"--user-data-dir={profile}", f"--window-size={width},{height}", "--default-background-color=00000000",
           "--force-device-scale-factor=1", "--virtual-time-budget=1500", f"--screenshot={out_png}", html_path.as_uri()]
    out_png.unlink(missing_ok=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # Chrome writes the PNG and then sometimes never exits; wait for a stable file, then stop it.
    deadline = time.time() + 45
    last = -1
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        if out_png.exists():
            size = out_png.stat().st_size
            if size > 0 and size == last:
                break
            last = size
        time.sleep(0.5)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if not out_png.exists() or out_png.stat().st_size == 0:
        err = proc.stderr.read().decode(errors="ignore")[-400:] if proc.stderr else ""
        raise RuntimeError(f"no screenshot written: {err}")


def alpha_report(png: Path) -> str:
    try:
        from PIL import Image
    except ImportError:
        return "alpha unchecked (no Pillow)"
    im = Image.open(png)
    if im.mode != "RGBA":
        return f"NO ALPHA (mode {im.mode})"
    a = im.getchannel("A")
    hist = a.histogram()
    transparent = hist[0]
    total = im.width * im.height
    return f"alpha ok, {100 * transparent / total:.0f}% transparent, {im.width}x{im.height}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=Path, default=ROOT / "out" / "live_state.json")
    ap.add_argument("--only", default="", help="comma separated widget ids")
    ap.add_argument("--chrome", default=os.environ.get("BBB_CHROME"))
    ap.add_argument("--out", type=Path, default=ROOT / "broadcast" / "assets")
    args = ap.parse_args(argv)

    chrome = find_chrome(args.chrome)
    if not chrome:
        print("no Chromium found; install Google Chrome or pass --chrome <binary>. The HTML widgets render in any browser.", file=sys.stderr)
        return 2
    state = None
    if args.state and args.state.exists():
        state = json.loads(args.state.read_text())
        print(f"state: {args.state}")
    else:
        print("state: example values embedded in the widgets")
    ids = [w.strip() for w in args.only.split(",") if w.strip()] or sorted(p.stem for p in WIDGETS.glob("*.html"))
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bbb-") as tmp:
        tmp = Path(tmp)
        profile = tmp / "profile"
        for wid in ids:
            src = WIDGETS / f"{wid}.html"
            if not src.exists():
                print(f"{wid}: no such widget", file=sys.stderr)
                continue
            wstate = state
            court_json = ROOT / "broadcast" / "end_summary.json"
            if wstate is None and wid in ("end_summary", "heat_map_frame") and court_json.exists():
                wstate = json.loads(court_json.read_text())
                print(f"{wid}: state from {court_json.relative_to(ROOT)}")
            html_tmp = tmp / f"{wid}.html"
            html_tmp.write_text(inject(src.read_text(), wstate))
            out_png = args.out / f"{wid}.png"
            try:
                render(chrome, html_tmp, out_png, profile)
            except (RuntimeError, OSError) as exc:
                print(f"{wid}: chrome failed: {exc}", file=sys.stderr)
                continue
            shown = out_png.relative_to(ROOT) if out_png.is_relative_to(ROOT) else out_png
            print(f"{wid}: {shown} ({alpha_report(out_png)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
