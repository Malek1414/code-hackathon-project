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


TEMPLATE_WIDGETS = {"score_bug": [None], "made_flash": [0, 1], "player_card": [0, 1], "team_overview": [None]}

# data-field in the widget -> (live_state field path, display format). {pct} = round(100*fg_pct).
FIELD_MAP = {
    "score_bug": {"team_a_name": ("teams[0].name", "{teams[0].name}"), "team_b_name": ("teams[1].name", "{teams[1].name}"),
                  "score_a": ("teams[0].score", "{teams[0].score}"), "score_b": ("teams[1].score", "{teams[1].score}"),
                  "clock": ("clock", "P{period}  {clock}")},
    "made_flash": {"points": ("last_event.points", "+{last_event.points}"), "who": ("player.number", "#{player.number}"),
                   "team": ("teams[last_event.team].name", "{teams[last_event.team].name}")},
    "player_card": {"number": ("player.number", "#{player.number}"), "team": ("teams[player.team].name", "{teams[player.team].name}"),
                    "pts": ("player.pts", "{player.pts}"), "fg": ("player.fgm/player.fga", "{player.fgm}/{player.fga}"),
                    "fg_pct": ("player.fg_pct", "{pct}%")},
    "team_overview": {"clock": ("clock", "Period {period}, {clock}"),
                      **{f"{k}_{i}": v for i in (0, 1) for k, v in {
                          "name": (f"teams[{i}].name", f"{{teams[{i}].name}}"), "score": (f"teams[{i}].score", f"{{teams[{i}].score}}"),
                          "fg": (f"teams[{i}].fg_pct", "{pct}%"), "fg_sub": (f"teams[{i}].fgm/teams[{i}].fga", f"FG, {{teams[{i}].fgm}} of {{teams[{i}].fga}}"),
                          "possessions": (f"teams[{i}].possessions", f"{{teams[{i}].possessions}}"),
                          "top_scorer": (f"top_scorer[{i}]", "#{top_scorer.number}, {top_scorer.pts} pts")}.items()}},
}


def _rgb(css: str) -> tuple[int, int, int] | None:
    m = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", css or "")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _hex_rgb(h: str) -> tuple[int, int, int] | None:
    h = (h or "").lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)) if len(h) == 6 else None


def color_role(css: str, teams: list[dict]) -> str:
    rgb = _rgb(css)
    if rgb is None:
        return "white"
    for i, t in enumerate(teams[:2]):
        tc = _hex_rgb(t.get("color", ""))
        if tc and max(abs(a - b) for a, b in zip(rgb, tc)) <= 8:
            return f"team{i}"
    alpha = re.search(r"rgba\([^)]*,\s*([\d.]+)\)", css or "")
    if alpha and float(alpha.group(1)) < 0.9:
        return "muted"
    return "white"


# widget data-field -> the field id live.py fills, and a minimum box width so longer values fit
LIVE_FIELDS = {
    "score_bug": {"team_a_name": ("team_a_name", 190), "team_b_name": ("team_b_name", 190), "score_a": ("score_a", 130),
                  "score_b": ("score_b", 130), "clock": ("clock", 200)},
    "made_flash": {},  # one synthetic "label" box, see below
    "player_card": {"number": ("number", 132), "team": ("team_name", 290), "pts": ("pts", 90), "fg": ("fg", 110), "fg_pct": ("fg_pct", 110)},
    "team_overview": {"clock": ("clock", 320), "name_0": ("team_a_name", 240), "name_1": ("team_b_name", 240),
                      "score_0": ("score_a", 130), "score_1": ("score_b", 130), "fg_0": ("fg_pct_a", 110), "fg_1": ("fg_pct_b", 110),
                      "fg_sub_0": ("fg_a", 180), "fg_sub_1": ("fg_b", 180), "possessions_0": ("poss_a", 90), "possessions_1": ("poss_b", 90),
                      "top_scorer_0": ("top_a", 90), "top_scorer_1": ("top_b", 90)},
}
CAP_HEIGHT = 0.72  # Hershey font_px is about the cap height; CSS px is the em size


def _box(bid, f, path, fmt, min_w, teams, extra=None):
    w = max(int(f["w"]), min_w)
    if f["align"] == "right":
        x = f["x"] - w
    elif f["align"] == "center":
        x = f["x"] - w // 2
    else:
        x = f["x"]
    role = color_role(f.get("color", ""), teams).replace("team0", "team_a").replace("team1", "team_b")
    d = {"id": bid, "field": bid, "x": int(x), "y": int(f["y"]), "w": int(w), "h": int(f["h"]),
         "font_px": int(round(f["size"] * CAP_HEIGHT)), "css_px": int(f["size"]), "weight": "bold" if f["weight"] >= 600 else "regular",
         "align": f["align"], "color_role": role, "baseline": int(f["baseline"]), "source": path, "format": fmt, "example": f.get("example", "")}
    if extra:
        d.update(extra)
    return d


def boxes_for(wid: str, fields: dict, teams: list[dict]) -> list[dict]:
    out = []
    if wid == "made_flash":
        pts = fields.get("points", {"x": 723, "y": 82, "w": 73, "h": 52, "size": 44, "weight": 900, "color": "rgb(255,255,255)", "align": "left"})
        f = dict(pts, w=476, h=52, y=82, size=44, weight=900, align="left", color="rgb(255, 255, 255)")
        out.append(_box("label", f, "last_event", "BASKET  {teams[last_event.team].name} +{last_event.points}", 476, teams,
                        {"note": "one line across the panel, the panel tint is the scoring team's colour (template = team A, template_team1 = team B)"}))
        return out
    for name, f in fields.items():
        if name not in LIVE_FIELDS.get(wid, {}):
            continue
        bid, min_w = LIVE_FIELDS[wid][name]
        path, fmt = FIELD_MAP.get(wid, {}).get(name, (name, "{" + name + "}"))
        out.append(_box(bid, f, path, fmt, min_w, teams))
        if bid in ("top_a", "top_b"):  # points of the top scorer right after the number
            g = dict(f, x=f["x"] + 96, w=120)
            out.append(_box("top_pts_" + bid[-1], g, path + ".pts", "{top_scorer.pts} pts", 120, teams))
    return out


def dump_layout(chrome: str, html_path: Path, profile: Path) -> dict:
    """Run the page once with --dump-dom and read the <pre id="bbb-layout"> the widget writes."""
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
           f"--user-data-dir={profile}", "--window-size=1920,1080", "--force-device-scale-factor=1",
           "--virtual-time-budget=1500", "--dump-dom", html_path.as_uri()]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out, deadline = "", time.time() + 45
    while time.time() < deadline:  # Chrome prints the DOM and then sometimes never exits
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
            continue
        out += line
        if "</html>" in line:
            break
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    m = re.search(r'<pre id="bbb-layout"[^>]*>(.*?)</pre>', out, re.S)
    if not m:
        raise RuntimeError("widget wrote no layout block")
    import html as _html
    return json.loads(_html.unescape(m.group(1)))


def render_templates(args) -> int:
    """Chrome-only templates for the live compositor: same panels, text areas hidden,
    plus layout.json (canvas pixels, y = top of the text box, baseline given too) so
    live.py can draw the numbers itself every frame."""
    chrome = find_chrome(args.chrome)
    if not chrome:
        print("no Chromium found", file=sys.stderr)
        return 2
    base = json.loads((WIDGETS / "state_example.json").read_text())
    if args.state and args.state.exists():
        live = json.loads(args.state.read_text())
        base["teams"] = live.get("teams") or base["teams"]  # names and colours from the live config
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    layout: dict = {"_meta": {"canvas": [1920, 1080], "state_schema": "out/live_state.json", "templates": {},
                              "note": "boxes in canvas pixels, x/y top-left, text vertically centred in h, align inside the box; "
                                      "font_px is about the cap height (css_px is the CSS em size, cap height 0.72 of it); "
                                      "color_role team_a/team_b = teams[i].color, muted = white at 66 percent; "
                                      "templates: team_a is the chrome with team A's accent, team_b the same panel in team B's colour"}}
    with tempfile.TemporaryDirectory(prefix="bbb-tpl-") as tmp:
        tmp = Path(tmp)
        profile = tmp / "profile"
        for wid, variants in TEMPLATE_WIDGETS.items():
            src = (WIDGETS / f"{wid}.html").read_text()
            files = []
            for team in variants:
                state = json.loads(json.dumps(base))
                state["template"] = True
                if team is not None:
                    state["last_event"] = {"type": "made", "team": team, "player_key": None, "points": 2}
                html_tmp = tmp / f"{wid}_{team}.html"
                html_tmp.write_text(inject(src, state))
                name = f"{wid}_template{'' if team in (None, 0) else '_b'}.png"
                png = out_dir / name
                render(chrome, html_tmp, png, profile)
                files.append(str(png.relative_to(ROOT)))
                print(f"{wid}{'' if team is None else ' team ' + 'AB'[team]}: {png.relative_to(ROOT)} ({alpha_report(png)})")
                if team in (None, 0):
                    fields = dump_layout(chrome, html_tmp, profile)
            layout[wid] = boxes_for(wid, fields, base["teams"])
            layout["_meta"]["templates"][wid] = {"team_a": files[0], **({"team_b": files[1]} if len(files) > 1 else {})}
    (WIDGETS / "layout.json").write_text(json.dumps(layout, indent=1))
    print(f"layout: {(WIDGETS / 'layout.json').relative_to(ROOT)} ({sum(len(v) for k, v in layout.items() if isinstance(v, list))} boxes)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=Path, default=ROOT / "out" / "live_state.json")
    ap.add_argument("--only", default="", help="comma separated widget ids")
    ap.add_argument("--chrome", default=os.environ.get("BBB_CHROME"))
    ap.add_argument("--out", type=Path, default=ROOT / "broadcast" / "assets")
    ap.add_argument("--templates", action="store_true",
                    help="render score_bug, made_flash, player_card, team_overview as chrome-only templates "
                         "(text hidden) into <out>/templates/ and write broadcast/widgets/layout.json with the text fields")
    args = ap.parse_args(argv)
    if args.templates:
        return render_templates(args)

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
