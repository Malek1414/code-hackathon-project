"""PIPELINE smoke test: run_all on a 10 s cut of dev60, then check the contract files.

    .venv/bin/python -m vision.smoke_test [--out-dir out/smoke_pipeline] [--max-wait 1800] [--with-numbers] [--in-place]

1. cuts data/clips/smoke10.mp4 (48 s to 58 s of dev60, contains its one shot) with the
   imageio_ffmpeg binary and -c copy, once;
2. waits while another model job holds the GPU (ps -axo command, see run_all.GPU_JOB_PATTERNS);
3. runs vision.run_all into --out-dir (default out/smoke_pipeline, so the team's live
   artifacts in out/ stay untouched; --in-place writes to out/ as docs/tasks/PIPELINE.md
   literally asks, which overwrites tracks.jsonl, events.json, stats.json, dashboard.html);
4. asserts tracks.jsonl, events.json, stats.json, dashboard.html exist and parse.

Exit 0 = PASS, 1 = FAIL, 2 = GPU still busy after --max-wait. Log: out/smoke_test.log.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.run_all import PY, gpu_jobs, rel, wait_for_gpu  # noqa: E402

SOURCE = ROOT / "data" / "clips" / "dev60.mp4"
CUT = ROOT / "data" / "clips" / "smoke10.mp4"
CUT_START_S, CUT_LEN_S = 48, 10
LOG = ROOT / "out" / "smoke_test.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def make_cut(force: bool = False) -> Path:
    if CUT.exists() and CUT.stat().st_size > 0 and not force:
        return CUT
    if not SOURCE.exists():
        raise SystemExit(f"Quelle fehlt: {rel(SOURCE)}")
    import imageio_ffmpeg  # in .venv

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(CUT_START_S), "-t", str(CUT_LEN_S),
           "-i", str(SOURCE), "-c", "copy", "-movflags", "+faststart", str(CUT)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not CUT.exists():
        raise SystemExit(f"ffmpeg Schnitt fehlgeschlagen: {proc.stderr.strip()[-300:]}")
    log(f"Schnitt {rel(CUT)} ({CUT.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f} s")
    return CUT


def check_contract(od: Path) -> list[str]:
    """Returns a list of failures (empty = PASS)."""
    fails: list[str] = []

    def need(name: str) -> Path | None:
        p = od / name
        if not p.exists() or p.stat().st_size == 0:
            fails.append(f"{rel(p)} fehlt oder leer")
            return None
        return p

    p = need("tracks.jsonl")
    if p:
        n = frames_with_players = 0
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                fails.append(f"tracks.jsonl Zeile {i} kein JSON")
                break
            n += 1
            if not isinstance(row.get("players"), list) or "frame" not in row or "t" not in row:
                fails.append(f"tracks.jsonl Zeile {i} ohne frame/t/players")
                break
            frames_with_players += bool(row["players"])
        if n == 0:
            fails.append("tracks.jsonl hat keine Zeilen")
        elif frames_with_players == 0:
            fails.append("tracks.jsonl: kein Frame mit Spielern")
        else:
            log(f"tracks.jsonl: {n} Zeilen, {frames_with_players} mit Spielern")
    p = need("events.json")
    if p:
        try:
            ev = json.loads(p.read_text())
            if not isinstance(ev.get("shots"), list):
                fails.append("events.json ohne shots-Liste")
            else:
                log(f"events.json: {len(ev['shots'])} Würfe")
        except ValueError:
            fails.append("events.json kein JSON")
    p = need("stats.json")
    if p:
        try:
            st = json.loads(p.read_text())
            if not isinstance(st.get("players"), list) or not isinstance(st.get("teams"), list):
                fails.append("stats.json ohne players/teams")
            else:
                log(f"stats.json: {len(st['players'])} Spieler, {len(st['teams'])} Teams")
        except ValueError:
            fails.append("stats.json kein JSON")
    p = need("dashboard.html")
    if p:
        text = p.read_text(errors="replace")
        if "<html" not in text.lower() or len(text) < 1000:
            fails.append("dashboard.html ist keine vollständige Seite")
        else:
            log(f"dashboard.html: {len(text) / 1e3:.0f} kB")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--out-dir", default="out/smoke_pipeline")
    ap.add_argument("--in-place", action="store_true", help="nach out/ schreiben (überschreibt die Live-Artefakte des Teams)")
    ap.add_argument("--max-wait", type=float, default=1800, help="Sekunden auf freie GPU warten")
    ap.add_argument("--with-numbers", action="store_true", help="NUMBERS (EasyOCR, CPU, langsam) mitlaufen lassen")
    ap.add_argument("--force", action="store_true", help="run_all --force und Schnitt neu")
    ap.add_argument("--no-wait", action="store_true", help="GPU-Prüfung auslassen")
    a = ap.parse_args(argv)
    od = ROOT / "out" if a.in_place else Path(a.out_dir)
    if a.in_place:
        log("ACHTUNG: --in-place überschreibt out/tracks.jsonl, events.json, stats.json, dashboard.html")

    t0 = time.time()
    log(f"smoke test start -> {rel(od)}")
    cut = make_cut(force=a.force)

    if not a.no_wait:
        busy = gpu_jobs()
        if busy:
            log(f"GPU belegt ({len(busy)} Modell-Job(s)), warte bis zu {a.max_wait:.0f} s")
        if not wait_for_gpu(a.max_wait, log=log):
            log("FAIL: GPU nach Wartezeit immer noch belegt")
            return 2
        log("GPU frei")

    skip = ["qa"] if a.in_place else ["qa"]
    if not a.with_numbers:
        skip.append("numbers")
    cmd = [PY, "-m", "vision.run_all", "--clip", str(cut), "--out-dir", str(od), "--skip", ",".join(skip)]
    if a.force:
        cmd.append("--force")
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        log(f"FAIL: run_all exit {proc.returncode} nach {time.time() - t0:.0f} s (siehe {rel(od / 'run_all.log')})")
        return 1

    fails = check_contract(od)
    if fails:
        for f in fails:
            log("FAIL: " + f)
        return 1
    log(f"PASS in {time.time() - t0:.0f} s: {rel(od / 'dashboard.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
