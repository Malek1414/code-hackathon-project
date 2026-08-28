"""PIPELINE: one command runs the whole analytics pipeline on a clip.

    .venv/bin/python -m vision.run_all --clip data/clips/dev60.mp4 [--weights models/best.pt]
        [--calib out/court_calib_dev60.json] [--out-dir out] [--skip numbers,qa] [--force] [--dry-run]

Steps, in order (each is skipped when it already ran on the same inputs and its
outputs still exist; the stamp lives in <out-dir>/.run_all/<step>.json):

  TRACK     vision/track/run.py            -> tracks.jsonl, tracks_meta.json, overlay.mp4
  NUMBERS   vision.numbers.read + merge    -> identities.json (read.py also writes out/numbers_reads.json)
  COURT     vision/court/propagate.py + minimap.py, only with a calibration for this clip
  STATS     vision.stats.build             -> events.json, stats.json
  QA        vision.qa.watch --once         -> out/qa/ (fixed paths, only when --out-dir is out)
  FRONTEND  vision/dashboard/build.py      -> dashboard.html

Flags were taken from each CLI's --help (docs/tasks/PIPELINE.md), nothing is guessed.
One status line per step with the elapsed seconds; the first failure stops the
run and prints the failing command plus the tail of its log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python vision/run_all.py` as well as `-m vision.run_all`
    sys.path.insert(0, str(ROOT))

PY = sys.executable
STEP_ORDER = ("track", "numbers", "court", "stats", "qa", "frontend")

# Processes that use the GPU (the MPS schedule in docs/ORCHESTRATION.md). The smoke
# test waits while any of these run; run_all itself only waits with --wait-for-gpu.
GPU_JOB_PATTERNS = (
    "vision/track/", "vision.track.",  # anything of TRACK (run, compare, experiments), ORCH rule 12:52
    "vision/label/train.py", "vision.label.train", "vision/label/autolabel.py", "vision.label.autolabel",
    "vision/live/live.py", "vision.live.live", "vision/run_all.py", "vision.run_all", "vision.smoke_test", "vision/smoke_test.py",
)


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def show(cmd: list[str]) -> str:
    """Command line for humans: repo-relative paths, .venv python as-is."""
    prefix = str(ROOT) + "/"
    return " ".join(c[len(prefix):] if c.startswith(prefix) else c for c in cmd)  # no resolve(): .venv/bin/python is a symlink


def sig(path: Path) -> list | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return [rel(path), st.st_mtime, st.st_size]


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def gpu_jobs(ignore_pid: int | None = None) -> list[str]:
    """Other .venv python processes that look like model jobs (ps -axo pid,command)."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    me = {os.getpid(), os.getppid()}
    if ignore_pid:
        me.add(ignore_pid)
    jobs = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, cmd = int(parts[0]), parts[1]
        if pid in me or ".venv/bin/python" not in cmd or "/bin/zsh" in cmd:
            continue
        if any(pat in cmd for pat in GPU_JOB_PATTERNS):
            jobs.append(f"{pid} {cmd[:110]}")
    return jobs


def wait_for_gpu(max_wait_s: float, poll_s: float = 10.0, log=print) -> bool:
    """Block until no other model job runs. Returns False on timeout."""
    t0 = time.time()
    last = None
    while True:
        jobs = gpu_jobs()
        if not jobs:
            return True
        if jobs != last:
            log(f"warte auf GPU, laufende Modell-Jobs: {len(jobs)}")
            for j in jobs:
                log(f"    {j}")
            last = jobs
        if time.time() - t0 > max_wait_s:
            return False
        time.sleep(poll_s)


@dataclass
class Step:
    name: str
    cmds: list[list[str]]
    inputs: list[Path]
    outputs: list[Path]
    skip_reason: str | None = None  # set when the step cannot run at all (no calib etc.)
    note: str = ""
    log_path: Path | None = None
    stamp: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, a: argparse.Namespace):
        self.clip = Path(a.clip)
        if not self.clip.exists():
            raise SystemExit(f"Clip fehlt: {a.clip}")
        self.clip = Path(rel(self.clip))  # repo-relative everywhere: stable stamps, short logs (subprocess cwd is ROOT)
        self.stem = self.clip.stem
        self.od = Path(a.out_dir)
        self.od.mkdir(parents=True, exist_ok=True)
        self.od = Path(rel(self.od))
        self.stamps = self.od / ".run_all"
        self.stamps.mkdir(exist_ok=True)
        self.in_place = self.od.resolve() == (ROOT / "out").resolve()
        self.weights = Path(a.weights) if a.weights else None
        if self.weights and not self.weights.exists():
            raise SystemExit(f"Gewichte fehlen: {a.weights}")
        if self.weights:
            self.weights = Path(rel(self.weights))
        self.calib = Path(rel(Path(a.calib) if a.calib else self.default_calib()))
        self.force = a.force
        self.dry = a.dry_run
        self.stride = a.stride
        self.skip = {s.strip() for s in (a.skip or "").split(",") if s.strip()}
        self.only = {s.strip() for s in (a.only or "").split(",") if s.strip()}
        unknown = (self.skip | self.only) - set(STEP_ORDER)
        if unknown:
            raise SystemExit(f"unbekannte Schritte: {', '.join(sorted(unknown))} (erlaubt: {', '.join(STEP_ORDER)})")
        self.log_file = self.od / "run_all.log"
        self.team_a, self.team_b = a.team_a, a.team_b
        self.wait_gpu_s = a.wait_for_gpu

    # -- paths -------------------------------------------------------------
    def default_calib(self) -> Path:
        per_clip = ROOT / "out" / f"court_calib_{self.stem}.json"
        return per_clip if per_clip.exists() else ROOT / "out" / "court_calib.json"

    def p(self, name: str) -> Path:
        return self.od / name

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        try:
            with self.log_file.open("a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    # -- step definitions (flags from each CLI's --help) --------------------
    def steps(self) -> list[Step]:
        tracks, meta = self.p("tracks.jsonl"), self.p("tracks_meta.json")
        identities, events, stats = self.p("identities.json"), self.p("events.json"), self.p("stats.json")
        court_h, minimap, overlay, dashboard = self.p(f"court_H_{self.stem}.npz"), self.p("minimap.mp4"), self.p("overlay.mp4"), self.p("dashboard.html")
        steps: list[Step] = []

        # TRACK
        cmd = [PY, "vision/track/run.py", "--video", str(self.clip), "--out", str(tracks), "--overlay", str(overlay),
               "--events", str(events), "--identities", str(identities), "--calib", str(self.calib)]
        if self.weights:
            cmd += ["--weights", str(self.weights)]
        if self.stride:
            cmd += ["--stride", str(self.stride)]
        track_inputs = [self.clip] + ([self.weights] if self.weights else [])
        steps.append(Step("track", [cmd], track_inputs, [tracks, meta]))

        # NUMBERS: read.py has fixed outputs in out/ (numbers_reads.json, preview, cache); merge takes --reads/--out
        reads = Path("out/numbers_reads.json")
        steps.append(Step(
            "numbers",
            [[PY, "-m", "vision.numbers.read", "--tracks", str(tracks), "--video", str(self.clip)],
             [PY, "-m", "vision.numbers.merge", "--reads", str(reads), "--out", str(identities)]],
            [tracks], [identities],
            note="" if self.in_place else "read.py schreibt out/numbers_reads.json und numbers_preview.jpg (feste Pfade)",
        ))

        # COURT: only with a calibration that belongs to this clip and has keyframes
        court = Step(
            "court",
            [[PY, "vision/court/propagate.py", str(self.clip), "--calib", str(self.calib), "--tracks", str(tracks), "--out", str(court_h)],
             [PY, "vision/court/minimap.py", "--tracks", str(tracks), "--calib", str(self.calib), "--clip", str(self.clip), "--out", str(minimap)]],
            [tracks, self.calib], [court_h, minimap],
        )
        cal = load_json(self.calib) if self.calib.exists() else None
        if cal is None:
            court.skip_reason = f"keine Kalibrierung ({rel(self.calib)} fehlt)"
        elif cal.get("clip") not in (None, rel(self.clip), str(self.clip)):
            court.skip_reason = f"{rel(self.calib)} gehört zu {cal.get('clip')}, nicht zu {rel(self.clip)}"
        elif not (cal.get("frames") or cal.get("H_px_to_m")):
            court.skip_reason = f"keine Keyframes in {rel(self.calib)}"
        steps.append(court)

        # STATS
        cmd = [PY, "-m", "vision.stats.build", "--tracks", str(tracks), "--clip", str(self.clip), "--calib", str(self.calib),
               "--out-dir", str(self.od), "--identities", str(identities)]
        steps.append(Step("stats", [cmd], [tracks, identities, self.calib], [events, stats]))

        # QA: fixed paths out/tracks.jsonl, out/events.json, out/overlay.mp4 -> out/qa/
        qa = Step("qa", [[PY, "-m", "vision.qa.watch", "--once"]], [tracks, events, overlay], [])
        if not self.in_place:
            qa.skip_reason = "vision.qa.watch kennt nur out/ (keine Pfad-Flags), nur mit --out-dir out"
        steps.append(qa)

        # FRONTEND
        cmd = [PY, "vision/dashboard/build.py", "--events", str(events), "--stats", str(stats), "--calib", str(self.calib),
               "--tracks", str(tracks), "--tracks-meta", str(meta), "--identities", str(identities),
               "--minimap", minimap.name if minimap.exists() else "", "--overlay", overlay.name if overlay.exists() else "",
               "--out", str(dashboard)]
        if self.team_a:
            cmd += ["--team-a", self.team_a]
        if self.team_b:
            cmd += ["--team-b", self.team_b]
        steps.append(Step("frontend", [cmd], [events, stats, identities, self.calib, minimap, overlay, meta], [dashboard]))
        return steps

    # -- stamps ------------------------------------------------------------
    def stamp_of(self, step: Step) -> dict:
        return {"clip": rel(self.clip), "cmds": [show(c) for c in step.cmds], "inputs": [sig(p) for p in step.inputs]}

    def is_current(self, step: Step) -> tuple[bool, str]:
        if self.force:
            return False, "erzwungen"
        missing = [o for o in step.outputs if not o.exists()]
        if missing:
            return False, f"{rel(missing[0])} fehlt"
        old = load_json(self.stamps / f"{step.name}.json")
        if old is None:
            return False, "noch nie gelaufen"
        if old != self.stamp_of(step):
            return False, "Eingaben geändert"
        if step.name == "track":
            meta = load_json(self.p("tracks_meta.json")) or {}
            if meta.get("clip") not in (rel(self.clip), str(self.clip)):
                return False, f"tracks_meta.json gehört zu {meta.get('clip')}"
        return True, "aktuell"

    def write_stamp(self, step: Step) -> None:
        (self.stamps / f"{step.name}.json").write_text(json.dumps(self.stamp_of(step), indent=1))

    # -- run ---------------------------------------------------------------
    def run_step(self, step: Step) -> bool:
        log_path = self.od / f"pipeline_{step.name}.log"
        step.log_path = log_path
        t0 = time.time()
        with log_path.open("w") as fh:
            for cmd in step.cmds:
                fh.write("$ " + show(cmd) + "\n")
                fh.flush()
                proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
                if proc.returncode != 0:
                    dt = time.time() - t0
                    self.log(f"{step.name.upper():9s} FEHLER nach {dt:6.1f} s (exit {proc.returncode})")
                    self.log(f"    Befehl: {show(cmd)}")
                    self.log(f"    Log:    {rel(log_path)}")
                    tail = log_path.read_text(errors="replace").splitlines()[-12:]
                    for ln in tail:
                        print("    | " + ln)
                    return False
        missing = [o for o in step.outputs if not o.exists()]
        dt = time.time() - t0
        if missing:
            self.log(f"{step.name.upper():9s} FEHLER nach {dt:6.1f} s: Ausgabe fehlt {rel(missing[0])}")
            self.log(f"    Befehl: {show(step.cmds[-1])}")
            return False
        self.write_stamp(step)
        outs = ", ".join(rel(o) for o in step.outputs) or "out/qa/"
        self.log(f"{step.name.upper():9s} ok     {dt:6.1f} s  -> {outs}")
        return True

    def run(self) -> int:
        t_all = time.time()
        self.log(f"run_all: {rel(self.clip)} -> {rel(self.od)}" + (f", Gewichte {rel(self.weights)}" if self.weights else "")
                 + f", Kalibrierung {rel(self.calib)}" + (" (fehlt)" if not self.calib.exists() else ""))
        for name in STEP_ORDER:
            step = next(st for st in self.steps() if st.name == name)  # fresh: earlier steps change the inputs
            if (self.only and step.name not in self.only) or step.name in self.skip:
                self.log(f"{step.name.upper():9s} übersprungen (per Flag)")
                continue
            if step.skip_reason:
                self.log(f"{step.name.upper():9s} übersprungen: {step.skip_reason}")
                continue
            current, why = self.is_current(step)
            if current:
                self.log(f"{step.name.upper():9s} aktuell, übersprungen")
                continue
            if step.note:
                self.log(f"{step.name.upper():9s} Hinweis: {step.note}")
            if self.dry:
                self.log(f"{step.name.upper():9s} würde laufen ({why}):")
                for cmd in step.cmds:
                    print("    $ " + show(cmd))
                continue
            if step.name == "track" and self.wait_gpu_s and not wait_for_gpu(self.wait_gpu_s, log=self.log):
                self.log("GPU vor TRACK immer noch belegt, Abbruch")
                return 2
            self.log(f"{step.name.upper():9s} läuft ({why})")
            if not self.run_step(step):
                self.log(f"abgebrochen nach {time.time() - t_all:.1f} s")
                return 1
        self.log(f"fertig in {time.time() - t_all:.1f} s: {rel(self.p('dashboard.html'))}")
        return 0


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--clip", required=True, help="Video, z.B. data/clips/dev60.mp4")
    ap.add_argument("--weights", default=None, help="single contract model for TRACK (LABEL's best.pt); default: TRACK's two-model setup")
    ap.add_argument("--calib", default=None, help="Kalibrierung; Standard out/court_calib_<clip>.json, sonst out/court_calib.json")
    ap.add_argument("--out-dir", default="out", help="Ausgabeverzeichnis (Vertrag: out)")
    ap.add_argument("--skip", default="", help=f"Schritte auslassen, kommagetrennt aus {','.join(STEP_ORDER)}")
    ap.add_argument("--only", default="", help="nur diese Schritte laufen lassen, kommagetrennt")
    ap.add_argument("--stride", type=int, default=None, help="an TRACK durchgereicht (jeden N-ten Frame)")
    ap.add_argument("--team-a", default=None, help="Vereinsname Team 0 fürs Dashboard")
    ap.add_argument("--team-b", default=None, help="Vereinsname Team 1 fürs Dashboard")
    ap.add_argument("--force", action="store_true", help="alle Schritte neu rechnen")
    ap.add_argument("--dry-run", action="store_true", help="nur zeigen, was laufen würde")
    ap.add_argument("--wait-for-gpu", type=float, default=0, metavar="SEK", help="vor dem Start bis zu SEK Sekunden warten, bis kein anderer Modell-Job läuft")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    pipe = Pipeline(a)
    if a.wait_for_gpu and not a.dry_run:
        if not wait_for_gpu(a.wait_for_gpu, log=pipe.log):
            pipe.log("GPU nach Wartezeit immer noch belegt, Abbruch")
            return 2
    return pipe.run()


if __name__ == "__main__":
    sys.exit(main())
