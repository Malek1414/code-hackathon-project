"""Re-run the QA sheets whenever out/events.json or out/tracks.jsonl changes.

    .venv/bin/python -m vision.qa.watch            # loop, log in out/qa/watch.log
    .venv/bin/python -m vision.qa.watch --once     # one pass, then exit

Plain mtime polling. A file is only picked up once its size and mtime have
been stable for STABLE_S (TRACK streams tracks.jsonl line by line).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from .clips import OVERLAY
from .ball_check import DEFAULT_REJECTS
from .numbers_sheet import IDENTITIES
from .common import EVENTS, QA_DIR, ROOT, TRACKS

POLL_S, STABLE_S = 2.0, 4.0
PY = sys.executable
JOBS = {  # module -> files it depends on
    "vision.qa.ball_recall": (TRACKS,),
    "vision.qa.team_check": (TRACKS,),
    "vision.qa.ball_check": (TRACKS, *DEFAULT_REJECTS),
    "vision.qa.shot_sheets": (TRACKS, EVENTS, OVERLAY, IDENTITIES),
}


def sig(path: Path) -> tuple[float, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_mtime, st.st_size)


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (QA_DIR / "watch.log").open("a") as fh:
        fh.write(line + "\n")


def run(module: str) -> None:
    t = time.time()
    proc = subprocess.run([PY, "-m", module], cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stdout.strip().splitlines() or [""])[-1]
    if proc.returncode != 0:
        err = (proc.stderr.strip().splitlines() or ["?"])[-1]
        log(f"{module} failed ({proc.returncode}) in {time.time() - t:.0f}s: {err}")
    else:
        log(f"{module} ok in {time.time() - t:.0f}s: {tail}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    (QA_DIR / "watch.pid").write_text(str(os.getpid()))
    if args.once:
        for m in JOBS:
            run(m)
        return 0
    done: dict[Path, tuple[float, int] | None] = {p: None for p in (TRACKS, EVENTS, OVERLAY, IDENTITIES, *DEFAULT_REJECTS)}  # signature last processed
    seen: dict[Path, tuple[tuple[float, int] | None, float]] = {p: (sig(p), 0.0) for p in done}  # (sig, since)
    log(f"watching {TRACKS.relative_to(ROOT)}, {EVENTS.relative_to(ROOT)} and {OVERLAY.relative_to(ROOT)} (+identities.json) every {POLL_S:g}s, pid {os.getpid()}")
    while True:
        now = time.time()
        changed: set[Path] = set()
        for p in done:
            cur = sig(p)
            last, since = seen[p]
            if cur != last:
                seen[p] = (cur, now)
                continue
            if cur is not None and cur != done[p] and now - since >= STABLE_S:
                changed.add(p)
        if changed:
            log("changed: " + ", ".join(p.name for p in sorted(changed)))
            for p in changed:
                done[p] = seen[p][0]
            for m, deps in JOBS.items():
                if changed & set(deps):
                    run(m)
        time.sleep(POLL_S)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
