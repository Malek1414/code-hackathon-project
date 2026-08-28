"""Re-run the QA sheets whenever out/events.json or out/tracks.jsonl changes.

    .venv/bin/python -m vision.qa.watch            # loop, log in out/qa/watch.log
    .venv/bin/python -m vision.qa.watch --once     # one pass, then exit

Plain mtime polling. A file is only picked up once its size and mtime have
been stable for STABLE_S (TRACK streams tracks.jsonl line by line).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .clips import OVERLAY
from .ball_check import DEFAULT_REJECTS
from .numbers_sheet import IDENTITIES
from .common import EVENTS, OUT as OUT_DIR, QA_DIR, ROOT, TRACKS

POLL_S, STABLE_S = 2.0, 4.0
PY = sys.executable
JOBS = {  # module -> files it depends on; order = priority (shot videos first, ball_check last)
    "vision.qa.ball_eval": (TRACKS,),  # cheap: reads 12 jpgs, exits at once on non-game10 tracks
    "vision.qa.shot_sheets": (TRACKS, EVENTS, OVERLAY, IDENTITIES),
    "vision.qa.ball_recall": (TRACKS,),
    "vision.qa.team_check": (TRACKS,),
    "vision.qa.ball_check": (TRACKS, *DEFAULT_REJECTS),
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


ARCHIVE_GLOB = "*_v[0-9]*/tracks.jsonl"  # out/dev60_v4/tracks.jsonl -> out/qa/dev60_v4/
ARCHIVE_JOBS = ("vision.qa.ball_eval", "vision.qa.ball_recall", "vision.qa.ball_check")  # ORCH 13:59: eval decides the publish, the render comes last
STATE = QA_DIR / "watch_state.json"  # signatures already processed, so a restart does not redo unchanged files
archive_done: dict[Path, tuple[float, int] | None] = {}
archive_seen: dict[Path, tuple[tuple[float, int] | None, float]] = {}


def load_state() -> dict[str, list]:
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(done: dict[Path, tuple[float, int] | None]) -> None:
    state = load_state()
    state.update({str(k.relative_to(ROOT)): list(v) for k, v in done.items() if v})
    STATE.write_text(json.dumps(state, indent=1))


def stored_sig(path: Path) -> tuple[float, int] | None:
    v = load_state().get(str(path.relative_to(ROOT)))
    return (float(v[0]), int(v[1])) if v else None


def archive_complete(tracks: Path) -> bool:
    """TRACK writes tracks_meta.json (with last_frame) first and streams the lines; a
    4 s pause must not pass a fragment as a finished run (that happened with dev60_v4)."""
    meta = {}
    try:
        meta = json.loads(tracks.with_name("tracks_meta.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    last, stride = meta.get("last_frame"), int(meta.get("stride") or 1)
    if last is None:
        return False
    try:
        with tracks.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            lines = [ln for ln in fh.read().decode("utf-8", "ignore").splitlines() if ln.strip()]
        last_written = json.loads(lines[-1])["frame"] if lines else -1
    except (OSError, json.JSONDecodeError, KeyError, IndexError):
        return False
    return last_written >= int(last) - max(2 * stride, int(0.05 * int(last)))  # container frame count can exceed the decodable end


def run_archives() -> None:
    """Ball sheets for archived runs (ORCH 13:39: contract paths stay game10, dev60 vN land in out/dev60_vN/)."""
    now = time.time()
    for tracks in sorted((OUT_DIR).glob(ARCHIVE_GLOB)):
        cur = sig(tracks)
        last, since = archive_seen.get(tracks, (None, now))
        if cur != last:
            archive_seen[tracks] = (cur, now)
            continue
        if tracks not in archive_done:
            archive_done[tracks] = stored_sig(tracks)
        if cur is None or cur == archive_done.get(tracks) or now - since < STABLE_S:
            continue
        if not archive_complete(tracks):
            continue  # still being written (or no meta): wait for the last frame
        archive_done[tracks] = cur
        save_state({tracks: cur})
        out = QA_DIR / tracks.parent.name
        log(f"archive {tracks.parent.name}: {tracks.relative_to(ROOT)} -> {out.relative_to(ROOT)}")
        for m in ARCHIVE_JOBS:
            run(m, ["--tracks", str(tracks), "--out", str(out)])


def run(module: str, extra: list[str] | None = None) -> None:
    t = time.time()
    proc = subprocess.run([PY, "-m", module, *(extra or [])], cwd=ROOT, capture_output=True, text=True)
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
        run_archives()
        return 0
    done: dict[Path, tuple[float, int] | None] = {p: stored_sig(p) for p in (TRACKS, EVENTS, OVERLAY, IDENTITIES, *DEFAULT_REJECTS)}  # signature last processed (persisted)
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
            save_state({p: done[p] for p in changed})
            for m, deps in JOBS.items():
                if changed & set(deps):
                    run(m)
                    run_archives()  # archived runs get their turn between the main jobs
        run_archives()
        time.sleep(POLL_S)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
