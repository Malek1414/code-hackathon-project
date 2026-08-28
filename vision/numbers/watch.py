"""Re-run read.py + merge.py whenever out/tracks.jsonl changes (mtime loop).

    .venv/bin/python -m vision.numbers.watch [--once] [--poll 10] [--settle 20] [--tracks out/game10/tracks.jsonl] [--publish out/identities.json]

Two passes per change (read.PASSES): tracks >= 2 s first, identities.json is
written right after, then the fragments and identities.json is rewritten.

`settle`: tracks.jsonl is appended frame by frame while TRACK runs; we wait
until its mtime has been quiet for that many seconds before re-reading, so we
do not OCR the same half-written file every poll. The OCR cache makes a re-run
after new frames cheap (only new crops are read).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time

from pathlib import Path

from vision.numbers import merge, read

log = logging.getLogger("numbers.watch")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="run once and exit")
    ap.add_argument("--poll", type=float, default=10.0, help="seconds between mtime checks")
    ap.add_argument("--settle", type=float, default=20.0, help="seconds of unchanged mtime before a run")
    ap.add_argument("--tracks", type=Path, default=read.TRACKS)
    ap.add_argument("--publish", type=Path, default=None,
                    help="also copy identities.json here after every pass (contract path out/identities.json)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)

    done_mtime = None
    while True:
        try:
            mtime = a.tracks.stat().st_mtime
        except FileNotFoundError:
            log.info("waiting for %s", a.tracks)
            time.sleep(a.poll)
            continue
        quiet = time.time() - mtime
        if mtime != done_mtime and (quiet >= a.settle or a.once):
            log.info("tracks.jsonl changed (quiet %.0f s), running", quiet)
            try:
                # pass 1: long tracks only, identities.json out fast; pass 2: the fragments, rewrite
                shooters = read.shooter_ids(a.tracks)
                passes = [(0.0, read.MAX_CROPS, shooters)] if shooters else []  # shooters first, whatever their length
                passes += [(min_s, crops, None) for min_s, crops in read.PASSES]
                for min_s, crops, only in passes:
                    read.run(a.tracks, min_track_s=min_s, max_crops=crops, only_ids=only, priority_ids=shooters)
                    m = merge.run(a.tracks.parent / read.READS_NAME, a.tracks.parent / "identities.json")
                    if a.publish:
                        shutil.copyfile(a.tracks.parent / "identities.json", a.publish)
                    s = m["summary"]
                    print(f"NUMBERS pass {'shooters' if only else f'>= {min_s:.0f} s'}: {s['tracks_with_number']}/{s['tracks']} tracks got a "
                          f"number ({100 * s['share']:.0f}%), {s['players_with_number']} numbered players", flush=True)
                    if a.tracks.stat().st_mtime != mtime:
                        break  # new frames arrived, start over with the long tracks
            except Exception:  # keep the loop alive, TRACK may be mid-write
                log.exception("run failed")
            done_mtime = mtime
            if a.once:
                return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
