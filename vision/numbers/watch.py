"""Re-run read.py + merge.py whenever out/tracks.jsonl changes (mtime loop).

    .venv/bin/python -m vision.numbers.watch [--once] [--poll 10] [--settle 20]

`settle`: tracks.jsonl is appended frame by frame while TRACK runs; we wait
until its mtime has been quiet for that many seconds before re-reading, so we
do not OCR the same half-written file every poll. The OCR cache makes a re-run
after new frames cheap (only new crops are read).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from vision.numbers import merge, read

log = logging.getLogger("numbers.watch")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="run once and exit")
    ap.add_argument("--poll", type=float, default=10.0, help="seconds between mtime checks")
    ap.add_argument("--settle", type=float, default=20.0, help="seconds of unchanged mtime before a run")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)

    done_mtime = None
    while True:
        try:
            mtime = read.TRACKS.stat().st_mtime
        except FileNotFoundError:
            log.info("waiting for %s", read.TRACKS)
            time.sleep(a.poll)
            continue
        quiet = time.time() - mtime
        if mtime != done_mtime and (quiet >= a.settle or a.once):
            log.info("tracks.jsonl changed (quiet %.0f s), running", quiet)
            try:
                r = read.run()
                m = merge.run()
                s = m["summary"]
                print(f"NUMBERS {s['tracks_with_number']}/{s['tracks']} tracks got a number "
                      f"({100 * s['share']:.0f}%), {s['players_with_number']} numbered players", flush=True)
            except Exception:  # keep the loop alive, TRACK may be mid-write
                log.exception("run failed")
            done_mtime = mtime
            if a.once:
                return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
