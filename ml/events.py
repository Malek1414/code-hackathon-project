#!/usr/bin/env python3
"""First event heads on top of tracks.jsonl: ball possession + shot candidates.

Scope (honest): with COCO classes only (person, sports ball) we can compute
who has the ball and when a shot ATTEMPT likely happens (ball rises above
every player for a stretch). Made/missed classification needs a hoop
detector — that's the fine-tune step in docs/ML_DATA_PLAN.md, not this file.

Usage:
  ml/.venv/bin/python ml/events.py clip_analysis/tracks.jsonl
"""
import argparse
import json
from pathlib import Path

POSSESS_MARGIN = 40      # px: ball center within this of a person box = held
SHOT_MIN_FRAMES = 4      # consecutive airborne frames to call a shot candidate


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def possession_of(ball, persons):
    """Track id of the person holding the ball, or None."""
    bx, by = center(ball["xyxy"])
    best = None
    for p in persons:
        x1, y1, x2, y2 = p["xyxy"]
        if (x1 - POSSESS_MARGIN <= bx <= x2 + POSSESS_MARGIN
                and y1 - POSSESS_MARGIN <= by <= y2 + POSSESS_MARGIN):
            area = (x2 - x1) * (y2 - y1)
            if best is None or area < best[1]:  # smallest enclosing = closest
                best = (p["id"], area)
    return best[0] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    possession = []          # (frame, person_id | None)
    airborne_run = 0
    shot_candidates = []     # {"start": f, "end": f}

    with open(a.tracks) as f:
        for line in f:
            rec = json.loads(line)
            persons = [o for o in rec["objects"] if o["cls"] == "person"]
            balls = [o for o in rec["objects"] if o["cls"] == "sports ball"]
            ball = max(balls, key=lambda o: o["conf"]) if balls else None

            holder = possession_of(ball, persons) if ball else None
            possession.append((rec["frame"], holder))

            # airborne: ball visible, above the top of every person box
            airborne = bool(ball and persons and
                            center(ball["xyxy"])[1] < min(p["xyxy"][1] for p in persons))
            if airborne:
                airborne_run += 1
                if airborne_run == SHOT_MIN_FRAMES:
                    shot_candidates.append({"start": rec["frame"] - SHOT_MIN_FRAMES + 1})
            else:
                if airborne_run >= SHOT_MIN_FRAMES:
                    shot_candidates[-1]["end"] = rec["frame"] - 1
                airborne_run = 0
    if airborne_run >= SHOT_MIN_FRAMES:
        shot_candidates[-1]["end"] = possession[-1][0]

    # collapse possession into runs
    runs, cur = [], None
    for frame, holder in possession:
        if cur is None or holder != cur["holder"]:
            cur = {"holder": holder, "start": frame, "end": frame}
            runs.append(cur)
        else:
            cur["end"] = frame

    out = Path(a.out) if a.out else Path(a.tracks).parent / "events.json"
    with open(out, "w") as f:
        json.dump({"possession_runs": runs, "shot_candidates": shot_candidates}, f, indent=1)

    held = sum(r["end"] - r["start"] + 1 for r in runs if r["holder"] is not None)
    print(f"{len(possession)} frames: {held} with possession, "
          f"{len(shot_candidates)} shot candidate(s) -> {out}")


if __name__ == "__main__":
    main()
