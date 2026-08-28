#!/usr/bin/env python3
"""Correlate heart rate with on-court events: mistake/attempt rate per HR
zone, and in-bounds vs out-of-bounds comparison.

Inputs: hr_<stamp>.csv from the app (unix_ts,bpm; first row = video t 0) and
events.json from ml/events.py. Events default to shot candidates; pass
--mistakes with hand-tagged frame ranges (csv: start_frame,end_frame) for
true mistakes.

Usage:
  ml/.venv/bin/python ml/correlate_hr.py hr_1756380000.csv clip_analysis/events.json \
      --fps 30 --hr-limit 165
"""
import argparse
import csv
import json
from pathlib import Path

ZONES = [(0.0, 0.6, "Z1 <60%"), (0.6, 0.7, "Z2 60-70%"), (0.7, 0.8, "Z3 70-80%"),
         (0.8, 0.9, "Z4 80-90%"), (0.9, 2.0, "Z5 90%+")]


def zone_of(bpm, max_hr):
    f = bpm / max_hr
    return next(name for lo, hi, name in ZONES if lo <= f < hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hr_csv")
    ap.add_argument("events_json")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-hr", type=float, default=190.0)
    ap.add_argument("--hr-limit", type=float, default=165.0,
                    help="'out of bounds' threshold in bpm")
    ap.add_argument("--mistakes", help="csv of start_frame,end_frame hand labels")
    ap.add_argument("--whoop-json", help="whoop_workout.json exported by the app; "
                    "its measured max HR replaces --max-hr for accurate zones")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.whoop_json:
        workout = json.load(open(a.whoop_json))
        measured = (workout.get("score") or {}).get("max_heart_rate")
        if measured:
            a.max_hr = float(measured) / 0.95  # session max ~ 95% of true max
            print(f"WHOOP workout max HR {measured:.0f} -> using max_hr {a.max_hr:.0f}")

    with open(a.hr_csv) as f:
        hr = [(float(r["unix_ts"]), int(r["bpm"])) for r in csv.DictReader(f)]
    if not hr:
        raise SystemExit("empty HR log")
    t0 = hr[0][0]
    samples = [(ts - t0, bpm) for ts, bpm in hr]

    def bpm_at(t):
        return min(samples, key=lambda s: abs(s[0] - t))[1]

    if a.mistakes:
        with open(a.mistakes) as f:
            events = [int(r["start_frame"]) for r in csv.DictReader(f)]
        kind = "mistakes (hand-labeled)"
    else:
        events = [e["start"] for e in json.load(open(a.events_json))["shot_candidates"]]
        kind = "shot attempts"

    minutes = {name: 0.0 for _, _, name in ZONES}
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i - 1][0]
        minutes[zone_of(samples[i][1], a.max_hr)] += dt / 60

    per_zone = {name: 0 for _, _, name in ZONES}
    above = below = 0
    for frame in events:
        bpm = bpm_at(frame / a.fps)
        per_zone[zone_of(bpm, a.max_hr)] += 1
        if bpm > a.hr_limit:
            above += 1
        else:
            below += 1

    min_above = sum((samples[i][0] - samples[i-1][0]) / 60
                    for i in range(1, len(samples)) if samples[i][1] > a.hr_limit)
    min_below = max(samples[-1][0] / 60 - min_above, 1e-9)

    print(f"{len(events)} {kind} over {samples[-1][0]/60:.1f} min")
    print(f"{'zone':<12}{'min':>7}{'events':>8}{'per min':>9}")
    for _, _, name in ZONES:
        rate = per_zone[name] / minutes[name] if minutes[name] > 0.05 else 0
        print(f"{name:<12}{minutes[name]:>7.1f}{per_zone[name]:>8}{rate:>9.2f}")
    ra = above / min_above if min_above > 0.05 else 0
    rb = below / min_below
    print(f"\n> {a.hr_limit:.0f} bpm: {above} events in {min_above:.1f} min "
          f"({ra:.2f}/min)  vs  <= limit: {below} in {min_below:.1f} min ({rb:.2f}/min)")
    if rb > 0 and min_above > 0.05:
        print(f"rate multiplier out-of-bounds: x{ra/rb:.2f}")

    out = Path(a.out) if a.out else Path(a.events_json).parent / "correlation.json"
    json.dump({"kind": kind, "events": len(events), "per_zone": per_zone,
               "minutes_per_zone": {k: round(v, 2) for k, v in minutes.items()},
               "above_limit": {"events": above, "minutes": round(min_above, 2)},
               "below_limit": {"events": below, "minutes": round(min_below, 2)}},
              open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
