"""Score events.json against Sami's verdict sheet (out/qa/verdicts_<clip>.json).

    .venv/bin/python -m vision.stats.eval_human --events out/events.json --verdicts out/qa/verdicts_game10.json \
        --out out/qa/stats_eval_game10.json

Verdict schema (QA): shots[].t, made (system verdict at review time), player_id
(system shooter at review time), shot = ok | flipped | no_shot | open,
shooter_ok, number / number_team (jersey the human read), note.
Verdicts are matched to the current events by time (+-1 s), so a rebuilt
events.json can be re-scored; a shooter that changed since the review is
counted separately (the human judged the old one).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def match(verdicts: list[dict], shots: list[dict], tol_s: float = 1.0) -> list[tuple[dict, dict | None]]:
    pairs = []
    used: set[int] = set()
    for v in verdicts:
        best, best_d = None, tol_s
        for i, s in enumerate(shots):
            d = abs(s["t"] - v["t"])
            if i not in used and d <= best_d:
                best, best_d = i, d
        if best is not None:
            used.add(best)
            pairs.append((v, shots[best]))
        else:
            pairs.append((v, None))
    return pairs


def evaluate(events: dict, verdicts: dict) -> dict:
    shots = events["shots"]
    pairs = match(verdicts["shots"], shots)
    answered = [(v, s) for v, s in pairs if v.get("shot") in ("ok", "flipped", "no_shot")]
    real = [(v, s) for v, s in answered if v["shot"] != "no_shot"]
    no_shot = [v for v, _ in answered if v["shot"] == "no_shot"]

    verdict_ok = [(v, s) for v, s in real if v["shot"] == "ok"]
    flipped = [(v, s) for v, s in real if v["shot"] == "flipped"]

    # shooter: the human judged the shooter shown at review time (v.player_id)
    shooter_same = [(v, s) for v, s in real if s is not None and s.get("player_id") == v.get("player_id")]
    shooter_changed = [(v, s) for v, s in real if s is not None and s.get("player_id") != v.get("player_id")]
    shooter_ok_same = [1 for v, s in shooter_same if v.get("shooter_ok")]
    # team of the shooter vs the jersey team the human typed (independent of track ids)
    team_pairs = [(v, s) for v, s in real if s is not None and v.get("number_team") in (0, 1) and s.get("team") in (0, 1)]
    team_ok = [1 for v, s in team_pairs if v["number_team"] == s["team"]]

    flips = []
    for v, s in flipped:
        true_made = not v["made"]
        flips.append(
            {
                "t": v["t"],
                "system_made_at_review": v["made"],
                "true_made": true_made,
                "current_made": None if s is None else s.get("made"),
                "made_confirmed": None if s is None else s.get("made_confirmed"),
                "made_hint": None if s is None else s.get("made_hint"),
                "unconfirmed_counted_as_miss": bool(s and not s.get("made_confirmed") and true_made),
                "note": v.get("note", ""),
            }
        )
    unconfirmed_flips = sum(1 for f in flips if f["unconfirmed_counted_as_miss"])

    def pct(a: int, b: int) -> float | None:
        return round(a / b, 3) if b else None

    return {
        "clip": events.get("clip"),
        "verdicts_reviewed": verdicts.get("reviewed"),
        "system_shots": len(shots),
        "verdicts_answered": len(answered),
        "verdicts_open": sum(1 for v, _ in pairs if v.get("shot") not in ("ok", "flipped", "no_shot")),
        "attempts": {"real": len(real), "no_shot": len(no_shot), "precision": pct(len(real), len(answered))},
        "made_miss": {
            "right": len(verdict_ok),
            "flipped": len(flipped),
            "accuracy": pct(len(verdict_ok), len(real)),
            "flips_where_unconfirmed_was_counted_as_miss": unconfirmed_flips,
            "accuracy_if_unconfirmed_makes_were_asked": pct(len(verdict_ok) + unconfirmed_flips, len(real)),
        },
        "shooter": {
            "judged_on_current_shooter": len(shooter_same),
            "right": len(shooter_ok_same),
            "accuracy": pct(len(shooter_ok_same), len(shooter_same)),
            "shooter_changed_since_review": len(shooter_changed),
            "team_pairs": len(team_pairs),
            "team_right": len(team_ok),
            "team_accuracy": pct(len(team_ok), len(team_pairs)),
        },
        "flipped_shots": flips,
        "no_shot": [{"t": v["t"], "note": v.get("note", "")} for v in no_shot],
        "unmatched_verdicts": [v["t"] for v, s in pairs if s is None],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default="out/events.json")
    ap.add_argument("--verdicts", default="out/qa/verdicts_game10.json")
    ap.add_argument("--out", default="out/qa/stats_eval_game10.json")
    args = ap.parse_args(argv)
    events = json.loads(Path(args.events).read_text())
    verdicts = json.loads(Path(args.verdicts).read_text())
    result = evaluate(events, verdicts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    a, m, s = result["attempts"], result["made_miss"], result["shooter"]
    print(f"attempts: {a['real']} real of {result['verdicts_answered']} answered (precision {a['precision']}), {a['no_shot']} no-shot")
    print(f"made/miss: {m['right']}/{a['real']} right ({m['accuracy']}), {m['flipped']} flipped, "
          f"{m['flips_where_unconfirmed_was_counted_as_miss']} of them unconfirmed-counted-as-miss "
          f"-> {m['accuracy_if_unconfirmed_makes_were_asked']} if those were asked")
    print(f"shooter: {s['right']}/{s['judged_on_current_shooter']} right on the shooter the human saw ({s['accuracy']}), "
          f"{s['shooter_changed_since_review']} changed since review; team right {s['team_right']}/{s['team_pairs']} ({s['team_accuracy']})")
    for f in result["flipped_shots"]:
        print(f"  flip {f['t']:7.2f}s true_made={f['true_made']} current_made={f['current_made']} "
              f"confirmed={f['made_confirmed']} hint={f['made_hint']} {f['note']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
