"""Score out/identities.json against Sami's QA verdicts (out/qa/verdicts_<clip>.json).

    .venv/bin/python -m vision.numbers.eval [--verdicts out/qa/verdicts_dev60.json]

QA schema (vision/qa): "numbers" cards carry key, track_ids, detected, true_number
(null = detected is right when detected is set) and unreadable; "shots" carry
player_id, number, number_team. Cards are matched to the current identities by
key first, then by track_ids overlap, so a re-merge that changed keys still
scores. Prints per-card verdicts and totals; exit code 1 when something is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDENTITIES = ROOT / "out" / "identities.json"


def match_player(card: dict, players: list[dict]) -> dict | None:
    for p in players:
        if p["key"] == card.get("key"):
            return p
    ids = set(card.get("track_ids", []))
    best = max(players, key=lambda p: len(ids & set(p["track_ids"])), default=None)
    return best if best and ids & set(best["track_ids"]) else None


def score(identities: dict, verdicts: dict) -> tuple[list[str], int, int, int]:
    lines, right, wrong, open_ = [], 0, 0, 0
    players = identities["players"]
    for card in verdicts.get("numbers", []):
        p = match_player(card, players)
        if p is None:
            lines.append(f"  {card.get('key')}: no matching player any more")
            continue
        truth = card.get("true_number")
        if truth is None and card.get("detected") is not None and not card.get("unreadable"):
            truth = card["detected"]
        if card.get("unreadable"):
            status = "ok (unreadable, open)" if p["number"] is None else f"WRONG: read {p['number']} on an unreadable shirt"
        elif truth is None:
            status = "no label"
        elif p["number"] is None:
            status = f"open (truth {truth})"
        elif p["number"] == truth:
            status = "ok"
        else:
            status = f"WRONG: {p['number']} != {truth}"
        right += status == "ok"
        wrong += status.startswith("WRONG")
        open_ += status.startswith("open")
        lines.append(f"  {card.get('key')} -> {p['key']} ids {p['track_ids']}: {status}")
    for shot in verdicts.get("shots", []):
        tr = identities["tracks"].get(str(shot.get("player_id")))
        if tr is None or shot.get("number") is None:
            continue
        got = (tr["team"], tr["number"])
        want = (shot.get("number_team"), shot["number"])
        status = "ok" if got == want else f"WRONG: {got} != {want}"
        right += status == "ok"
        wrong += status != "ok"
        lines.append(f"  shot {shot.get('n')} t={shot.get('t')} id {shot.get('player_id')} -> {tr['key']}: {status}")
    return lines, right, wrong, open_


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identities", type=Path, default=IDENTITIES)
    ap.add_argument("--verdicts", type=Path, default=None, help="default: out/qa/verdicts_<clip stem>.json")
    a = ap.parse_args(argv)
    identities = json.load(open(a.identities))
    verdicts_path = a.verdicts or ROOT / "out" / "qa" / f"verdicts_{Path(identities['clip']).stem}.json"
    if not verdicts_path.exists():
        print(f"no verdicts yet: {verdicts_path}")
        return 0
    verdicts = json.load(open(verdicts_path))
    lines, right, wrong, open_ = score(identities, verdicts)
    print("\n".join(lines))
    print(f"NUMBERS eval {verdicts_path.name}: {right} right, {wrong} wrong, {open_} open")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
