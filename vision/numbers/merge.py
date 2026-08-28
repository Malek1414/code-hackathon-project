"""Merge track ids into real players (NUMBERS role) -> out/identities.json.

Input: out/numbers_reads.json (from read.py) and out/tracks.jsonl.
Players = groups of track ids with the same (team, number) whose lifetimes do
not overlap (two boxes with the same number on screen at once are two people or
one wrong read; the overlapping id then falls back to its own key). Tracks
without a confident number keep their own key, e.g. A?7 (team letter, ?, id);
team -1 uses the letter "X".

Contract (docs/ORCHESTRATION.md):
{"clip": ..., "tracks": {"7": {"team": 0, "number": 12, "conf": 0.83,
                                "votes": {"12": 9, "17": 1}, "reads": 10}},
 "players": [{"key": "A12", "team": 0, "number": 12, "track_ids": [7, 41, 88],
              "first_t": 0.0, "last_t": 58.2}]}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("numbers.merge")

ROOT = Path(__file__).resolve().parents[2]
READS = ROOT / "out" / "numbers_reads.json"
IDENTITIES = ROOT / "out" / "identities.json"

TEAM_LETTER = {0: "A", 1: "B"}


def letter(team: int) -> str:
    return TEAM_LETTER.get(team, "X")


def own_key(team: int, tid: int) -> str:
    return f"{letter(team)}?{tid}"


def group_players(tracks: dict[str, dict]) -> list[dict]:
    by_identity: dict[tuple[int, int], list[tuple[int, dict]]] = defaultdict(list)
    singles: list[tuple[int, dict]] = []
    for tid_s, tr in tracks.items():
        tid = int(tid_s)
        if tr.get("number") is None or tr.get("team", -1) not in (0, 1):
            singles.append((tid, tr))
        else:
            by_identity[(tr["team"], tr["number"])].append((tid, tr))

    players: list[dict] = []
    for (team, number), members in by_identity.items():
        # greedy: sort by first_t, longest-lived / most confident first when ties;
        # a member overlapping the group's span so far becomes its own player
        members.sort(key=lambda m: (m[1]["first_t"], -m[1].get("conf", 0)))
        groups: list[list[tuple[int, dict]]] = []
        for tid, tr in members:
            placed = False
            for g in groups:
                if all(tr["first_t"] > o["last_t"] or tr["last_t"] < o["first_t"] for _, o in g):
                    g.append((tid, tr))
                    placed = True
                    break
            if not placed:
                groups.append([(tid, tr)])
        groups.sort(key=lambda g: -sum(o["frames"] for _, o in g))
        for i, g in enumerate(groups):
            key = f"{letter(team)}{number}"
            if i:  # a second, overlapping group with the same number: keep the ids visible
                key = f"{key}~{g[0][0]}"
            players.append({
                "key": key, "team": team, "number": number,
                "track_ids": sorted(t for t, _ in g),
                "first_t": min(o["first_t"] for _, o in g),
                "last_t": max(o["last_t"] for _, o in g),
                "frames": sum(o["frames"] for _, o in g),
            })
    for tid, tr in singles:
        players.append({
            "key": own_key(tr.get("team", -1), tid), "team": tr.get("team", -1), "number": None,
            "track_ids": [tid], "first_t": tr["first_t"], "last_t": tr["last_t"], "frames": tr["frames"],
        })
    players.sort(key=lambda p: (p["number"] is None, p["team"] if p["team"] >= 0 else 9, p["number"] or 0, p["first_t"]))
    return players


def run(reads_path: Path = READS, out_path: Path = IDENTITIES) -> dict:
    reads = json.load(open(reads_path))
    tracks_out: dict[str, dict] = {}
    for tid, tr in reads["tracks"].items():
        tracks_out[tid] = {
            "team": tr["team"], "number": tr["number"], "conf": tr["conf"],
            "votes": tr.get("counts", {}), "reads": tr["reads"],
        }
    players = group_players(reads["tracks"])
    key_of = {t: p["key"] for p in players for t in p["track_ids"]}
    for tid in tracks_out:
        tracks_out[tid]["key"] = key_of[int(tid)]
    n_tracks = len(tracks_out)
    n_num = sum(1 for t in tracks_out.values() if t["number"] is not None)
    n_players = sum(1 for p in players if p["number"] is not None)
    out = {
        "clip": reads["clip"],
        "tracks": tracks_out,
        "players": players,
        "summary": {"tracks": n_tracks, "tracks_with_number": n_num, "players_with_number": n_players,
                    "share": round(n_num / max(1, n_tracks), 3)},
    }
    tmp = out_path.with_suffix(".tmp")
    json.dump(out, open(tmp, "w"), indent=1)
    tmp.replace(out_path)
    log.info("identities: %d/%d tracks with number (%.0f%%) -> %d numbered players, %d unnamed -> %s",
             n_num, n_tracks, 100 * n_num / max(1, n_tracks), n_players, len(players) - n_players, out_path)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", type=Path, default=READS)
    ap.add_argument("--out", type=Path, default=IDENTITIES)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    run(a.reads, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
