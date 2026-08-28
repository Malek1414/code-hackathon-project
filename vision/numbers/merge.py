"""Merge track ids into real players (NUMBERS role) -> out/identities.json.

Input: out/numbers_reads.json (from read.py) and out/tracks.jsonl.
Players = groups of track ids with the same (team, number) whose lifetimes
overlap by at most MAX_OVERLAP_S (ORCH rule: two boxes with the same number
are two people only if both are on court at once for > 1 s; a shorter overlap
is a hand-over between ids or a misread and merges. The 0.5 s allowance after
a cut is covered by the same tolerance). A longer overlap becomes a second
player with key `A12~<id>` so the conflict stays visible. Tracks
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
MAX_OVERLAP_S = 1.0  # same (team, number) ids may overlap this long and still be one player


def overlap_s(a: dict, b: dict) -> float:
    return min(a["last_t"], b["last_t"]) - max(a["first_t"], b["first_t"])


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
                if all(overlap_s(tr, o) <= MAX_OVERLAP_S for _, o in g):
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
    # ORCH rule: the same number on both teams with a lopsided vote is suspect (TRACK's
    # color rule reads black sleeves on a blue shirt as team B); we keep the player but
    # flag it with the team share of its tracks so consumers can show it
    mass_of = {p["key"]: sum(sum(tracks[str(t)].get("votes", {}).values()) for t in p["track_ids"]) for p in players}
    for p in players:
        rival = next((q for q in players if q is not p and q["number"] == p["number"] and q["team"] != p["team"]), None)
        if rival and mass_of[rival["key"]] >= 3 * mass_of[p["key"]]:
            p["suspect_team"] = True
            p["team_share"] = round(min(tracks[str(t)].get("team_share", 1.0) for t in p["track_ids"]), 3)
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
            "team": tr["team"], "team_share": tr.get("team_share", 1.0), "number": tr["number"], "conf": tr["conf"],
            "votes": tr.get("counts", {}), "reads": tr["reads"],
        }
        if tr.get("switch_t") is not None:
            tracks_out[tid]["switch_t"] = tr["switch_t"]  # id jumped to another player here; number/team are the later one
    players = group_players(reads["tracks"])
    key_of = {t: p["key"] for p in players for t in p["track_ids"]}
    for tid in tracks_out:
        tracks_out[tid]["key"] = key_of[int(tid)]
    n_tracks = len(tracks_out)
    n_num = sum(1 for t in tracks_out.values() if t["number"] is not None)
    n_players = sum(1 for p in players if p["number"] is not None)
    out = {
        "clip": reads["clip"],
        # which TRACK run the ids belong to (STATS checks this before trusting the mapping)
        "tracks_path": reads.get("tracks_path"), "tracks_mtime": reads.get("tracks_mtime"),
        "tracks_frames": reads.get("tracks_frames"), "tracks_ids": reads.get("tracks_ids"),
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
