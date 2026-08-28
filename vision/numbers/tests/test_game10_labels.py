"""Integration check of the game10 identities against Sami's confirmed labels.

Shots are matched by time through out/events.json (STATS' shooter track), cards by
frame + box in the tracks file, so both survive TRACK re-runs. Skipped when out/
does not hold a game10 run. A label whose track has no number yet is reported as
open (xfail), a wrong number fails.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LABELS = json.load(open(Path(__file__).parent / "fixtures" / "game10_labels.json"))


def _identities():
    path = ROOT / "out" / "identities.json"
    if not path.exists():
        pytest.skip("no out/identities.json")
    d = json.load(open(path))
    if d.get("clip") != LABELS["clip"]:
        pytest.skip(f"identities are for {d.get('clip')}")
    tracks = ROOT / d.get("tracks_path", "out/tracks.jsonl")
    if not tracks.exists() or round(tracks.stat().st_mtime, 3) != d.get("tracks_mtime"):
        pytest.skip("tracks file is not the run identities.json was built from")
    return d, tracks


def _check(d, tid, label):
    tr = d["tracks"].get(str(tid))
    assert tr is not None, f"track {tid} missing from identities"
    if tr["number"] is None:
        pytest.xfail(f"track {tid} has no number yet (want {label['team']}/{label['number']})")
    assert (tr["team"], tr["number"]) == (label["team"], label["number"]), f"track {tid}: {tr}"


@pytest.mark.parametrize("label", LABELS["shots"], ids=lambda l: l["label"])
def test_game10_shot_shooter(label):
    d, _ = _identities()
    events = ROOT / "out" / "events.json"
    if not events.exists():
        pytest.skip("no out/events.json")
    e = json.load(open(events))
    if e.get("clip") != LABELS["clip"] or not e.get("shots"):
        pytest.skip("events.json is not game10")
    if label.get("disputed"):
        pytest.skip(label["disputed"])
    shot = min(e["shots"], key=lambda s: abs(s["t"] - label["t"]))
    if abs(shot["t"] - label["t"]) > 0.5:
        pytest.skip(f"no event at {label['t']} s")
    _check(d, shot["player_id"], label)


@pytest.mark.parametrize("label", LABELS["players"], ids=lambda l: l["label"])
def test_game10_card(label):
    d, tracks = _identities()
    best, dist = None, 1e9
    with open(tracks) as fh:
        for line in fh:
            x = json.loads(line)
            if x["frame"] != label["frame"]:
                continue
            for p in x["players"]:
                dd = abs(p["bbox"][0] - label["box"][0]) + abs(p["bbox"][1] - label["box"][1])
                if dd < dist:
                    best, dist = p["id"], dd
            break
    assert best is not None and dist < 120, "no track at the labelled box"
    _check(d, best, label)
