"""Integration check of out/identities.json against Sami's confirmed dev60 labels.

Skipped when out/ does not hold a dev60 run. Labels are anchored by frame + box so
they survive TRACK re-runs that renumber the ids.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LABELS = json.load(open(Path(__file__).parent / "fixtures" / "dev60_labels.json"))


def _identities():
    path = ROOT / "out" / "identities.json"
    if not path.exists():
        pytest.skip("no out/identities.json")
    d = json.load(open(path))
    if d.get("clip") != LABELS["clip"]:
        pytest.skip(f"identities are for {d.get('clip')}, labels for {LABELS['clip']}")
    tracks = ROOT / d.get("tracks_path", "out/tracks.jsonl")
    if not tracks.exists() or round(tracks.stat().st_mtime, 3) != d.get("tracks_mtime"):
        pytest.skip("out/tracks.jsonl is not the run identities.json was built from")
    return d


def _id_at(frame: int, box: list[int]) -> int | None:
    d = json.load(open(ROOT / "out" / "identities.json"))
    path = ROOT / d.get("tracks_path", "out/tracks.jsonl")
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d["frame"] != frame:
                continue
            best, dist = None, 1e9
            for p in d["players"]:
                dd = abs(p["bbox"][0] - box[0]) + abs(p["bbox"][1] - box[1])
                if dd < dist:
                    best, dist = p["id"], dd
            return best if dist < 120 else None
    return None


@pytest.mark.parametrize("label", LABELS["players"], ids=lambda l: l["label"][:40])
def test_dev60_label(label):
    d = _identities()
    if "key" in label:
        p = next((p for p in d["players"] if p["key"] == label["key"]), None)
        assert p is not None and p["number"] == label["number"] and p["team"] == label["team"]
        return
    tid = _id_at(label["frame"], label["box"])
    assert tid is not None, "no track at the labelled box"
    tr = d["tracks"][str(tid)]
    if "not_number" in label:
        assert tr["number"] != label["not_number"]
    if label.get("may_be_open") and tr["number"] is None:
        return
    assert (tr["team"], tr["number"]) == (label["team"], label["number"]), f"track {tid}: {tr}"
