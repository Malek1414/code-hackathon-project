from vision.numbers.merge import group_players
from vision.numbers.read import pick_samples, vote


def test_vote_assigns_clear_winner():
    n, share, mass, count = vote([("23", 0.9), ("23", 0.8), ("2", 0.5)])
    assert n == 23 and share > 0.7 and count["23"] == 2


def test_vote_needs_two_reads_and_majority():
    assert vote([("23", 0.99)])[0] is None  # one read only
    assert vote([("23", 0.6), ("23", 0.5), ("7", 0.6), ("7", 0.6)])[0] is None  # < 60 %
    assert vote([])[0] is None


def test_vote_strips_leading_zero():
    assert vote([("07", 0.6), ("7", 0.6)])[0] == 7


def test_pick_samples_spread_and_tallest():
    rows = [{"frame": i, "t": i / 25, "team": 0, "bbox": [0, 0, 50, 100 + (i % 7) * 10]} for i in range(120)]
    s = pick_samples(rows, 12)
    assert len(s) == 12
    assert s[0]["frame"] < 10 and s[-1]["frame"] > 110  # spread over the lifetime
    assert all(r["bbox"][3] == 160 for r in s)  # tallest of each bin


def _tr(team, number, a, b, conf=0.9):
    return {"team": team, "number": number, "conf": conf, "first_t": a, "last_t": b, "frames": int((b - a) * 25) + 1}


def test_group_players_merges_non_overlapping_same_number():
    tracks = {"7": _tr(0, 12, 0.0, 10.0), "41": _tr(0, 12, 12.0, 20.0), "88": _tr(0, 12, 15.0, 30.0),
              "9": _tr(1, 12, 0.0, 30.0), "5": _tr(0, None, 3.0, 4.0), "6": _tr(-1, None, 3.0, 4.0)}
    players = {p["key"]: p for p in group_players(tracks)}
    assert players["A12"]["track_ids"] == [7, 41]
    assert players["A12~88"]["track_ids"] == [88]  # overlaps 41, stays separate but visible
    assert players["B12"]["track_ids"] == [9]
    assert players["A?5"]["number"] is None and players["X?6"]["team"] == -1
    assert players["A12"]["first_t"] == 0.0 and players["A12"]["last_t"] == 20.0
