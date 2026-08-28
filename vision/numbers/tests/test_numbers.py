from vision.numbers.merge import group_players
import numpy as np

from vision.numbers.read import pick_samples, pixel_team, red_channel, vote


def test_vote_assigns_clear_winner():
    n, share, mass, count = vote([("23", 0.9), ("23", 0.8), ("7", 0.5)])
    assert n == 23 and share > 0.7 and count["23"] == 2


def test_vote_single_strong_read_counts_only_without_competitor():
    assert vote([("23", 0.99)])[0] == 23
    assert vote([("23", 0.85)])[0] is None
    assert vote([("23", 0.99), ("7", 0.45)])[0] is None


def test_vote_needs_two_reads_and_majority():
    assert vote([("23", 0.6), ("23", 0.5), ("7", 0.6), ("7", 0.6)])[0] is None  # < 60 %
    assert vote([])[0] is None


def test_vote_substring_read_supports_two_digit_number():
    n, share, mass, count = vote([("55", 0.9), ("5", 0.8)])
    assert n == 55 and share == 1.0 and mass == {"55": 1.3} and count == {"55": 2}
    assert vote([("23", 0.6), ("2", 0.9)])[0] == 23


def test_vote_strips_leading_zero():
    assert vote([("07", 0.6), ("7", 0.6)])[0] == 7


def test_pick_samples_spread_and_tallest():
    rows = [{"frame": i, "t": i / 25, "team": 0, "bbox": [0, 0, 50, 100 + (i % 7) * 10]} for i in range(120)]
    s = pick_samples(rows, 12)
    assert len(s) == 12
    assert s[0]["frame"] < 10 and s[-1]["frame"] > 110  # spread over the lifetime
    assert all(r["bbox"][3] == 160 for r in s)  # tallest of each bin


def _tr(team, number, a, b, conf=0.9, votes=None, team_share=1.0):
    return {"team": team, "number": number, "conf": conf, "first_t": a, "last_t": b, "frames": int((b - a) * 25) + 1,
            "votes": votes or ({str(number): 3.0} if number else {}), "team_share": team_share}


def test_group_players_merges_non_overlapping_same_number():
    tracks = {"7": _tr(0, 12, 0.0, 10.0), "41": _tr(0, 12, 9.2, 20.0), "88": _tr(0, 12, 15.0, 30.0),
              "9": _tr(1, 12, 0.0, 30.0), "5": _tr(0, None, 3.0, 4.0), "6": _tr(-1, None, 3.0, 4.0)}
    players = {p["key"]: p for p in group_players(tracks)}
    assert players["A12"]["track_ids"] == [7, 41]  # 0.8 s overlap = id hand-over, merged
    assert players["A12~88"]["track_ids"] == [88]  # 5 s overlap with 41: second person, visible
    assert players["B12"]["track_ids"] == [9]
    assert players["A?5"]["number"] is None and players["X?6"]["team"] == -1
    assert players["A12"]["first_t"] == 0.0 and players["A12"]["last_t"] == 20.0
    assert "suspect_team" not in players["B12"]  # A12 mass 6.0 vs 3.0 is under the 3x rule


def test_same_number_on_both_teams_is_flagged_when_lopsided():
    tracks = {"1": _tr(0, 55, 0.0, 10.0, votes={"55": 9.0}), "2": _tr(1, 55, 12.0, 14.0, votes={"55": 1.0}, team_share=0.85)}
    players = {p["key"]: p for p in group_players(tracks)}
    assert players["B55"]["suspect_team"] is True and players["B55"]["team_share"] == 0.85
    assert "suspect_team" not in players["A55"]


def _crop(t, reads, team):
    return ("k", {"t": t, "frame": int(t * 50), "bbox": [0, 0, 10, 100]}, reads, team)


def test_fixture_611_id_switch_black9_to_blue55():
    # ground truth (Sami's screenshots, dev60): 611 is black #9 until ~44 s, then the id jumps to blue #55
    crops = [_crop(40.4, [], -1), _crop(44.4, [], 1), _crop(48.1, [("55", 0.999)], 0),
             _crop(50.6, [("55", 0.999), ("88", 0.43)], 0), _crop(53.1, [], 0), _crop(55.7, [("55", 0.999)], 0)]
    team, switch_t = pixel_team(crops)
    assert (team, switch_t) == (0, 48.1)
    crops[1] = _crop(44.4, [("9", 0.99)], 1)  # if the 9 had been read it must not pollute the blue segment
    reads = [x for _, r, rd, ct in crops for x in rd if ct == team]
    assert vote(reads)[0] == 55


def test_fixture_616_neighbour_read_loses_the_vote():
    # blue #55; the 9:1.00 is the black #9's back inside the crop at frame 2128 (team of the crop still blue)
    reads = [("9", 0.9997), ("55", 0.667), ("5", 0.959), ("55", 0.827)]
    assert vote(reads)[0] == 55


def test_fixture_599_single_weak_read_stays_open():
    assert vote([("23", 0.623), ("0", 0.429)])[0] is None


def test_pixel_team_single_stray_crop_is_not_a_switch():
    crops = [_crop(1, [], 0), _crop(2, [], 0), _crop(3, [], 0), _crop(4, [], 1)]
    assert pixel_team(crops) == (0, None)
    assert pixel_team([_crop(1, [], -1)]) == (-1, None)


def test_red_channel_lifts_red_on_black():
    img = np.zeros((20, 20, 3), np.uint8)  # black jersey
    img[5:15, 5:15] = (30, 30, 200)  # red number (BGR)
    out = red_channel(img)
    assert out.shape == (20, 20, 3) and out[10, 10, 0] == 255 and out[0, 0, 0] == 0


def test_eval_scores_cards_by_key_then_track_ids():
    from vision.numbers.eval import score

    identities = {"players": [{"key": "A55", "team": 0, "number": 55, "track_ids": [611, 701]},
                              {"key": "B9", "team": 1, "number": 9, "track_ids": [805]},
                              {"key": "A?7", "team": 0, "number": None, "track_ids": [7]}],
                  "tracks": {"805": {"team": 1, "number": 9, "key": "B9"}}}
    verdicts = {"numbers": [{"key": "A55", "track_ids": [611], "detected": 55, "true_number": None, "unreadable": False},
                            {"key": "B?805", "track_ids": [805], "detected": None, "true_number": 9, "unreadable": False},
                            {"key": "A?7", "track_ids": [7], "detected": None, "true_number": None, "unreadable": True}],
                "shots": [{"n": 1, "t": 56.96, "player_id": 805, "number": 9, "number_team": 1}]}
    lines, right, wrong, open_ = score(identities, verdicts)
    assert (right, wrong, open_) == (3, 0, 0) and len(lines) == 4


def test_group_players_uses_switch_t_as_start_after_an_id_jump():
    # dev60 best.pt ids: 257 (31.3-34.7) and 284 (33.6-38.2, id jumped at 35.6) both read B12
    a = _tr(1, 12, 31.28, 34.68)
    b = dict(_tr(1, 12, 33.6, 38.2), switch_t=35.6)
    players = {p["key"]: p for p in group_players({"257": a, "284": b})}
    assert players["B12"]["track_ids"] == [257, 284] and "B12~284" not in players
