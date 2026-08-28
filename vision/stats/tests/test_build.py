import json

from vision.stats.build import build, distances_m, main
from vision.stats.io import synthetic_scenario, write_tracks

CONTRACT_SHOT_KEYS = {"t", "frame", "player_id", "team", "made", "shooter_foot", "hoop_bbox"}
CONTRACT_PLAYER_KEYS = {"id", "team", "fga", "fgm", "fg_pct", "possession_s", "distance_m"}


def test_build_matches_contract():
    frames = synthetic_scenario("made")
    events, stats = build(frames, fps=50, clip="data/clips/game10.mp4")
    assert events["fps"] == 50 and events["clip"] == "data/clips/game10.mp4"
    assert len(events["shots"]) == 1 and CONTRACT_SHOT_KEYS <= set(events["shots"][0])
    assert set(stats) == {"players", "teams"}
    for p in stats["players"]:
        assert CONTRACT_PLAYER_KEYS <= set(p) and p["distance_m"] is None
    shooter = next(p for p in stats["players"] if p["id"] == 2)
    assert (shooter["fga"], shooter["fgm"], shooter["fg_pct"]) == (1, 1, 1.0)
    assert shooter["possession_s"] > 1.0
    assert {(t["team"], t["fga"], t["fgm"]) for t in stats["teams"]} == {(0, 1, 1), (1, 0, 0)}


def test_miss_and_shooter_unknown_go_to_team_minus_one():
    frames = synthetic_scenario("miss")
    for fr in frames:
        fr.players = []
    events, stats = build(frames, fps=50, clip="x")
    assert events["shots"][0]["made"] is False and events["shots"][0]["player_id"] is None
    assert stats["players"] == []
    assert {(t["team"], t["fga"]) for t in stats["teams"]} == {(0, 0), (1, 0), (-1, 1)}


def test_distance_with_identity_homography():
    frames = synthetic_scenario("pass", duration_s=1.0)
    for n, fr in enumerate(frames):  # player 1 walks 1 px/frame to the right
        p = fr.players[0]
        fr.players[0] = type(p)(id=p.id, bbox=p.bbox, foot=(p.foot[0] + n, p.foot[1]), team=p.team)
    calib = {"H_px_to_m": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]]}  # 1 px = 1 cm
    d = distances_m(frames, calib)
    assert abs(d[1] - 0.49) < 1e-6 and d[2] == 0.0


def test_cli_writes_both_files(tmp_path):
    tracks = tmp_path / "tracks.jsonl"
    write_tracks(synthetic_scenario("made"), tracks)
    rc = main(["--tracks", str(tracks), "--clip", "c.mp4", "--out-dir", str(tmp_path / "out"),
               "--calib", str(tmp_path / "none.json")])
    assert rc == 0
    events = json.loads((tmp_path / "out" / "events.json").read_text())
    stats = json.loads((tmp_path / "out" / "stats.json").read_text())
    assert events["fps"] == 50 and len(events["shots"]) == 1
    assert stats["players"][0]["id"] == 2
