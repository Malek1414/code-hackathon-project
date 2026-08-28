import json

from vision.stats.build import build, load_cuts, load_identities, main
from vision.stats.court import distances_m, on_court
from vision.stats.io import Player
from vision.stats.io import synthetic_scenario, write_tracks

CONTRACT_SHOT_KEYS = {"t", "frame", "player_id", "team", "made", "shooter_foot", "hoop_bbox"}
CONTRACT_PLAYER_KEYS = {"id", "team", "fga", "fgm", "fg_pct", "possession_s", "distance_m", "key", "number"}


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


def test_meta_fps_prefers_source_rate(tmp_path):
    from vision.stats.build import meta_fps

    (tmp_path / "tracks_meta.json").write_text(json.dumps({"source_fps": 50.0, "stride": 2, "fps": 25.0}))
    assert meta_fps(tmp_path / "tracks.jsonl") == 50.0
    assert meta_fps(tmp_path / "elsewhere" / "tracks.jsonl") is None


def test_identities_merge_tracks_into_players(tmp_path):
    frames = synthetic_scenario("made")
    for fr in frames:  # the shooter (id 2) is re-identified as id 20 after 3.0 s
        fr.players = [type(p)(id=20 if (p.id == 2 and fr.t >= 3.0) else p.id, bbox=p.bbox, foot=p.foot, team=p.team)
                      for p in fr.players]
    ident_file = tmp_path / "identities.json"
    ident_file.write_text(json.dumps({
        "clip": "x",
        "tracks": {"3": {"team": 1, "number": 7, "conf": 0.9}},
        "players": [{"key": "A12", "team": 0, "number": 12, "track_ids": [2, 20]}],
    }))
    identities = load_identities(ident_file)
    assert identities[3].key == "B7" and identities[20].key == "A12"
    events, stats = build(frames, fps=50, clip="x", identities=identities)
    assert events["shots"][0]["player_key"] == "A12"
    row = next(p for p in stats["players"] if p["key"] == "A12")
    assert row["track_ids"] == [2, 20] and row["number"] == 12 and row["fga"] == 1 and row["id"] == 2
    assert next(p for p in stats["players"] if p["key"] == "B7")["number"] == 7
    assert next(p for p in stats["players"] if p["id"] == 1)["key"] == "A?1"


def test_cut_resets_possession_and_shot_state():
    """A cut while the ball is in the air: the shooter's possession before the
    cut must not be carried across (shot stays, shooter unknown), and no
    possession segment spans the cut."""
    frames = synthetic_scenario("made")
    cut_frame = 180  # 3.6 s: ball is in the air above the rim
    events, stats = build(frames, fps=50, clip="x", cuts=[cut_frame])
    assert events["cuts"] == [cut_frame]
    assert len(events["shots"]) == 1
    shot = events["shots"][0]
    assert shot["made"] is True and shot["player_id"] is None and shot["shooter_confirmed"] is False
    assert all(not (s["start_frame"] < cut_frame <= s["end_frame"]) for s in events["possessions"])
    # a cut earlier, while the ball is still in the shooter's hands, keeps the attribution
    events2, _ = build(synthetic_scenario("made"), fps=50, clip="x", cuts=[100])
    assert events2["shots"][0]["player_id"] == 2
    # without the cut the same frames give the made shot
    assert build(synthetic_scenario("made"), fps=50, clip="x")[0]["shots"][0]["made"] is True


def test_load_cuts_formats(tmp_path):
    (tmp_path / "a.json").write_text("[5, 900, 5]")
    (tmp_path / "b.json").write_text(json.dumps({"clip": "x", "cuts": [{"frame": 700, "t": 14.0}, 2800]}))
    (tmp_path / "c.txt").write_text("# cuts\n650 close-up ends\n2800\n\n")
    assert load_cuts(tmp_path / "a.json") == [5, 900]
    assert load_cuts(tmp_path / "b.json") == [700, 2800]
    assert load_cuts(tmp_path / "c.txt") == [650, 2800]


def test_on_court_filter_heuristic_drops_small_high_tracks():
    """A seated spectator: small box near the top of the image, never on court."""
    frames = synthetic_scenario("made")
    for fr in frames:
        fr.players.append(Player(id=99, bbox=(1500.0, 200.0, 1540.0, 280.0), foot=(1520.0, 280.0), team=0))
    events, stats = build(frames, fps=50, clip="x")
    assert events["off_court_track_ids"] == [99]
    assert all(p["id"] != 99 for p in stats["players"])
    assert events["shots"][0]["player_id"] == 2  # real players untouched
    events2, stats2 = build(synthetic_scenario("made"), fps=50, clip="x")
    assert events2["off_court_track_ids"] == [] and len(stats2["players"]) == 3


def test_on_court_filter_with_calibration():
    calib = {"H_px_to_m": [[0.015, 0, 0], [0, 0.015, 0], [0, 0, 1]], "court_m": {"length": 28.0, "width": 15.0}}
    assert on_court(calib, 0, (700.0, 400.0)) is True  # 10.5 m, 6 m
    assert on_court(calib, 0, (2000.0, 400.0)) is False  # 30 m: outside + margin
    assert on_court({"frames": {}}, 0, (1.0, 1.0)) is None
    frames = synthetic_scenario("made")
    for fr in frames:  # a big box outside the court must go too (x = 29.9 m)
        fr.players.append(Player(id=77, bbox=(1950.0, 500.0, 2030.0, 700.0), foot=(1990.0, 700.0), team=1))
    events, stats = build(frames, fps=50, clip="x", calib=calib)
    assert events["off_court_track_ids"] == [77]
    assert {p["id"] for p in stats["players"]} == {1, 2, 3}
