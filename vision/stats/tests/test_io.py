import json

from vision.stats.io import Ball, Frame, clean_ball, frame_from_dict, infer_fps, read_tracks, synthetic_scenario, write_tracks


def test_roundtrip(tmp_path):
    frames = synthetic_scenario("made", duration_s=1.0)
    path = tmp_path / "tracks.jsonl"
    write_tracks(frames, path)
    back = read_tracks(path)
    assert len(back) == len(frames)
    assert back[10].frame == 10 and abs(back[10].t - 0.2) < 1e-6
    assert [p.id for p in back[10].players] == [1, 2, 3]
    assert back[10].ball is not None and back[10].ball.center == frames[10].ball.center
    assert back[10].hoops == [tuple(round(v, 1) for v in frames[10].hoops[0])]


def test_truncated_last_line_is_skipped(tmp_path):
    frames = synthetic_scenario("made", duration_s=0.5)
    path = tmp_path / "tracks.jsonl"
    write_tracks(frames, path)
    with path.open("a") as f:
        f.write('{"frame": 999, "t": 19.98, "players": [{"id": 1, "bb')
    back = read_tracks(path)
    assert len(back) == len(frames)


def test_contract_example_and_defaults():
    d = {
        "frame": 1250,
        "t": 25.0,
        "players": [{"id": 7, "bbox": [10, 20, 50, 120], "foot": [30, 120], "team": 0, "conf": 0.91}],
        "ball": {"bbox": [100, 100, 110, 110], "center": [105, 105], "conf": 0.6},
        "hoops": [{"bbox": [500, 200, 560, 240]}],
    }
    fr = frame_from_dict(d)
    assert fr.players[0].foot == (30.0, 120.0) and fr.players[0].height == 100
    assert fr.ball.center == (105.0, 105.0)
    assert fr.hoops == [(500.0, 200.0, 560.0, 240.0)]

    # tolerant reader: no foot, no t, null ball, no hoops
    d2 = {"frame": 100, "players": [{"id": 1, "bbox": [0, 0, 40, 100]}], "ball": None}
    fr2 = frame_from_dict(d2, fps=50)
    assert fr2.t == 2.0 and fr2.ball is None and fr2.hoops == []
    assert fr2.players[0].foot == (20.0, 100.0) and fr2.players[0].team == -1


def test_infer_fps_on_subsampled_tracks():
    frames = [Frame(frame=n * 2, t=n * 2 / 50) for n in range(20)]
    assert abs(infer_fps(frames) - 50.0) < 1e-6


def test_only_the_largest_hoop_is_kept():
    d = {"frame": 1, "t": 0.02, "players": [], "ball": None,
         "hoops": [{"bbox": [10, 10, 20, 20]}, {"bbox": [500, 100, 560, 140]}]}
    assert frame_from_dict(d).hoops == [(500.0, 100.0, 560.0, 140.0)]


def test_clean_ball_drops_strays_but_keeps_confirmed_jumps():
    frames = [Frame(frame=n, t=n / 10, ball=Ball(center=(100 + 5 * n, 300), bbox=(0, 0, 20, 20))) for n in range(20)]
    frames[5].ball = Ball(center=(1500, 50), bbox=(0, 0, 20, 20))  # exit sign for one frame
    frames[9].ball = Ball(center=(300, 300), bbox=(0, 0, 40, 12))  # not round
    dropped = clean_ball(frames)
    assert dropped == 2 and frames[5].ball is None and frames[9].ball is None
    assert sum(1 for f in frames if f.ball) == 18

    # a jump confirmed by the next detection is real (ball re-appears elsewhere)
    frames = [Frame(frame=n, t=n / 10, ball=Ball(center=(100, 300), bbox=(0, 0, 20, 20))) for n in range(5)]
    frames += [Frame(frame=n, t=n / 10, ball=Ball(center=(1500, 300), bbox=(0, 0, 20, 20))) for n in range(5, 10)]
    assert clean_ball(frames) == 0
