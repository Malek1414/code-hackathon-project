import pytest

from vision.stats.io import FIXTURE_HOOP, FIXTURE_SHOOTER_ID, Ball, Frame, synthetic_scenario
from vision.stats.possession import track_possession
from vision.stats.shots import ShotParams, detect_shots, in_zone


def _run(kind, **kw):
    frames = synthetic_scenario(kind, **kw)
    return frames, detect_shots(frames, track_possession(frames))


def test_made_shot():
    frames, shots = _run("made")
    assert len(shots) == 1
    s = shots[0]
    assert s.made is True
    assert s.player_id == FIXTURE_SHOOTER_ID and s.team == 0 and s.shooter_confirmed
    assert 3.5 < s.t < 3.8
    assert s.shooter_foot == (900.0, 820.0)
    assert 2.5 <= frames[s.release_frame].t <= 2.8  # ball left the hands right after the shot started
    assert s.hoop_bbox == FIXTURE_HOOP


def test_missed_shot():
    _, shots = _run("miss")
    assert len(shots) == 1 and shots[0].made is False and shots[0].player_id == FIXTURE_SHOOTER_ID


def test_pass_only_has_no_shot():
    _, shots = _run("pass")
    assert shots == []


@pytest.mark.parametrize("kind,made", [("made", True), ("miss", False)])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_robust_to_dropouts_and_jitter(kind, made, seed):
    _, shots = _run(kind, ball_dropout=0.3, hoop_dropout=0.3, jitter_px=3.0, seed=seed)
    assert [(s.player_id, s.made) for s in shots] == [(FIXTURE_SHOOTER_ID, made)]


def test_no_hoop_in_frame_means_no_shot():
    frames = synthetic_scenario("made")
    for fr in frames:
        fr.hoops = []
    assert detect_shots(frames, track_possession(frames)) == []


def test_panning_camera_shifts_everything_together():
    """Hoop-relative logic: a pan moves ball, players and hoop by the same offset."""
    frames = synthetic_scenario("made")
    for n, fr in enumerate(frames):
        dx = 3.0 * n  # 150 px/s pan
        fr.players = [
            type(p)(id=p.id, bbox=(p.bbox[0] + dx, p.bbox[1], p.bbox[2] + dx, p.bbox[3]),
                    foot=(p.foot[0] + dx, p.foot[1]), team=p.team, conf=p.conf)
            for p in fr.players
        ]
        if fr.ball:
            fr.ball = Ball(center=(fr.ball.center[0] + dx, fr.ball.center[1]), conf=fr.ball.conf)
        fr.hoops = [(h[0] + dx, h[1], h[2] + dx, h[3]) for h in fr.hoops]
    shots = detect_shots(frames, track_possession(frames))
    assert [(s.player_id, s.made) for s in shots] == [(FIXTURE_SHOOTER_ID, True)]


def test_rim_rattle_counts_once():
    """Ball leaves and re-enters the zone within the cooldown: still one attempt."""
    frames = synthetic_scenario("miss")
    # after the rim bounce, push the ball back into the zone from above once more
    for fr in frames:
        if 3.95 <= fr.t < 4.1 and fr.ball:
            fr.ball = Ball(center=(1500.0, 250.0 + (fr.t - 3.95) * 300), conf=0.6)
    shots = detect_shots(frames, track_possession(frames))
    assert len(shots) == 1


def test_zone_geometry():
    p = ShotParams()
    hoop = (100.0, 100.0, 160.0, 140.0)  # w 60, h 40, rim y 100
    assert in_zone((130, 50), hoop, p)  # above the rim, inside 1.5 h
    assert not in_zone((130, 30), hoop, p)  # too high
    assert in_zone((72, 120), hoop, p) and not in_zone((68, 120), hoop, p)  # 2x width
    assert not in_zone((130, 150), hoop, p)  # below the bbox


def test_unknown_shooter_is_flagged():
    frames = synthetic_scenario("made")
    for fr in frames:
        fr.players = []  # nobody tracked
    shots = detect_shots(frames, track_possession(frames))
    assert len(shots) == 1
    assert shots[0].player_id is None and shots[0].team == -1 and not shots[0].shooter_confirmed
