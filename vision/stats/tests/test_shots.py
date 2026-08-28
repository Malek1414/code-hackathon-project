import pytest

from vision.stats.io import FIXTURE_HOOP, FIXTURE_SHOOTER_ID, Ball, Frame, synthetic_scenario
from vision.stats.possession import track_possession
from vision.stats.shots import ShotParams, crosses_rim, detect_shots, in_up_zone


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
    assert 2.4 <= frames[s.release_frame].t <= 2.8  # release = start of the flight
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


@pytest.mark.parametrize("stride", [2, 5])  # 25 fps and 10 fps tracks
def test_sparse_tracks(stride):
    """TRACK delivers 25 fps (dev60) and 10 fps (game10): windows are in seconds."""
    for kind, made in (("made", True), ("miss", False)):
        hits = 0
        for seed in range(10):
            frames = synthetic_scenario(kind, ball_dropout=0.3, jitter_px=3.0, seed=seed)[::stride]
            shots = detect_shots(frames, track_possession(frames))
            assert all(s.player_id == FIXTURE_SHOOTER_ID and s.made == made for s in shots), kind
            hits += len(shots) == 1
        assert hits >= (8 if stride == 2 else 7), f"{kind} @ stride {stride}: {hits}/10"  # 10 fps + 30 % dropout: the ball is sometimes never seen at the rim
    frames = synthetic_scenario("pass")[::stride]
    assert detect_shots(frames, track_possession(frames)) == []


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
    p = ShotParams(zone_width_scale=2.0, zone_above_scale=1.5)
    hoop = (100.0, 100.0, 160.0, 140.0)  # w 60, h 40, rim y 100
    assert in_up_zone((130, 50), hoop, p)  # above the rim, inside 1.5 h
    assert not in_up_zone((130, 30), hoop, p)  # too high
    assert in_up_zone((72, 80), hoop, p) and not in_up_zone((68, 80), hoop, p)  # 2x width
    assert not in_up_zone((130, 120), hoop, p)  # below the rim line


def test_rim_crossing_is_interpolated():
    p = ShotParams()
    hoop = (0.0, 0.0, 60.0, 40.0)
    assert crosses_rim((20, -20), (40, 20), hoop, p)  # crosses at x 30 = center
    assert not crosses_rim((-40, -20), (0, 20), hoop, p)  # crosses at x -20, in front of the rim
    assert not crosses_rim((30, 20), (30, -20), hoop, p)  # moving up


def test_unknown_shooter_is_flagged():
    frames = synthetic_scenario("made")
    for fr in frames:
        fr.players = []  # nobody tracked
    shots = detect_shots(frames, track_possession(frames))
    assert len(shots) == 1
    assert shots[0].player_id is None and shots[0].team == -1 and not shots[0].shooter_confirmed


def test_ball_vanishing_at_the_rim_counts_as_unconfirmed_attempt():
    """dev60 at 57 s: clean arc into the hoop, then the detector loses the ball."""
    frames = synthetic_scenario("made")
    for fr in frames:
        if fr.t >= 3.66:  # last sample above the rim is at ~3.64 s
            fr.ball = None
    shots = detect_shots(frames, track_possession(frames))
    assert len(shots) == 1
    s = shots[0]
    assert s.made_confirmed is False and s.player_id == FIXTURE_SHOOTER_ID
    assert s.made is False and s.made_hint is True  # counted as miss; the arc aimed at the middle
    assert s.decided_t - s.t <= 1.2  # vanish window + 0.5 s engine lookahead

    # heading for the front iron: extrapolated verdict is a miss
    frames = synthetic_scenario("made")
    for fr in frames:
        if fr.t >= 3.66:
            fr.ball = None
        elif fr.ball is not None and fr.t >= 3.3:
            fr.ball = Ball(center=(fr.ball.center[0] - 40.0, fr.ball.center[1]), conf=0.6)
    shots = detect_shots(frames, track_possession(frames))
    assert len(shots) == 1 and shots[0].made is False and shots[0].made_hint is False

    # ball vanishing far from the rim (lost in the crowd) is not an attempt
    frames = synthetic_scenario("made")
    for fr in frames:
        if fr.t >= 3.3:
            fr.ball = None
    assert detect_shots(frames, track_possession(frames)) == []


def test_shooter_from_release_point_without_possession():
    """Ball never detected in the hands (no possession segment): the flight
    still leads back into the shooter's box."""
    frames = synthetic_scenario("made")
    for fr in frames:
        if fr.t < 2.56:
            fr.ball = None
    shots = detect_shots(frames, track_possession(frames))
    assert len(shots) == 1
    assert shots[0].player_id == FIXTURE_SHOOTER_ID and shots[0].shooter_confirmed is True
    assert shots[0].team == 0 and shots[0].shooter_foot == (900.0, 820.0)


def test_release_point_beats_nearest_foot():
    """A bystander stands right next to the first flight samples; the release
    point is still inside the shooter's box (dev60 free throw, QA finding)."""
    from vision.stats.io import Player

    frames = synthetic_scenario("made")
    for fr in frames:
        if fr.t < 2.56:
            fr.ball = None
        # bystander with feet under the early flight path, box not containing the release point
        fr.players.append(Player(id=9, bbox=(960.0, 560.0, 1040.0, 760.0), foot=(1000.0, 760.0), team=1))
    shots = detect_shots(frames, track_possession(frames))
    assert [(s.player_id, s.shooter_confirmed) for s in shots] == [(FIXTURE_SHOOTER_ID, True)]


def test_release_point_in_nobodys_box_falls_back_unconfirmed():
    frames = synthetic_scenario("made")
    for fr in frames:  # move the shooter's box away from the release point but keep him nearest by foot
        fr.players = [
            type(p)(id=p.id, bbox=(p.bbox[0] - 200, p.bbox[1], p.bbox[2] - 200, p.bbox[3]),
                    foot=(p.foot[0] - 200, p.foot[1]), team=p.team) if p.id == FIXTURE_SHOOTER_ID else p
            for p in fr.players
        ]
    shots = detect_shots(frames, track_possession(frames))
    assert len(shots) == 1 and shots[0].shooter_confirmed is False


def test_shooter_team_never_unknown_when_shooter_is_known():
    from vision.stats.io import Player

    frames = synthetic_scenario("made")
    for fr in frames:  # the shooter's track never gets a colour from TRACK
        fr.players = [type(p)(id=p.id, bbox=p.bbox, foot=p.foot, team=-1 if p.id == FIXTURE_SHOOTER_ID else p.team)
                      for p in fr.players]
        fr.players.append(Player(id=8, bbox=(940.0, 600.0, 1020.0, 800.0), foot=(980.0, 800.0), team=0))
    shots = detect_shots(frames, track_possession(frames))
    assert shots[0].player_id == FIXTURE_SHOOTER_ID and shots[0].team == 0  # from the players around him
    assert shots[0].team_source == "nearby_players"
    assert detect_shots(synthetic_scenario("made"), track_possession(synthetic_scenario("made")))[0].team_source == "track_majority"


def test_free_throw_pass_from_referee_does_not_become_the_release():
    """game10 121 s: the referee bounces the ball to the shooter (rightwards),
    the shot then flies leftwards; the chain must stop at the shooter."""
    from vision.stats.io import Player

    frames = synthetic_scenario("made")
    ref = Player(id=77, bbox=(1150.0, 560.0, 1230.0, 760.0), foot=(1190.0, 760.0), team=-1)
    for fr in frames:
        fr.players.append(ref)
        if fr.t < 2.0:
            fr.ball = None
        elif fr.t < 2.5:  # pass from the referee (right) to the shooter (left), rising slightly
            s = (fr.t - 2.0) / 0.5
            fr.ball = Ball(center=(1190.0 - 310.0 * s, 660.0 - 60.0 * s), conf=0.6)
    shots = detect_shots(frames, track_possession(frames))
    assert [(s.player_id, s.shooter_confirmed) for s in shots] == [(FIXTURE_SHOOTER_ID, True)]
