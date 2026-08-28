from vision.stats.io import Ball, Frame, Player, synthetic_scenario
from vision.stats.possession import PossessionParams, nearest_player, track_possession


def _frame(n, ball, players):
    return Frame(frame=n, t=n / 50, players=players, ball=Ball(center=ball) if ball else None)


P1 = Player(id=1, bbox=(360, 600, 440, 800), foot=(400, 800), team=0)
P2 = Player(id=2, bbox=(860, 620, 940, 820), foot=(900, 820), team=1)


def test_nearest_player_needs_ball_within_1_5_heights():
    assert nearest_player(_frame(0, (430, 700), [P1, P2])).id == 1
    assert nearest_player(_frame(0, (880, 720), [P1, P2])).id == 2
    assert nearest_player(_frame(0, (400, 700 + 1.4 * 200), [P1])) is not None
    assert nearest_player(_frame(0, (400, 700 + 1.6 * 200), [P1])) is None
    assert nearest_player(_frame(0, None, [P1])) is None


def test_pass_scenario_switches_once_and_is_backdated():
    frames = synthetic_scenario("pass")
    res = track_possession(frames)
    ids = [s.player_id for s in res.segments]
    assert ids == [1, 2]
    a, b = res.segments
    assert a.start_t == 0.0 and 1.1 < a.end_t < 1.35  # ball leaves player 1 mid-pass
    assert 1.2 < b.start_t < 1.35 and b.end_frame == frames[-1].frame
    assert a.team == 0 and b.team == 0


def test_flicker_between_two_players_does_not_switch():
    frames = [_frame(n, (430, 700), [P1, P2]) for n in range(30)]
    for n in range(30, 130):
        ball = (430, 700) if n % 2 == 0 else (870, 720)  # alternates every frame
        frames.append(_frame(n, ball, [P1, P2]))
    res = track_possession(frames, PossessionParams(min_frames=10))
    assert [(s.player_id, s.start_frame, s.end_frame) for s in res.segments] == [(1, 0, 129)]


def test_flicker_from_the_start_gives_nobody_possession():
    frames = [_frame(n, (430, 700) if n % 2 == 0 else (870, 720), [P1, P2]) for n in range(100)]
    assert track_possession(frames).segments == []


def test_short_glitch_is_ignored_but_real_change_counts():
    frames = [_frame(n, (430, 700), [P1, P2]) for n in range(50)]
    frames += [_frame(n, (870, 720), [P1, P2]) for n in range(50, 55)]  # 5-frame glitch
    frames += [_frame(n, (430, 700), [P1, P2]) for n in range(55, 100)]
    frames += [_frame(n, (870, 720), [P1, P2]) for n in range(100, 150)]  # real change
    res = track_possession(frames)
    assert [(s.player_id, s.start_frame) for s in res.segments] == [(1, 0), (2, 100)]


def test_missing_ball_frames_carry_state():
    frames = [_frame(n, (430, 700), [P1, P2]) for n in range(20)]
    frames += [_frame(n, None, [P1, P2]) for n in range(20, 60)]
    frames += [_frame(n, (430, 700), [P1, P2]) for n in range(60, 80)]
    res = track_possession(frames)
    assert len(res.segments) == 1 and res.holder[40] == 1


def test_loose_ball_ends_possession():
    frames = [_frame(n, (430, 700), [P1, P2]) for n in range(20)]
    frames += [_frame(n, (650, 100), [P1, P2]) for n in range(20, 60)]  # far from everyone
    res = track_possession(frames)
    assert len(res.segments) == 1
    assert res.segments[0].end_frame == 19 and res.holder[30] is None
