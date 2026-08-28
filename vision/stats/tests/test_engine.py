from vision.stats.build import build
from vision.stats.engine import StatsEngine
from vision.stats.io import FIXTURE_HOOP, Ball, clean_ball, frame_to_dict, median_dt, synthetic_scenario
from vision.stats.possession import track_possession
from vision.stats.shots import detect_shots


def _batch(frames):
    """build.py's path: the engine over the whole list at once."""
    engine = StatsEngine(dt=median_dt(frames), fps=50)
    for fr in frames:
        engine.push(fr)
    engine.finish()
    return engine.shots, engine.possession.segments


def test_incremental_equals_batch_on_fixture():
    for kind in ("made", "miss", "pass"):
        for seed in (0, 1, 2):
            frames = synthetic_scenario(kind, ball_dropout=0.3 if seed else 0.0, jitter_px=3.0, seed=seed)
            dt = median_dt(frames)
            engine = StatsEngine(dt=dt, fps=50)
            live = []
            for fr in frames:
                live += engine.push(frame_to_dict(fr))  # one tracks line at a time
            live += engine.finish()
            batch_shots, batch_segments = _batch(synthetic_scenario(kind, ball_dropout=0.3 if seed else 0.0, jitter_px=3.0, seed=seed))
            assert [s.to_dict() for s in live] == [s.to_dict() for s in batch_shots], (kind, seed)
            assert [(s.player_id, s.start_frame, s.end_frame) for s in engine.possession.segments] == [
                (s.player_id, s.start_frame, s.end_frame) for s in batch_segments
            ], (kind, seed)


def test_verdict_is_final_within_half_a_second():
    frames = synthetic_scenario("miss")
    engine = StatsEngine(dt=0.02)
    for fr in frames:
        for ev in engine.push(fr):
            assert ev.decided_t - ev.t <= 0.5 + 0.3  # down sighting + made_window_s
    engine.finish()
    assert len(engine.shots) == 1 and engine.shots[0].made is False


def test_static_false_ball_next_to_hoop_gives_no_shot():
    """TRACK's ball model fires on orange wall fixtures beside the backboard:
    a ball at a constant hoop-relative offset must never become a shot."""
    hx, hy = FIXTURE_HOOP[0], FIXTURE_HOOP[1]
    for dy in (-30.0, 20.0):  # above the rim / below the rim
        frames = synthetic_scenario("pass")
        for fr in frames:
            fr.ball = Ball(center=(hx - 25.0, hy + dy), bbox=(hx - 37, hy + dy - 12, hx - 13, hy + dy + 12), conf=0.5)
        events, stats = build(frames, fps=50, clip="x")
        assert events["shots"] == []

    # alternating with a real ball below the rim (the model picks one per frame)
    frames = synthetic_scenario("pass")
    for n, fr in enumerate(frames):
        if n % 2 == 0:
            fr.ball = Ball(center=(hx - 25.0, hy - 30.0), bbox=(hx - 37, hy - 42, hx - 13, hy - 18), conf=0.5)
    events, _ = build(frames, fps=50, clip="x")
    assert events["shots"] == []


def test_static_ball_with_panning_camera_gives_no_shot():
    frames = synthetic_scenario("pass")
    for n, fr in enumerate(frames):
        dx = 3.0 * n
        fr.hoops = [(h[0] + dx, h[1], h[2] + dx, h[3]) for h in fr.hoops]
        fr.ball = Ball(center=(FIXTURE_HOOP[0] - 25.0 + dx, FIXTURE_HOOP[1] - 30.0), conf=0.5)
        fr.players = []
    events, _ = build(frames, fps=50, clip="x")
    assert events["shots"] == []


def test_static_false_ball_is_dropped_while_real_ball_flies():
    """dev60 56 s: an orange wall sign is reported as the ball in some frames
    while the real ball is in the air; the static samples must go."""
    frames = synthetic_scenario("made")
    static = (1189.0, 384.0)
    for n, fr in enumerate(frames):
        if 2.6 <= fr.t <= 3.4 and n % 3 == 0:
            fr.ball = Ball(center=static, bbox=(static[0] - 12, static[1] - 15, static[0] + 12, static[1] + 15), conf=0.5)
    engine = StatsEngine(dt=0.02, fps=50)
    for fr in frames:
        engine.push(fr)
    engine.finish()
    assert engine.dropped_static >= 10
    assert [(s.player_id, s.made) for s in engine.shots] == [(2, True)]
