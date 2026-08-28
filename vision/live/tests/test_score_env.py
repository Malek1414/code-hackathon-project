import os

from vision.live.env import load_dotenv, redact, rtmp_url
from vision.live.live import handle_key
from vision.live.replay import ReplayTracker
from vision.live.score import ScoreBoard
from vision.stats.io import synthetic_scenario, write_tracks
from vision.stats.shots import ShotEvent


def _ev(team, made, confirmed=True, hint=None):
    return ShotEvent(frame=0, t=1.0, player_id=5, team=team, made=made, shooter_foot=None,
                     hoop_bbox=(0, 0, 1, 1), shooter_confirmed=team >= 0, made_confirmed=confirmed, made_hint=hint)


def test_auto_call_manual_and_undo():
    b = ScoreBoard()
    b.auto_shot(_ev(0, True), 1.0)
    b.auto_shot(_ev(1, False), 2.0)
    assert b.line() == "A 2 : 0 B" and b.fg_line() == "FG  A 1/1   B 0/1"
    b.manual(1, 3, 3.0)
    assert b.line() == "A 2 : 3 B" and b.teams[1].fga == 2
    b.undo(4.0)
    assert b.line() == "A 2 : 0 B" and b.teams[1].fga == 1
    b.undo(5.0)  # the miss
    b.undo(6.0)  # the auto basket
    assert b.line() == "A 0 : 0 B" and b.teams[0].fga == 0
    assert b.undo(7.0) is None


def test_unconfirmed_verdict_gives_no_points():
    b = ScoreBoard()
    act = b.auto_shot(_ev(0, False, confirmed=False, hint=True), 1.0)
    assert "press 1 or 2" in act.label and b.line() == "A 0 : 0 B" and b.teams[0].fga == 1 and b.unassigned == 1
    b.undo(2.0)
    assert b.teams[0].fga == 0 and b.unassigned == 0


def test_unassigned_basket_asks_the_human():
    b = ScoreBoard()
    act = b.auto_shot(_ev(-1, True), 1.0)
    assert act.label.startswith("BASKET?") and b.unassigned == 1 and b.line() == "A 0 : 0 B"
    b.undo(2.0)
    assert b.unassigned == 0


def test_dotenv_loader_and_redaction(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nexport FOLLOWCAM_RTMP_URL='rtmp://host/app/secret-key'\nOTHER=1\n")
    monkeypatch.delenv("FOLLOWCAM_RTMP_URL", raising=False)
    monkeypatch.setenv("OTHER", "keep")
    assert load_dotenv(env) == 1
    assert rtmp_url() == "rtmp://host/app/secret-key"
    assert os.environ["OTHER"] == "keep"  # existing environment wins
    assert redact(rtmp_url()) == "rtmp://host/app/***"
    assert "secret-key" not in redact(rtmp_url())
    monkeypatch.setenv("FOLLOWCAM_RTMP_URL", "")
    assert rtmp_url() is None


def test_replay_tracker_serves_nearest_earlier_record(tmp_path):
    path = tmp_path / "tracks.jsonl"
    write_tracks(synthetic_scenario("made", duration_s=1.0)[::5], path)  # frames 0, 5, 10, ...
    rt = ReplayTracker(path, fps=50)
    rec = rt.step(None, 7, 0.14)
    assert rec["frame"] == 7 and rec["t"] == 0.14 and len(rec["players"]) == 3
    assert rt.step(None, 0, 0.0)["frame"] == 0


def test_hotkeys_drive_the_board():
    b = ScoreBoard()
    assert handle_key(ord("1"), b, 1.0) == ("Team A +2 (manual)", True)
    assert handle_key(ord("4"), b, 2.0) == ("Team B +3 (manual)", True)
    assert b.line() == "A 2 : 3 B"
    label, made = handle_key(ord("z"), b, 3.0)
    assert label.startswith("undo:") and made is False and b.line() == "A 2 : 0 B"
    assert handle_key(ord("x"), b, 4.0) is None


def test_capture_survives_camera_silence(monkeypatch):
    import time as _time

    import numpy as np

    import vision.live.live as live

    class FakeCap:
        instances = 0

        def __init__(self, src):
            FakeCap.instances += 1
            self.fail = FakeCap.instances == 1  # the first device stays silent, the reopened one works
            self.props = {}

        def isOpened(self):
            return True

        def set(self, k, v):
            self.props[k] = v

        def get(self, k):
            return 30.0

        def read(self):
            if self.fail:
                return False, None
            return True, np.zeros((4, 4, 3), np.uint8)

        def grab(self):
            pass

        def release(self):
            pass

    monkeypatch.setattr(live.cv2, "VideoCapture", FakeCap)
    cap = live.Capture("0")
    cap.retry_s, cap.reopen_s = 0.0, 0.05
    assert cap.read() is None and not cap.healthy and cap.reopens == 0
    _time.sleep(0.06)
    frame = cap.read()  # reopen kicks in -> the second device delivers
    assert cap.reopens == 1 and frame is not None and cap.healthy
    assert FakeCap.instances == 2


def test_pan_controller_law_and_rate():
    from vision.live.pan import PanController

    pan = PanController(None, dry=True, frame_width=1920)
    # ball far right of centre: angle decreases (default sign), clamped, at most 20 commands/s, >= 1 deg steps
    t = 0.0
    angles = []
    for _ in range(40):
        angles.append(pan.update(1800.0, now=t))
        t += 0.02
    assert angles[0] >= 87.0 and angles[-1] < angles[0] and angles[-1] == 40.0 and all(40 <= a <= 140 for a in angles)
    assert max(abs(b - a) for a, b in zip(angles, angles[1:])) <= 90 * 0.02 + 1e-6  # 90 deg/s slew limit
    assert pan.commands <= 17  # 0.8 s at <= 20 Hz
    # inside the deadband nothing moves
    a0 = pan.update(960.0 + 10, now=t)
    assert abs(pan.update(960.0 - 10, now=t + 0.05) - a0) < 1e-9
    # lost for > 3 s: drifts back towards 90
    pan.update(None, now=t + 3.5)
    a1 = pan.update(None, now=t + 4.5)
    assert abs(a1 - 90.0) < abs(a0 - 90.0)
    inv = PanController(None, dry=True, invert=True, frame_width=1920)
    assert inv.update(1800.0, now=0.0) > 90.0


def test_widget_scheduler_timing():
    import numpy as np

    from vision.live.state import DEFAULT_TEAMS
    from vision.live.widgets import Assets, WidgetScheduler

    teams = [dict(t, score=0, fga=0, fgm=0, fg_pct=None, possessions=0) for t in DEFAULT_TEAMS]
    players = [{"key": "A5", "number": 5, "team": 0, "pts": 2, "fga": 1, "fgm": 1, "fg_pct": 1.0, "possession_s": 3.0,
                "distance_m": None, "track_ids": [5]}]
    ws = WidgetScheduler(Assets("/nonexistent"), title="Test")
    frame = np.zeros((1080, 1920, 3), np.uint8)
    ws.render(frame, 2.0, teams, players, "00:02")
    assert frame[1020, 960].any()  # lower third visible in the first 10 s
    frame[:] = 0
    ws.render(frame, 20.0, teams, players, "00:20")
    assert not frame[1020, 960].any() and frame[60, 960].any()  # lower third gone, score bug always
    ws.made(30.0, 0, "A5", "BASKET Team A +2")
    frame[:] = 0
    ws.render(frame, 31.0, teams, players, "00:31")
    assert frame[950, 100].any()  # player card lower left within 3 s
    frame[:] = 0
    ws.render(frame, 34.5, teams, players, "00:34")
    assert not frame[950, 100].any()  # card gone after 3 s
    ws.hotkey("t", 40.0)
    frame[:] = 0
    ws.render(frame, 42.0, teams, players, "00:42")
    assert frame[540, 960].any()  # team overview centre for 6 s
    ws.hotkey("e", 50.0)
    frame[:] = 0
    ws.render(frame, 51.0, teams, players, "00:51")
    assert frame[100, 100].any() and frame[1000, 1800].any()  # end summary covers the frame


def test_team_config_flags_win_over_menu(tmp_path):
    import json

    from vision.live.state import load_team_config

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"teams": [{"name": "Lions", "color": "#112233"}, {"name": "Wiesel", "color": "#445566"}]}))
    t = load_team_config(cfg)
    assert t[0]["name"] == "Lions" and t[1]["color"] == "#445566"
    t = load_team_config(cfg, team_b="Gäste", color_a="#ffffff")
    assert t[1]["name"] == "Gäste" and t[0]["color"] == "#ffffff" and t[0]["name"] == "Lions"
    t = load_team_config(tmp_path / "missing.json")
    assert [x["name"] for x in t] == ["Team A", "Team B"]
