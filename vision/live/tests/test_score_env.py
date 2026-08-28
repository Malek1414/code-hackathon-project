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
