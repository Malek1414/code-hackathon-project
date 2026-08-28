"""Degradation check: with an empty root every section must answer without
an error and the images must be None. Run: .venv/bin/python -m pytest vision/monitor -q"""
from pathlib import Path

from vision.monitor import images, status


def _point_to(tmp: Path):
    status.FRAMES_DIR = tmp / "data" / "frames"
    status.DATASET_DIR = tmp / "data" / "dataset"
    status.LABELS_DIR = status.DATASET_DIR / "labels"
    status.RUNS_DIR = tmp / "runs"
    status.OUT_DIR = tmp / "out"
    status.QA_DIR = status.OUT_DIR / "qa"
    status.CLIPS_DIR = tmp / "data" / "clips"
    status._tracks_counter = status._LineCounter(status.OUT_DIR / "tracks.jsonl")
    status._memo = None
    images._cache.clear()


def test_empty_root_degrades(tmp_path):
    _point_to(tmp_path)
    snap = status.collect()
    for key in ("label", "track", "stats", "court", "numbers", "qa", "live", "footage", "logs"):
        assert "error" not in snap[key], snap[key]
    assert snap["label"]["labels"] == 0 and snap["label"]["frames"] == 0
    assert snap["track"]["ok"] is False
    assert snap["images"] == {"label": None, "track": None, "numbers": None, "court": None}


def test_partial_and_broken_files(tmp_path):
    _point_to(tmp_path)
    out = status.OUT_DIR
    out.mkdir(parents=True)
    (out / "tracks.jsonl").write_text('{"frame": 0, "t": 0.0, "players": [{"id": 1, "team": 0}], "ball": null}\n{"frame": 2, "t": 0.04, "players": [], "ball": {"center": [1, 2]}}\n{"frame": 4, "pl')
    (out / "events.json").write_text("{ not json")
    (out / "overlay.mp4").write_bytes(b"\x00" * 100)
    (out / "x.log").write_text("a\nb\n")
    snap = status.collect()
    tr = snap["track"]
    assert tr["lines"] == 2 and tr["ball_hits"] == 1 and tr["team_boxes"]["0"] == 1
    assert "error" in snap["stats"]  # reported, not raised
    assert snap["images"]["track"] is None
    assert snap["logs"]["logs"][0]["lines"] == ["a", "b"]


def test_identities_and_live(tmp_path):
    _point_to(tmp_path)
    out = status.OUT_DIR
    (out / "qa").mkdir(parents=True)
    (out / "qa" / "shot_1_made.jpg").write_bytes(b"x")
    (out / "qa" / "index.html").write_text("<p>qa</p>")
    (out / "identities.json").write_text('{"clip": "c", "tracks": {"7": {"team": 0, "number": 12, "votes": {"12": 9, "17": 1}, "reads": 10}, "8": {"team": 1, "number": null, "votes": {}, "reads": 0}}, "players": [{"key": "A12", "team": 0, "number": 12, "track_ids": [7], "first_t": 0, "last_t": 5}, {"key": "B?8", "team": 1, "number": null, "track_ids": [8]}]}')
    (out / "live_events.json").write_text('{"clip": "0", "shots": [{}], "score": {"0": {"points": 4, "fga": 3, "fgm": 2}, "1": {"points": 0, "fga": 1, "fgm": 0}}, "unassigned_baskets": 1, "frames_processed": 10}')
    snap = status.collect()
    n = snap["numbers"]
    assert n["tracks_numbered"] == 1 and n["tracks_total"] == 2 and n["players"][0]["votes"] == 10
    assert snap["qa"]["sheets"] == 1 and snap["qa"]["index"] is True and snap["qa"]["kinds"] == {"shot": 1}
    assert snap["live"]["events"]["teams"]["0"]["points"] == 4 and snap["live"]["events"]["unassigned"] == 1
