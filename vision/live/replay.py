"""Replay tracker: serves tracks lines from an out/tracks.jsonl instead of
running the models. Lets live.py run without the GPU (development, and a
fallback for the demo) with exactly the same downstream path."""

from __future__ import annotations

import bisect
from pathlib import Path

from vision.stats.io import frame_to_dict, read_tracks


class ReplayTracker:
    def __init__(self, path: str | Path, fps: float | None = None) -> None:
        frames = read_tracks(path, fps=fps)
        if not frames:
            raise SystemExit(f"no tracks in {path}")
        self._by_index = {fr.frame: frame_to_dict(fr) for fr in frames}
        self._indices = sorted(self._by_index)
        self.served = 0

    def step(self, frame, frame_index: int, t: float) -> dict:
        """The record of the nearest processed frame at or before `frame_index`
        (re-stamped with the requested frame/t so the engine sees a monotonic
        stream); an empty record before the first one."""
        pos = bisect.bisect_right(self._indices, frame_index) - 1
        if pos < 0:
            return {"frame": frame_index, "t": round(t, 3), "players": [], "ball": None, "hoops": []}
        rec = dict(self._by_index[self._indices[pos]])
        rec["frame"], rec["t"] = frame_index, round(t, 3)
        self.served += 1
        return rec
