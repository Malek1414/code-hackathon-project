#!/usr/bin/env python3
"""Fetch NBA play-by-play rows as event labels (tier-1 weak supervision).

Pulls games for a season and writes one CSV per game with the events the
model cares about: made shots (points; assist named in the description),
missed shots, and rebounds. Timestamps (PERIOD + PCTIMESTRING) are what get
aligned against broadcast video later.

Usage:
  ml/.venv/bin/python ml/fetch_playbyplay.py --season 2024-25 --games 5
"""
import argparse
import csv
import time
from pathlib import Path

from nba_api.stats.endpoints import leaguegamefinder, playbyplayv2

# EVENTMSGTYPE: 1 = made shot, 2 = missed shot, 4 = rebound
EVENT_TYPES = {1: "made_shot", 2: "missed_shot", 4: "rebound"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2024-25")
    ap.add_argument("--games", type=int, default=5, help="how many games to pull")
    ap.add_argument("--out", default="ml/data/pbp")
    a = ap.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    finder = leaguegamefinder.LeagueGameFinder(season_nullable=a.season,
                                               league_id_nullable="00")
    games = finder.get_data_frames()[0]["GAME_ID"].drop_duplicates().head(a.games)

    for game_id in games:
        rows = playbyplayv2.PlayByPlayV2(game_id=game_id).get_data_frames()[0]
        rows = rows[rows["EVENTMSGTYPE"].isin(EVENT_TYPES)]
        path = outdir / f"{game_id}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event_num", "event", "period", "clock",
                        "home_desc", "visitor_desc"])
            for _, r in rows.iterrows():
                w.writerow([r["EVENTNUM"], EVENT_TYPES[r["EVENTMSGTYPE"]],
                            r["PERIOD"], r["PCTIMESTRING"],
                            r["HOMEDESCRIPTION"] or "", r["VISITORDESCRIPTION"] or ""])
        print(f"{game_id}: {len(rows)} events -> {path}")
        time.sleep(0.6)  # stats.nba.com rate limit


if __name__ == "__main__":
    main()
