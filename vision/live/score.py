"""Running score with auto-called baskets and human veto.

The system calls a basket when STATS's engine reports a made shot with a
known team (+2; no court calibration live, so never +3 automatically). The
human corrects with hotkeys: 1/2 = +2 for team A/B, 3/4 = +3, z = undo the
last action, whether it was automatic or manual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vision.stats.shots import ShotEvent

TEAM_NAMES = {0: "A", 1: "B"}


@dataclass
class Action:
    kind: str  # "auto" | "manual"
    team: int  # 0 / 1, or -1 for an unassigned auto call
    points: int
    fga: int
    fgm: int
    t: float
    label: str


@dataclass
class TeamScore:
    points: int = 0
    fga: int = 0
    fgm: int = 0


@dataclass
class ScoreBoard:
    teams: dict[int, TeamScore] = field(default_factory=lambda: {0: TeamScore(), 1: TeamScore()})
    history: list[Action] = field(default_factory=list)
    unassigned: int = 0  # made baskets the system saw but could not attribute

    def auto_shot(self, ev: ShotEvent, t: float) -> Action:
        """Points only for a confirmed basket with a known team; an
        unconfirmed verdict (ball lost at the rim) is a question to the human."""
        team = ev.team if ev.team in self.teams else -1
        if not ev.made_confirmed:
            name = TEAM_NAMES.get(team, "?")
            likely = bool(ev.made_hint)
            label = f"SHOT team {name}: looked in, press 1 or 2" if likely else f"SHOT team {name}: result unclear"
            act = Action("auto", team, 0, 1 if team >= 0 else 0, 0, t, label)
            if likely:
                self.unassigned += 1
        elif team == -1:
            label = "BASKET? press 1 or 2" if ev.made else "miss (team unknown)"
            act = Action("auto", -1, 0, 0, 0, t, label)
            if ev.made:
                self.unassigned += 1
        else:
            pts = 2 if ev.made else 0
            label = f"BASKET  Team {TEAM_NAMES[team]} +2" if ev.made else f"miss  Team {TEAM_NAMES[team]}"
            act = Action("auto", team, pts, 1, int(ev.made), t, label)
        self._apply(act, +1)
        return act

    def manual(self, team: int, points: int, t: float) -> Action:
        act = Action("manual", team, points, 1, 1, t, f"Team {TEAM_NAMES[team]} +{points} (manual)")
        self._apply(act, +1)
        return act

    def undo(self, t: float) -> Action | None:
        if not self.history:
            return None
        act = self.history.pop()
        self._apply(act, -1, record=False)
        return Action("undo", act.team, act.points, act.fga, act.fgm, t, f"undo: {act.label}")

    def _apply(self, act: Action, sign: int, record: bool = True) -> None:
        if act.team in self.teams:
            ts = self.teams[act.team]
            ts.points += sign * act.points
            ts.fga += sign * act.fga
            ts.fgm += sign * act.fgm
        if sign < 0 and act.points == 0 and act.fgm == 0 and "press 1 or 2" in act.label:
            self.unassigned -= 1
        if record:
            self.history.append(act)

    def line(self) -> str:
        a, b = self.teams[0], self.teams[1]
        return f"A {a.points} : {b.points} B"

    def fg_line(self) -> str:
        a, b = self.teams[0], self.teams[1]
        return f"FG  A {a.fgm}/{a.fga}   B {b.fgm}/{b.fga}"
