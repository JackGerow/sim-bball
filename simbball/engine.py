"""Possession simulator.

Each simulation walks a full game: tip, rotation, shot clock, bonus,
offensive rebounds, and foul-outs. Counting stats are added only on the
event that creates them, so the box score always sums to the final score.

The 10,000-run path and the replay path share this function. Replay mode
records every event but does not draw any extra random numbers, so the
same seed reproduces the same game.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from simbball.matchups import MatchupBook, shot_multiplier, turnover_multiplier, usage_multiplier
from simbball.ratings import League, Player, TeamModel

FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, AST, STL, BLK, TOV, PF, PTS = range(14)
NSTATS = 14

# Share of turnovers that are offensive fouls (no steal, no team foul).
OFFENSIVE_FOUL_SHARE = 0.16
# Misses with no player rebound (out of bounds, team rebound).
DEAD_BALL_SHARE = 0.08
ASSIST_RATE_3 = 0.80
ASSIST_RATE_2 = 0.57
AND1_RATE_3 = 0.035
AND1_RATE_2 = 0.075
# Player-level shot making runs a little hot versus team ratings because
# high-usage players are the better shooters. This brings a league-average
# matchup back to the observed ~113 points a game.
MAKE_CALIBRATION = 0.962


@dataclass
class GameSim:
    home_score: int
    away_score: int
    box_home: np.ndarray
    sec_home: np.ndarray
    box_away: np.ndarray
    sec_away: np.ndarray
    possessions: int
    periods: list[tuple[int, int]]
    plays: list[dict] | None


@dataclass
class MatchupResult:
    scores: np.ndarray
    mean_box_home: np.ndarray
    mean_sec_home: np.ndarray
    mean_box_away: np.ndarray
    mean_sec_away: np.ndarray
    mode_box_home: np.ndarray
    mode_sec_home: np.ndarray
    mode_box_away: np.ndarray
    mode_sec_away: np.ndarray
    selected: GameSim
    selected_index: int
    mode_count: int
    home_win_prob: float
    exp_margin: float
    seeds: np.ndarray


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def implied_margin(home_win_prob: float, sd: float = 12.0) -> float:
    """Point margin whose pregame win probability matches the simulations."""
    target = _clamp(home_win_prob, 0.03, 0.97)
    lo, hi = -40.0, 40.0
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if _phi(mid / sd) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _wchoice(idxs, weights, unit: float) -> int:
    total = 0.0
    for idx in idxs:
        total += weights[idx]
    if total <= 0.0:
        return int(idxs[0])
    mark = unit * total
    acc = 0.0
    for idx in idxs:
        acc += weights[idx]
        if acc >= mark:
            return int(idx)
    return int(idxs[-1])


def _weights(players: list[Player], name: str) -> list[float]:
    return [float(getattr(player, name)) for player in players]


def build_rotation(players: list[Player], starter_idx: list[int]) -> list[list[int]]:
    """Half-minute lineup plan for regulation. Targets are in seconds."""
    n = len(players)
    slots = 96
    slot_sec = 30.0
    remaining = [player.target_sec for player in players]
    starters = set(starter_idx)
    sitting_until = [0] * n
    current = list(starter_idx)
    plan: list[list[int]] = []
    for slot in range(slots):
        quarter = slot // 24
        into = slot % 24
        start_q = into <= 3
        crunch = quarter == 3 and into >= 18
        scored: list[tuple[float, int]] = []
        for idx in range(n):
            score = remaining[idx]
            if idx in current:
                score += 150.0
            if (start_q or crunch) and idx in starters:
                score += 180.0
            if slot < sitting_until[idx] and not crunch and not start_q:
                score -= 10_000.0
            scored.append((score, idx))
        scored.sort(reverse=True)
        chosen = [idx for _, idx in scored[:5]]
        for idx in current:
            if idx not in chosen:
                sitting_until[idx] = slot + 4
        for idx in chosen:
            remaining[idx] -= slot_sec
        current = chosen
        plan.append(chosen)
    # Starters take the opening two minutes. The greedy plan already prefers
    # them, and pinning the tip avoids a substitution before the first play.
    plan[0] = list(starter_idx)
    plan[1] = list(starter_idx)
    plan[2] = list(starter_idx)
    plan[3] = list(starter_idx)
    return plan


def _zone(is3: bool, phase: str, unit: float) -> str:
    if phase == "putback":
        if unit < 0.78:
            return "paint"
        return "mid_left" if unit < 0.89 else "mid_right"
    if phase == "transition" and not is3:
        if unit < 0.7:
            return "paint"
        return "mid_left" if unit < 0.85 else "mid_right"
    if is3:
        if unit < 0.12:
            return "corner_left"
        if unit < 0.24:
            return "corner_right"
        if unit < 0.50:
            return "wing_left"
        if unit < 0.76:
            return "wing_right"
        return "top"
    if unit < 0.62:
        return "paint"
    if unit < 0.78:
        return "mid_left"
    if unit < 0.94:
        return "mid_right"
    return "elbow"


def _shot_text(zone: str, is3: bool) -> str:
    if is3:
        return "3-POINTER"
    if zone == "paint":
        return "LAYUP"
    return "PULL-UP"


class _Side:
    def __init__(self, team: TeamModel):
        self.team = team
        self.n = len(team.players)
        self.usage = _weights(team.players, "usage")
        self.p3 = _weights(team.players, "p3")
        self.fg2 = _weights(team.players, "fg2")
        self.fg3 = _weights(team.players, "fg3")
        self.ft = _weights(team.players, "ft")
        self.oreb = _weights(team.players, "oreb")
        self.dreb = _weights(team.players, "dreb")
        self.ast = _weights(team.players, "ast")
        self.stl = _weights(team.players, "stl")
        self.blk = _weights(team.players, "blk")
        self.pf = _weights(team.players, "pf")
        self.plan = build_rotation(team.players, team.starter_idx)
        self.box = np.zeros((self.n, NSTATS), dtype=np.int32)
        self.secs = np.zeros(self.n, dtype=np.float64)
        self.fouls = np.zeros(self.n, dtype=np.int16)
        self.lineup = list(team.starter_idx)
        self.order = sorted(range(self.n), key=lambda i: team.players[i].target_sec, reverse=True)


def _matchup(off: TeamModel, deff: TeamModel, home: bool, league: League) -> dict:
    p_tov = off.p_tov * _clamp(deff.tov_forced / max(league.tov_forced, 1e-6), 0.88, 1.14)
    p_sfl = off.p_sfl
    p_shot = off.p_shot
    total = p_tov + p_sfl + p_shot
    scale = ((off.ortg + deff.drtg) / 2.0) / max(off.ortg, 1.0)
    scale = _clamp(scale, 0.92, 1.08) * (1.012 if home else 0.992) * MAKE_CALIBRATION
    allowed = 1.0 - deff.dreb_share
    oreb = _clamp(0.5 * (off.oreb_share + allowed), 0.20, 0.36)
    return {
        "p_tov": p_tov / total,
        "p_sfl": p_sfl / total,
        "p_shot": p_shot / total,
        "make": scale,
        "oreb": oreb,
        "block": _clamp(deff.blk_per_miss, 0.04, 0.18),
        "steal": _clamp(deff.steal_share / (1.0 - OFFENSIVE_FOUL_SHARE), 0.45, 0.82),
    }


def _lineup_for(side: _Side, quarter: int, clock: float) -> list[int]:
    if quarter <= 4:
        elapsed = (quarter - 1) * 720.0 + (720.0 - clock)
        slot = int(min(95, max(0, elapsed // 30.0)))
        # Hold a lineup for two minutes so the video is not a stream of subs.
        slot = (slot // 4) * 4
        desired = list(side.plan[slot])
    else:
        desired = list(side.team.starter_idx)
    chosen: list[int] = []
    for idx in desired:
        if side.fouls[idx] < 6 and idx not in chosen:
            chosen.append(idx)
    if len(chosen) < 5:
        for idx in side.order:
            if side.fouls[idx] < 6 and idx not in chosen:
                chosen.append(idx)
            if len(chosen) == 5:
                break
    if len(chosen) < 5:
        for idx in range(side.n):
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) == 5:
                break
    return chosen[:5]


def _snap(side: _Side) -> list[list[float]]:
    rows = []
    for idx in range(side.n):
        box = side.box[idx]
        rows.append(
            [
                round(float(side.secs[idx]), 1),
                int(box[FGM]),
                int(box[FGA]),
                int(box[FG3M]),
                int(box[FG3A]),
                int(box[FTM]),
                int(box[FTA]),
                int(box[OREB] + box[DREB]),
                int(box[AST]),
                int(box[STL]),
                int(box[BLK]),
                int(box[TOV]),
                int(box[PF]),
                int(box[PTS]),
            ]
        )
    return rows


def simulate_game(
    home: TeamModel,
    away: TeamModel,
    league: League,
    seed: int,
    record: bool = False,
) -> GameSim:
    rng = np.random.default_rng(seed)
    sides = [_Side(home), _Side(away)]
    profiles = [
        _matchup(home, away, True, league),
        _matchup(away, home, False, league),
    ]
    mean_base = max(6.0, 2880.0 / max(home.pace + away.pace, 1.0) - 0.55)
    score = [0, 0]
    quarter = 1
    clock = 720.0
    team_fouls = [0, 0]
    plays: list[dict] | None = [] if record else None
    periods: list[tuple[int, int]] = []
    period_start = [0, 0]
    possessions = 0
    offense = 0 if rng.random() < 0.5 else 1
    arrow = 1 - offense
    phase = "halfcourt"
    previous = [None, None]

    def lineup_now():
        return (
            _lineup_for(sides[0], quarter, clock),
            _lineup_for(sides[1], quarter, clock),
        )

    def stamp(clock0, clock1, shot_clock, off, actor, text, result, points, zone, shooter, assist, blocker, rebound, play_phase, made, sub_text=""):
        if plays is None:
            return
        plays.append(
            {
                "type": "shot" if text in {"3-POINTER", "LAYUP", "PULL-UP"} else "other",
                "q": quarter,
                "clock0": clock0,
                "clock1": clock1,
                "shot_clock": shot_clock,
                "off": "H" if off == 0 else "A",
                "actor": ("H" if actor[0] == 0 else "A", actor[1]),
                "text": text,
                "result": result,
                "points": points,
                "score_before": (score[0] - (points if off == 0 else 0), score[1] - (points if off == 1 else 0)),
                "score_after": (score[0], score[1]),
                "fouls": (team_fouls[0], team_fouls[1]),
                "lineup": {"H": list(sides[0].lineup), "A": list(sides[1].lineup)},
                "box": {"H": _snap(sides[0]), "A": _snap(sides[1])},
                "zone": zone,
                "shooter": shooter,
                "assist": assist,
                "blocker": blocker,
                "rebound": None if rebound is None else ("H" if rebound[0] == 0 else "A", rebound[1]),
                "phase": play_phase,
                "made": made,
                "sub_text": sub_text,
            }
        )

    # score_before above subtracts points that were already added. That is correct
    # only when those points were added for this play and for this team. Callers
    # must add the points before stamp(). FT and FG are stamped separately.

    def set_type(kind: str):
        if plays:
            plays[-1]["type"] = kind

    def sync_lineups():
        desired = lineup_now()
        notes = []
        changed = False
        for side_i, want in enumerate(desired):
            have = previous[side_i]
            sides[side_i].lineup = list(want)
            if have is not None and set(have) != set(want):
                changed = True
                gone = [i for i in have if i not in want]
                coming = [i for i in want if i not in have]
                team = sides[side_i].team
                label = team.abbrev
                bits = []
                for idx in coming:
                    bits.append(f"{team.players[idx].last_upper} IN")
                for idx in gone:
                    bits.append(f"{team.players[idx].last_upper} OUT")
                if bits:
                    notes.append(f"{label} {' · '.join(bits)}")
            previous[side_i] = list(want)
        if changed and plays is not None and notes:
            stamp(clock, clock, 24, offense, (0, desired[0][0]), "SUB", "", 0, None, None, None, None, None, "halfcourt", False, "   ".join(notes))
            set_type("sub")

    def add_points(side_i: int, player: int, amount: int):
        sides[side_i].box[player, PTS] += amount
        score[side_i] += amount

    def rebound(off_i: int, def_i: int, profile: dict, putbacks: int):
        unit = rng.random()
        if unit < DEAD_BALL_SHARE:
            return "dead", None
        if unit < DEAD_BALL_SHARE + (1.0 - DEAD_BALL_SHARE) * profile["oreb"] and putbacks < 3:
            player = _wchoice(sides[off_i].lineup, sides[off_i].oreb, rng.random())
            sides[off_i].box[player, OREB] += 1
            return "oreb", (off_i, player)
        player = _wchoice(sides[def_i].lineup, sides[def_i].dreb, rng.random())
        sides[def_i].box[player, DREB] += 1
        return "dreb", (def_i, player)

    def free_throws(off_i: int, shooter: int, fta: int, profile: dict, zone: str, clock0: float):
        kept = False
        for shot_i in range(fta):
            made = rng.random() < sides[off_i].ft[shooter]
            sides[off_i].box[shooter, FTA] += 1
            if made:
                sides[off_i].box[shooter, FTM] += 1
                add_points(off_i, shooter, 1)
            rebound_to = None
            last = shot_i == fta - 1
            if last and not made:
                kind, who = rebound(off_i, 1 - off_i, profile, 0)
                rebound_to = who
                kept = kind == "oreb"
            if record:
                stamp(
                    clock0, clock0, 24, off_i, (off_i, shooter),
                    "FREE THROW", "MADE" if made else "MISSED",
                    1 if made else 0, zone, shooter, None, None, rebound_to, "ft", made,
                )
                set_type("ft")
        return kept

    if record:
        sides[0].lineup = list(home.starter_idx)
        sides[1].lineup = list(away.starter_idx)
        stamp(720.0, 720.0, 24, offense, (offense, sides[offense].lineup[0]), "TIP OFF", "", 0, None, None, None, None, None, "halfcourt", False)
        set_type("intro")
        previous[0] = list(sides[0].lineup)
        previous[1] = list(sides[1].lineup)

    while quarter <= 10:
        if record and quarter > 1:
            stamp(clock, clock, 24, offense, (offense, 0), f"Q{quarter}" if quarter <= 4 else "OT", "", 0, None, None, None, None, None, "halfcourt", False)
            set_type("quarter")
        while clock > 0.15:
            sync_lineups()
            play_mark = len(plays) if plays is not None else 0
            clock_at_start = clock
            off_i = offense
            def_i = 1 - offense
            profile = profiles[off_i]
            off = sides[off_i]
            deff = sides[def_i]
            book: MatchupBook | None = league.matchups
            if book is None:
                usage = off.usage
                off_ids: list[int] = []
                def_ids: list[int] = []
                tov_mult = 1.0
            else:
                off_ids = [off.team.players[idx].person_id for idx in off.lineup]
                def_ids = [deff.team.players[idx].person_id for idx in deff.lineup]
                usage = list(off.usage)
                for idx, pid in zip(off.lineup, off_ids):
                    mates = [other for other in off_ids if other != pid]
                    usage[idx] = off.usage[idx] * usage_multiplier(book, pid, mates, def_ids)
                tov_mult = turnover_multiplier(book, def_ids)
            p_tov = profile["p_tov"] * tov_mult
            p_sfl = profile["p_sfl"]
            p_shot = profile["p_shot"]
            trip_total = p_tov + p_sfl + p_shot
            p_tov /= trip_total
            p_sfl /= trip_total
            dt = _clamp(mean_base + (rng.random() - 0.5) * 12.0, 5.0, 22.0)
            elapsed_extra = 0.0
            putbacks = 0
            trip_phase = phase
            end_possession = False
            while not end_possession:
                if rng.random() < 0.05 and team_fouls[def_i] < 5:
                    fouler = _wchoice(deff.lineup, deff.pf, rng.random())
                    deff.box[fouler, PF] += 1
                    deff.fouls[fouler] += 1
                    team_fouls[def_i] += 1
                handler = _wchoice(off.lineup, usage, rng.random())
                kind_u = rng.random()
                if kind_u < p_tov:
                    if rng.random() < OFFENSIVE_FOUL_SHARE:
                        off.box[handler, PF] += 1
                        off.fouls[handler] += 1
                        off.box[handler, TOV] += 1
                        if record:
                            stamp(clock, clock, 14, off_i, (off_i, handler), "OFFENSIVE FOUL", "TURNOVER", 0, "elbow", handler, None, None, None, trip_phase, False)
                            set_type("turnover")
                    else:
                        stealer = None
                        if rng.random() < profile["steal"]:
                            stealer = _wchoice(deff.lineup, deff.stl, rng.random())
                            deff.box[stealer, STL] += 1
                        off.box[handler, TOV] += 1
                        if record:
                            actor = (def_i, stealer) if stealer is not None else (off_i, handler)
                            stamp(
                                clock, clock, 16, off_i, actor,
                                "STEAL" if stealer is not None else "TURNOVER",
                                "STEAL" if stealer is not None else "TURNOVER",
                                0, "elbow", handler, None, None, None, "transition" if stealer is not None else trip_phase, False,
                            )
                            set_type("turnover")
                        if stealer is not None:
                            phase = "transition"
                        else:
                            phase = "halfcourt"
                    end_possession = True
                elif kind_u < p_tov + p_sfl:
                    is3 = rng.random() < (0.10 if off.p3[handler] >= 0.35 else 0.04)
                    fta = 3 if is3 else 2
                    fouler = _wchoice(deff.lineup, deff.pf, rng.random())
                    deff.box[fouler, PF] += 1
                    deff.fouls[fouler] += 1
                    team_fouls[def_i] += 1
                    zone = "top" if is3 else "paint"
                    if record:
                        stamp(clock, clock, 12, off_i, (off_i, handler), "3-SHOT FOUL" if is3 else "SHOOTING FOUL", "", 0, zone, handler, None, None, None, trip_phase, False)
                        set_type("foul")
                    kept = free_throws(off_i, handler, fta, profile, zone, clock)
                    if kept:
                        putbacks += 1
                        trip_phase = "putback"
                        elapsed_extra += 4.0
                        phase = "halfcourt"
                    else:
                        phase = "halfcourt"
                        end_possession = True
                else:
                    is3 = rng.random() < off.p3[handler]
                    zone = _zone(is3, trip_phase, rng.random())
                    p_make = (off.fg3[handler] if is3 else off.fg2[handler]) * profile["make"]
                    if book is not None:
                        mates = [other for other in off_ids if other != off.team.players[handler].person_id]
                        p_make *= shot_multiplier(book, off.team.players[handler].person_id, mates, def_ids, is3)
                    p_make = _clamp(p_make, 0.18 if is3 else 0.28, 0.55 if is3 else 0.82)
                    made = rng.random() < p_make
                    off.box[handler, FGA] += 1
                    if is3:
                        off.box[handler, FG3A] += 1
                    assist = None
                    blocker = None
                    rebound_to = None
                    if made:
                        off.box[handler, FGM] += 1
                        if is3:
                            off.box[handler, FG3M] += 1
                        points = 3 if is3 else 2
                        add_points(off_i, handler, points)
                        if rng.random() < (ASSIST_RATE_3 if is3 else ASSIST_RATE_2):
                            assist = _wchoice([i for i in off.lineup if i != handler], off.ast, rng.random())
                            off.box[assist, AST] += 1
                        if record:
                            stamp(clock, clock, max(1, int(round(24 - min(dt, 23)))), off_i, (off_i, handler), _shot_text(zone, is3), "MADE", points, zone, handler, assist, None, None, trip_phase, True)
                            set_type("shot")
                        kept_ball = False
                        if rng.random() < (AND1_RATE_3 if is3 else AND1_RATE_2):
                            fouler = _wchoice(deff.lineup, deff.pf, rng.random())
                            deff.box[fouler, PF] += 1
                            deff.fouls[fouler] += 1
                            team_fouls[def_i] += 1
                            kept_ball = free_throws(off_i, handler, 1, profile, zone, clock)
                        if kept_ball:
                            putbacks += 1
                            trip_phase = "putback"
                            elapsed_extra += 4.0
                        else:
                            phase = "halfcourt"
                            end_possession = True
                    else:
                        blocked = rng.random() < profile["block"]
                        if blocked:
                            blocker = _wchoice(deff.lineup, deff.blk, rng.random())
                            deff.box[blocker, BLK] += 1
                        kind, who = rebound(off_i, def_i, profile, putbacks)
                        rebound_to = who
                        if record:
                            stamp(
                                clock, clock, max(1, int(round(24 - min(dt, 23)))), off_i, (off_i, handler),
                                _shot_text(zone, is3), "BLOCKED" if blocked else "MISSED", 0, zone, handler, None, blocker, rebound_to, trip_phase, False,
                            )
                            set_type("shot")
                        if kind == "oreb":
                            putbacks += 1
                            trip_phase = "putback"
                            elapsed_extra += 4.0
                        else:
                            if kind == "dreb" and rng.random() < 0.20:
                                phase = "transition"
                            else:
                                phase = "halfcourt"
                            end_possession = True
            spent = min(clock, dt + elapsed_extra)
            for idx in off.lineup:
                off.secs[idx] += spent
            for idx in deff.lineup:
                deff.secs[idx] += spent
            clock -= spent
            if plays is not None and len(plays) > play_mark:
                group = plays[play_mark:]
                span = len(group)
                for index, play in enumerate(group):
                    play["clock0"] = clock_at_start - spent * index / span
                    play["clock1"] = clock_at_start - spent * (index + 1) / span
            possessions += 1
            offense = def_i
        periods.append((score[0] - period_start[0], score[1] - period_start[1]))
        period_start = [score[0], score[1]]
        if quarter >= 4 and score[0] != score[1]:
            break
        if quarter >= 8 and score[0] == score[1]:
            winner = 0 if rng.random() < 0.5 else 1
            add_points(winner, sides[winner].lineup[0], 2)
            break
        quarter += 1
        clock = 300.0 if quarter > 4 else 720.0
        team_fouls = [0, 0]
        if quarter > 4:
            offense = 0 if rng.random() < 0.5 else 1
            arrow = 1 - offense
        else:
            offense = arrow
            arrow = 1 - arrow
        phase = "halfcourt"

    if plays is not None:
        # Fix score_before for plays. It was computed assuming the points on
        # that play belong to the offense. Subs and intros have 0 points.
        # Recompute clocks so a multi-event possession does not replay the
        # whole clock on every event. Events between syncs share one window.
        _assign_clocks(plays)
        stamp(0.0, 0.0, 0, offense, (0, sides[0].lineup[0]), "FINAL", "", 0, None, None, None, None, None, "halfcourt", False)
        set_type("final")
        plays[-1]["score_before"] = (score[0], score[1])
        plays[-1]["score_after"] = (score[0], score[1])

    if int(sides[0].box[:, PTS].sum()) != score[0] or int(sides[1].box[:, PTS].sum()) != score[1]:
        raise RuntimeError(
            f"Box score does not add up: {sides[0].box[:, PTS].sum()}-{sides[1].box[:, PTS].sum()} vs {score[0]}-{score[1]}"
        )
    return GameSim(
        home_score=score[0],
        away_score=score[1],
        box_home=sides[0].box,
        sec_home=sides[0].secs,
        box_away=sides[1].box,
        sec_away=sides[1].secs,
        possessions=possessions,
        periods=periods,
        plays=plays,
    )


def _assign_clocks(plays: list[dict]) -> None:
    """Give each event inside a possession an even slice of the clock drop.

    Possessions are separated by sub/quarter/intro marks or by the offense
    flipping. Clock values are filled in later by walking scoreboard time,
    so here we only repair score_before, which was stored at creation.
    """
    for play in plays:
        before_h, before_a = play["score_before"]
        after_h, after_a = play["score_after"]
        # score_before was estimated from the offense's points. Recompute it
        # from the after-score and the points on the play, which is exact.
        pts = int(play["points"])
        if play["off"] == "H":
            play["score_before"] = (after_h - pts, after_a)
        else:
            play["score_before"] = (after_h, after_a - pts)
        if play["score_before"][0] < 0 or play["score_before"][1] < 0:
            play["score_before"] = (before_h, before_a)


def _spread_clocks(plays: list[dict]) -> None:
    groups: list[list[dict]] = []
    current: list[dict] = []
    last_off = None
    for play in plays:
        if play["type"] in {"intro", "quarter", "sub", "final"}:
            if current:
                groups.append(current)
                current = []
            groups.append([play])
            last_off = None
            continue
        if last_off is not None and play["off"] != last_off:
            groups.append(current)
            current = []
        current.append(play)
        last_off = play["off"]
    if current:
        groups.append(current)
    for group in groups:
        if len(group) <= 1 or group[0]["type"] in {"intro", "quarter", "sub", "final"}:
            continue
        start = group[0]["clock0"]
        end = group[-1]["clock1"]
        # clock1 was never reduced inside the possession; both ends still
        # hold the pre-possession clock. The real end is applied by the
        # game loop... see simulate_game, which sets both to `clock` before
        # subtracting. Spread using the spent time stored on the first item.
        spent = group[0].get("_spent", 0.0)
        if spent <= 0:
            continue
        end = start - spent
        span = len(group)
        for index, play in enumerate(group):
            play["clock0"] = start - spent * index / span
            play["clock1"] = start - spent * (index + 1) / span


def _most_common_score(
    scores: np.ndarray,
    pts_h: np.ndarray,
    pts_a: np.ndarray,
    reb_h: np.ndarray,
    reb_a: np.ndarray,
) -> tuple[int, int]:
    """Index of a game whose final score came up most often, and that count.

    Thousands of distinct finals show up in 10,000 games, so an exact box
    score almost never repeats. The outcome that does repeat is the score.
    When two scores tie for the lead, the one with more simulations within
    two points on each side wins, which drops a lone spike at the edge of
    the distribution. Several games share the winning score; the filmed one
    is the member nearest the middle of that group, so the box is a typical
    time that score happened.
    """
    home = scores[:, 0].astype(np.int32)
    away = scores[:, 1].astype(np.int32)
    key = home * 400 + away
    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    top = int(counts.max())
    candidates = np.flatnonzero(counts == top)
    if len(candidates) == 1:
        chosen = int(candidates[0])
    else:
        piles = []
        for slot in candidates:
            score_h = int(uniq[slot] // 400)
            score_a = int(uniq[slot] % 400)
            piles.append(
                int(np.count_nonzero((np.abs(home - score_h) <= 2) & (np.abs(away - score_a) <= 2)))
            )
        chosen = int(candidates[int(np.argmax(np.asarray(piles)))])
    members = np.flatnonzero(inverse == chosen)
    dist = (
        np.abs(pts_h[members] - np.median(pts_h[members], axis=0)).sum(axis=1)
        + np.abs(pts_a[members] - np.median(pts_a[members], axis=0)).sum(axis=1)
        + 0.35
        * (
            np.abs(reb_h[members] - np.median(reb_h[members], axis=0)).sum(axis=1)
            + np.abs(reb_a[members] - np.median(reb_a[members], axis=0)).sum(axis=1)
        )
    )
    return int(members[int(np.argmin(dist))]), top


def simulate_many(
    home: TeamModel,
    away: TeamModel,
    league: League,
    n: int,
    seed: int,
) -> MatchupResult:
    parent = np.random.default_rng(seed)
    seeds = parent.integers(0, 2_147_483_647, size=n, dtype=np.int64)
    scores = np.empty((n, 2), dtype=np.int16)
    pts_h = np.empty((n, len(home.players)), dtype=np.int16)
    pts_a = np.empty((n, len(away.players)), dtype=np.int16)
    reb_h = np.empty((n, len(home.players)), dtype=np.int16)
    reb_a = np.empty((n, len(away.players)), dtype=np.int16)
    sum_box_h = np.zeros((len(home.players), NSTATS), dtype=np.float64)
    sum_box_a = np.zeros((len(away.players), NSTATS), dtype=np.float64)
    sum_sec_h = np.zeros(len(home.players), dtype=np.float64)
    sum_sec_a = np.zeros(len(away.players), dtype=np.float64)
    boxes_h = np.empty((n, len(home.players), NSTATS), dtype=np.float32)
    boxes_a = np.empty((n, len(away.players), NSTATS), dtype=np.float32)
    secs_h = np.empty((n, len(home.players)), dtype=np.float32)
    secs_a = np.empty((n, len(away.players)), dtype=np.float32)
    label = f"{away.abbrev}@{home.abbrev}"
    for index, game_seed in enumerate(seeds):
        if index and index % 2000 == 0:
            print(f"  {label}  {index:,}/{n:,}", flush=True)
        sim = simulate_game(home, away, league, int(game_seed), record=False)
        scores[index, 0] = sim.home_score
        scores[index, 1] = sim.away_score
        pts_h[index] = sim.box_home[:, PTS]
        pts_a[index] = sim.box_away[:, PTS]
        reb_h[index] = sim.box_home[:, OREB] + sim.box_home[:, DREB]
        reb_a[index] = sim.box_away[:, OREB] + sim.box_away[:, DREB]
        sum_box_h += sim.box_home
        sum_box_a += sim.box_away
        sum_sec_h += sim.sec_home
        sum_sec_a += sim.sec_away
        boxes_h[index] = sim.box_home
        boxes_a[index] = sim.box_away
        secs_h[index] = sim.sec_home
        secs_a[index] = sim.sec_away
    best, mode_count = _most_common_score(scores, pts_h, pts_a, reb_h, reb_a)
    members = (scores[:, 0] == scores[best, 0]) & (scores[:, 1] == scores[best, 1])
    selected = simulate_game(home, away, league, int(seeds[best]), record=True)
    if selected.home_score != int(scores[best, 0]) or selected.away_score != int(scores[best, 1]):
        raise RuntimeError(
            f"Replay seed drifted for {label}: {selected.home_score}-{selected.away_score} vs {scores[best, 0]}-{scores[best, 1]}"
        )
    home_wins = float(np.mean(scores[:, 0] > scores[:, 1]) + 0.5 * np.mean(scores[:, 0] == scores[:, 1]))
    return MatchupResult(
        scores=scores,
        mean_box_home=sum_box_h / n,
        mean_sec_home=sum_sec_h / n,
        mean_box_away=sum_box_a / n,
        mean_sec_away=sum_sec_a / n,
        mode_box_home=boxes_h[members].mean(axis=0),
        mode_sec_home=secs_h[members].mean(axis=0),
        mode_box_away=boxes_a[members].mean(axis=0),
        mode_sec_away=secs_a[members].mean(axis=0),
        selected=selected,
        selected_index=best,
        mode_count=mode_count,
        home_win_prob=home_wins,
        exp_margin=implied_margin(home_wins, league.score_sd),
        seeds=seeds,
    )
