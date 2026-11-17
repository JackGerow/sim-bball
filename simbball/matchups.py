"""On-court usage and scoring for the five players actually sharing the floor.

Play-by-play through the 2025-26 season records substitutions, so each shot
and turnover can be charged to the five offensive players and the five
defenders on the court. An exact ten-man group is used when that group has
been seen enough. Most 2026 lineups are new, so a new group falls back to
how the player has played next to each of those teammates, and to how
opposing shooters have fared with each of those five defenders on the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

HISTORY_START = "2023-10-01"
HISTORY_END = "2026-07-01"
# Cells thinner than this are noise. The lookup still shrinks whatever remains.
EXACT_MIN = 20
PAIR_MIN = 30
DEF_MIN = 80


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _pid(value) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text or text in {"0", "0.0"}:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


@dataclass
class MatchupBook:
    """Counts are (possessions, usage events, 2PA, 2PM, 3PA, 3PM) unless noted."""

    overall: dict[int, tuple[int, int, int, int, int, int]] = field(default_factory=dict)
    pairs: dict[int, dict[int, tuple[int, int]]] = field(default_factory=dict)
    exact: dict[tuple, tuple[int, int, int, int, int, int]] = field(default_factory=dict)
    defense: dict[int, tuple[int, int, int, int, int, int]] = field(default_factory=dict)
    league_fg2: float = 0.54
    league_fg3: float = 0.36
    league_tov: float = 0.13
    games: int = 0
    possessions: int = 0


def _starters(raw: Path) -> dict[str, dict[int, list[int]]]:
    frame = (
        pl.scan_csv(
            raw / "PlayerStatistics.csv",
            schema_overrides={
                "gameId": pl.Utf8,
                "personId": pl.Utf8,
                "playerteamId": pl.Utf8,
                "startingPosition": pl.Utf8,
                "gameDateTimeEst": pl.Utf8,
                "gameType": pl.Utf8,
            },
            ignore_errors=True,
        )
        .filter(
            (pl.col("gameType") == "Regular Season")
            & (pl.col("gameDateTimeEst") >= HISTORY_START)
            & (pl.col("gameDateTimeEst") < HISTORY_END)
            & (pl.col("startingPosition").fill_null("") != "")
        )
        .select(["gameId", "personId", "playerteamId"])
        .collect()
    )
    out: dict[str, dict[int, list[int]]] = {}
    for game_id, person, team in frame.iter_rows():
        pid = _pid(person)
        tid = _pid(team)
        if not pid or not tid:
            continue
        roster = out.setdefault(str(game_id), {}).setdefault(tid, [])
        if pid not in roster and len(roster) < 5:
            roster.append(pid)
    return {game: teams for game, teams in out.items() if sum(len(v) == 5 for v in teams.values()) >= 2}


def build_matchup_book(raw: Path) -> MatchupBook:
    starters = _starters(raw)
    game_ids = list(starters)
    print(f"Reading lineup stints from {len(game_ids):,} games", flush=True)
    events = (
        pl.scan_parquet(raw / "PlayByPlay.parquet")
        .filter(
            (pl.col("gameDateTimeEst") >= HISTORY_START)
            & (pl.col("gameDateTimeEst") < HISTORY_END)
            & pl.col("gameId").is_in(game_ids)
            & pl.col("actionType").is_in(["period", "substitution", "2pt", "3pt", "turnover"])
        )
        .select(["gameId", "orderNumber", "period", "actionType", "subType", "personId", "teamId", "shotResult", "description"])
        .collect()
        .sort(["gameId", "orderNumber"])
    )
    overall: dict[int, list[int]] = {}
    pairs: dict[int, dict[int, list[int]]] = {}
    exact: dict[tuple, list[int]] = {}
    defense: dict[int, list[int]] = {}
    current = None
    base: dict[int, list[int]] = {}
    lineups: dict[int, set[int]] = {}
    credited = 0
    skipped = 0

    for game_id, _order, period, action, sub_type, person, team, result, description in events.iter_rows():
        if game_id != current:
            current = game_id
            base = {tid: list(pids) for tid, pids in starters.get(str(game_id), {}).items()}
            lineups = {tid: set(pids) for tid, pids in base.items()}
        if action == "period":
            # Quarter end is also a period row. Resetting there would wipe the
            # lineup that actually starts the next quarter.
            if period in (1, 3) and str(description or "").startswith("Period Start") and base:
                lineups = {tid: set(pids) for tid, pids in base.items()}
            continue
        if action == "substitution":
            pid = _pid(person)
            tid = _pid(team)
            if not pid:
                continue
            if tid not in lineups:
                for candidate, members in lineups.items():
                    if sub_type == "out" and pid in members:
                        tid = candidate
                        break
                    if sub_type == "in" and len(members) < 5 and pid not in members:
                        tid = candidate
                        break
            members = lineups.get(tid)
            if members is None:
                continue
            if sub_type == "out":
                members.discard(pid)
            elif sub_type == "in" and pid not in members and len(members) < 5:
                members.add(pid)
            continue
        actor = _pid(person)
        if not actor:
            skipped += 1
            continue
        offense = None
        for tid, members in lineups.items():
            if actor in members and len(members) == 5:
                offense = tid
                break
        if offense is None:
            skipped += 1
            continue
        defenders = None
        for tid, members in lineups.items():
            if tid != offense and len(members) == 5:
                defenders = members
                break
        if defenders is None:
            skipped += 1
            continue
        off = tuple(sorted(lineups[offense]))
        opp = tuple(sorted(defenders))
        is3 = action == "3pt"
        made = result == "Made"
        for pid in off:
            row = overall.get(pid)
            if row is None:
                row = [0, 0, 0, 0, 0, 0]
                overall[pid] = row
            row[0] += 1
            mates = tuple(other for other in off if other != pid)
            cell = exact.get((pid, mates, opp))
            if cell is None:
                cell = [0, 0, 0, 0, 0, 0]
                exact[(pid, mates, opp)] = cell
            cell[0] += 1
            bucket = pairs.get(pid)
            if bucket is None:
                bucket = {}
                pairs[pid] = bucket
            for mate in mates:
                pair = bucket.get(mate)
                if pair is None:
                    pair = [0, 0]
                    bucket[mate] = pair
                pair[0] += 1
                if pid == actor:
                    pair[1] += 1
            if pid != actor:
                continue
            row[1] += 1
            cell[1] += 1
            if action == "turnover":
                continue
            slot = 4 if is3 else 2
            row[slot] += 1
            cell[slot] += 1
            if made:
                row[slot + 1] += 1
                cell[slot + 1] += 1
        for pid in opp:
            drow = defense.get(pid)
            if drow is None:
                drow = [0, 0, 0, 0, 0, 0]
                defense[pid] = drow
            drow[0] += 1
            if action == "turnover":
                drow[5] += 1
                continue
            slot = 3 if is3 else 1
            drow[slot] += 1
            if made:
                drow[slot + 1] += 1
        credited += 1

    fga2 = sum(row[2] for row in overall.values())
    fgm2 = sum(row[3] for row in overall.values())
    fga3 = sum(row[4] for row in overall.values())
    fgm3 = sum(row[5] for row in overall.values())
    tov = sum(row[5] for row in defense.values()) // 5
    poss = credited or 1
    book = MatchupBook(
        overall={pid: tuple(row) for pid, row in overall.items() if row[0] >= DEF_MIN},
        pairs={
            pid: {mate: (pair[0], pair[1]) for mate, pair in bucket.items() if pair[0] >= PAIR_MIN}
            for pid, bucket in pairs.items()
        },
        exact={key: tuple(row) for key, row in exact.items() if row[0] >= EXACT_MIN},
        defense={pid: tuple(row) for pid, row in defense.items() if row[0] >= DEF_MIN},
        league_fg2=(fgm2 / fga2) if fga2 else 0.54,
        league_fg3=(fgm3 / fga3) if fga3 else 0.36,
        league_tov=tov / poss,
        games=len(starters),
        possessions=credited,
    )
    book.pairs = {pid: bucket for pid, bucket in book.pairs.items() if bucket}
    print(
        f"Lineup matchups  {credited:,} possessions  {skipped:,} skipped"
        f"  {len(book.exact):,} ten-man cells  {sum(len(v) for v in book.pairs.values()):,} teammate pairs",
        flush=True,
    )
    return book


def _rate(makes: int, attempts: int, prior: float, prior_n: float) -> float:
    return (makes + prior_n * prior) / (attempts + prior_n) if attempts + prior_n else prior


def usage_multiplier(book: MatchupBook, player: int, teammates: list[int], opponents: list[int]) -> float:
    """How this player's share of the ball moves with these four teammates."""
    base = book.overall.get(player)
    if base is None or base[0] < DEF_MIN or base[1] <= 0:
        return 1.0
    base_rate = base[1] / base[0]
    logs = []
    bucket = book.pairs.get(player, {})
    for mate in teammates:
        pair = bucket.get(mate)
        if pair is None:
            continue
        poss, used = pair
        rate = (used + 100.0 * base_rate) / (poss + 100.0)
        logs.append(math.log(_clamp(rate / base_rate, 0.45, 1.8)))
    mult = math.exp(0.5 * sum(logs) / len(logs)) if logs else 1.0
    mates_key = tuple(sorted(teammates))
    opp_key = tuple(sorted(opponents))
    cell = book.exact.get((player, mates_key, opp_key))
    if cell is not None and cell[0] >= EXACT_MIN:
        exact_rate = (cell[1] + 40.0 * base_rate) / (cell[0] + 40.0)
        exact_mult = _clamp(exact_rate / base_rate, 0.45, 1.8)
        weight = cell[0] / (cell[0] + 80.0)
        mult = (1.0 - weight) * mult + weight * exact_mult
    return _clamp(mult, 0.62, 1.55)


def _defense_fg(book: MatchupBook, opponents: list[int], is3: bool) -> float:
    num = 0.0
    den = 0.0
    league = book.league_fg3 if is3 else book.league_fg2
    for pid in opponents:
        row = book.defense.get(pid)
        if row is None:
            continue
        attempts = row[3] if is3 else row[1]
        makes = row[4] if is3 else row[2]
        if attempts < 40:
            continue
        rate = _rate(makes, attempts, league, 120.0)
        weight = min(attempts, 900)
        num += weight * (rate / league)
        den += weight
    if den <= 0:
        return 1.0
    # Each defender's number already includes the other four he usually plays with.
    return _clamp(1.0 + (num / den - 1.0) * 0.7, 0.90, 1.10)


def _exact_fg(book: MatchupBook, player: int, teammates: list[int], opponents: list[int], is3: bool) -> float:
    base = book.overall.get(player)
    cell = book.exact.get((player, tuple(sorted(teammates)), tuple(sorted(opponents))))
    if base is None or cell is None:
        return 1.0
    base_a, base_m = (base[4], base[5]) if is3 else (base[2], base[3])
    cell_a, cell_m = (cell[4], cell[5]) if is3 else (cell[2], cell[3])
    if base_a < 40 or cell_a < 8:
        return 1.0
    base_fg = base_m / base_a
    cell_fg = _rate(cell_m, cell_a, base_fg, 25.0)
    if base_fg <= 0.05:
        return 1.0
    weight = cell_a / (cell_a + 40.0)
    return _clamp(1.0 + weight * (cell_fg / base_fg - 1.0), 0.88, 1.12)


def shot_multiplier(book: MatchupBook, player: int, teammates: list[int], opponents: list[int], is3: bool) -> float:
    """Make-probability multiplier for this shooter against these five defenders."""
    return _clamp(_defense_fg(book, opponents, is3) * _exact_fg(book, player, teammates, opponents, is3), 0.86, 1.14)


def turnover_multiplier(book: MatchupBook, opponents: list[int]) -> float:
    """How often these five defenders force a turnover, relative to a league unit."""
    num = 0.0
    den = 0.0
    for pid in opponents:
        row = book.defense.get(pid)
        if row is None or row[0] < DEF_MIN:
            continue
        rate = _rate(row[5], row[0], book.league_tov, 200.0)
        weight = min(row[0], 1200)
        num += weight * (rate / book.league_tov if book.league_tov else 1.0)
        den += weight
    if den <= 0:
        return 1.0
    return _clamp(1.0 + (num / den - 1.0) * 0.7, 0.88, 1.14)
