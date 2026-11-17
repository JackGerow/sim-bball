"""Build opponent-adjusted team and player rates from data/raw."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from simbball.reports import Injury, canonical_team, load_injuries, match_players
from simbball.teams import TeamInfo, load_directory

# Games of league-average prior mixed into team ratings. The 2026-27
# sample before a given night is only ~10 games, so this keeps one hot
# week from dominating the projection. Team offense, defense, and pace
# stay on this season: last year's roster is a different team.
TEAM_PRIOR_GAMES = 6
# Rookies, and anyone without a prior-season sample, shrink toward the
# league by this many minutes.
PLAYER_PRIOR_MIN = 96.0
DECAY = 0.90
# Veteran skills shrink toward the player's own 2023-24 through 2025-26
# line. Recent seasons count more. Going further back starts describing
# a different player.
HISTORY_WINDOWS = (
    ("2025-10-01", 1.00),
    ("2024-10-01", 0.55),
    ("2023-10-01", 0.30),
)


@dataclass
class Player:
    person_id: int
    first: str
    last: str
    jersey: str
    usage: float
    p3: float
    fg2: float
    fg3: float
    ft: float
    oreb: float
    dreb: float
    ast: float
    stl: float
    blk: float
    pf: float
    target_sec: float
    starter: bool
    guard: bool
    center: bool

    @property
    def last_upper(self) -> str:
        return self.last.upper()


@dataclass
class TeamModel:
    team_id: int
    city: str
    name: str
    abbrev: str
    primary: tuple[int, int, int]
    secondary: tuple[int, int, int]
    ink: tuple[int, int, int]
    players: list[Player]
    starter_idx: list[int]
    ortg: float
    drtg: float
    pace: float
    p_tov: float
    p_sfl: float
    p_shot: float
    oreb_share: float
    dreb_share: float
    blk_per_miss: float
    steal_share: float
    tov_forced: float
    games: int


@dataclass
class League:
    ppg: float
    poss: float
    ortg: float
    fga: float
    fg3a: float
    fta: float
    tov: float
    oreb: float
    fg2: float
    fg3: float
    ft: float
    p3: float
    oreb_share: float
    dreb_share: float
    blk_per_miss: float
    steal_share: float
    tov_forced: float
    home_court: float
    score_sd: float
    matchups: object | None = None


@dataclass
class ScheduledGame:
    game_id: str
    tip_est: str
    arena: str
    home_id: int
    away_id: int


@dataclass
class World:
    as_of: date
    league: League
    teams: dict[int, TeamModel]
    games: list[ScheduledGame]


def _clean_jersey(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _flag(value) -> bool:
    return str(value).strip() in {"1", "1.0"}


def _is_out(comment) -> bool:
    if comment is None:
        return False
    text = str(comment).strip()
    if not text or "Rest" in text or text.startswith("DNP"):
        return False
    return True


def _shrink(value: float, sample: float, prior: float, prior_n: float) -> float:
    return (value + prior_n * prior) / (sample + prior_n)


def load_schedule(raw: Path, day: date) -> list[ScheduledGame]:
    key = day.isoformat()
    found: dict[str, ScheduledGame] = {}
    for name in (
        "LeagueSchedule26_27.csv",
        "LeagueSchedule25_26.csv",
        "LeagueSchedule24_25.csv",
    ):
        path = raw / name
        if not path.exists():
            continue
        frame = pl.read_csv(path, infer_schema_length=5000, ignore_errors=True)
        lower = {c.lower(): c for c in frame.columns}
        date_col = lower.get("gamedatetimeest")
        id_col = lower.get("gameid")
        if not date_col or not id_col:
            continue
        home_col = lower.get("hometeamid") or lower.get("hometeamid")
        away_col = lower.get("awayteamid") or lower.get("awayteamid")
        if home_col is None or away_col is None:
            continue
        home_city = lower.get("hometeamcity")
        home_name = lower.get("hometeamname")
        arena = lower.get("arenaname")
        city = lower.get("arenacity")
        for row in frame.iter_rows(named=True):
            stamp = str(row[date_col])
            if not stamp.startswith(key):
                continue
            game_id = str(row[id_col])
            arena_bits = []
            if arena and row.get(arena):
                arena_bits.append(str(row[arena]))
            if city and row.get(city):
                arena_bits.append(str(row[city]))
            found[game_id] = ScheduledGame(
                game_id=game_id,
                tip_est=stamp,
                arena=", ".join(arena_bits),
                home_id=int(row[home_col]),
                away_id=int(row[away_col]),
            )
    games = list(found.values())
    games.sort(key=lambda g: g.tip_est)
    return games


def _read_player_logs(raw: Path, day: date) -> pl.DataFrame:
    numeric = [
        "points",
        "assists",
        "blocks",
        "steals",
        "fieldGoalsAttempted",
        "fieldGoalsMade",
        "threePointersAttempted",
        "threePointersMade",
        "freeThrowsAttempted",
        "freeThrowsMade",
        "reboundsDefensive",
        "reboundsOffensive",
        "reboundsTotal",
        "foulsPersonal",
        "turnovers",
        "home",
        "win",
    ]
    overrides = {col: pl.Float64 for col in numeric}
    overrides.update(
        {
            "numMinutes": pl.Utf8,
            "plusMinusPoints": pl.Utf8,
            "fieldGoalsPercentage": pl.Utf8,
            "threePointersPercentage": pl.Utf8,
            "freeThrowsPercentage": pl.Utf8,
            "comment": pl.Utf8,
            "startingPosition": pl.Utf8,
        }
    )
    logs = pl.scan_csv(
        raw / "PlayerStatistics.csv",
        schema_overrides=overrides,
        ignore_errors=True,
    ).filter(
        (pl.col("gameType") == "Regular Season")
        & (pl.col("gameDateTimeEst") >= "2026-10-01")
        & (pl.col("gameDateTimeEst") < day.isoformat())
    ).with_columns(
        pl.col("numMinutes").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("comment").fill_null(""),
        pl.col("startingPosition").fill_null(""),
    ).collect()
    stat_cols = [c for c in numeric if c not in {"home", "win"}]
    logs = logs.with_columns([pl.col(c).fill_null(0.0) for c in stat_cols])
    return logs


def _league_player_priors(league: League) -> dict[str, float]:
    return {
        "usage": (league.fga + 0.44 * league.fta + league.tov) / 240.0,
        "p3": league.p3,
        "fg2": league.fg2,
        "fg3": league.fg3,
        "ft": league.ft,
        "oreb": league.oreb / 240.0,
        "dreb": 32.0 / 240.0,
        "ast": 26.0 / 240.0,
        "stl": 8.4 / 240.0,
        "blk": 5.0 / 240.0,
        "pf": 20.0 / 240.0,
    }


def _history_priors(raw: Path, league: League) -> dict[int, dict]:
    """Personal skill prior from the three seasons before 2026-27.

    Shooting and per-minute rates are regressed toward this season's league
    average, so a short prior sample cannot overwhelm a real one.
    """
    numeric = {
        "fieldGoalsAttempted": "fga",
        "fieldGoalsMade": "fgm",
        "threePointersAttempted": "fg3a",
        "threePointersMade": "fg3m",
        "freeThrowsAttempted": "fta",
        "freeThrowsMade": "ftm",
        "reboundsOffensive": "oreb",
        "reboundsDefensive": "dreb",
        "assists": "ast",
        "steals": "stl",
        "blocks": "blk",
        "turnovers": "tov",
        "foulsPersonal": "pf",
    }
    overrides = {col: pl.Float64 for col in numeric}
    overrides.update({"numMinutes": pl.Utf8, "personId": pl.Int64, "gameDateTimeEst": pl.Utf8, "gameType": pl.Utf8})
    weight = pl.when(pl.col("gameDateTimeEst") >= HISTORY_WINDOWS[0][0]).then(HISTORY_WINDOWS[0][1])
    for start, value in HISTORY_WINDOWS[1:]:
        weight = weight.when(pl.col("gameDateTimeEst") >= start).then(value)
    weight = weight.otherwise(HISTORY_WINDOWS[-1][1])
    frame = (
        pl.scan_csv(raw / "PlayerStatistics.csv", schema_overrides=overrides, ignore_errors=True)
        .filter(
            (pl.col("gameType") == "Regular Season")
            & (pl.col("gameDateTimeEst") >= HISTORY_WINDOWS[-1][0])
            & (pl.col("gameDateTimeEst") < "2026-10-01")
        )
        .with_columns(
            pl.col("numMinutes").cast(pl.Float64, strict=False).fill_null(0.0).alias("numMinutes"),
            weight.alias("w"),
            *[pl.col(src).fill_null(0.0) for src in numeric],
        )
        .filter(pl.col("numMinutes") > 0)
        .group_by("personId")
        .agg(
            [(pl.col("w") * pl.col("numMinutes")).sum().alias("min")]
            + [(pl.col("w") * pl.col(src)).sum().alias(dst) for src, dst in numeric.items()]
        )
        .collect()
    )
    league_prior = _league_player_priors(league)
    priors: dict[int, dict] = {}
    for row in frame.iter_rows(named=True):
        minutes = float(row["min"])
        if minutes <= 0:
            continue
        anchor = 400.0
        fga = float(row["fga"])
        fg3a = float(row["fg3a"])
        fta = float(row["fta"])
        fg2a = max(fga - fg3a, 0.0)
        fg2m = float(row["fgm"]) - float(row["fg3m"])
        usage_events = fga + 0.44 * fta + float(row["tov"])

        def per_minute(total: float, key: str) -> float:
            return (total + anchor * league_prior[key]) / (minutes + anchor)

        priors[int(row["personId"])] = {
            "minutes": minutes,
            "usage": (usage_events + anchor * league_prior["usage"]) / (minutes + anchor),
            "p3": _shrink(fg3a, fga, league_prior["p3"], 80.0),
            "fg2": _shrink(fg2m, fg2a, league_prior["fg2"], 100.0),
            "fg3": _shrink(float(row["fg3m"]), fg3a, league_prior["fg3"], 80.0),
            "ft": _shrink(float(row["ftm"]), fta, league_prior["ft"], 40.0),
            "oreb": per_minute(float(row["oreb"]), "oreb"),
            "dreb": per_minute(float(row["dreb"]), "dreb"),
            "ast": per_minute(float(row["ast"]), "ast"),
            "stl": per_minute(float(row["stl"]), "stl"),
            "blk": per_minute(float(row["blk"]), "blk"),
            "pf": per_minute(float(row["pf"]), "pf"),
        }
    return priors


def _team_games(logs: pl.DataFrame) -> pl.DataFrame:
    grouped = logs.group_by(["gameId", "playerteamId"]).agg(
        [
            pl.col("gameDateTimeEst").first(),
            pl.col("opponentteamId").first(),
            pl.col("home").max(),
            pl.col("playerteamCity").first(),
            pl.col("playerteamName").first(),
            pl.col("numMinutes").sum().alias("minutes"),
            pl.col("points").sum().alias("pts"),
            pl.col("fieldGoalsAttempted").sum().alias("fga"),
            pl.col("fieldGoalsMade").sum().alias("fgm"),
            pl.col("threePointersAttempted").sum().alias("fg3a"),
            pl.col("threePointersMade").sum().alias("fg3m"),
            pl.col("freeThrowsAttempted").sum().alias("fta"),
            pl.col("freeThrowsMade").sum().alias("ftm"),
            pl.col("reboundsOffensive").sum().alias("oreb"),
            pl.col("reboundsDefensive").sum().alias("dreb"),
            pl.col("assists").sum().alias("ast"),
            pl.col("steals").sum().alias("stl"),
            pl.col("blocks").sum().alias("blk"),
            pl.col("turnovers").sum().alias("tov"),
            pl.col("foulsPersonal").sum().alias("pf"),
        ]
    )
    opp = grouped.select(
        [
            pl.col("gameId"),
            pl.col("playerteamId").alias("opponentteamId"),
            pl.col("pts").alias("opp_pts"),
            pl.col("fga").alias("opp_fga"),
            pl.col("fgm").alias("opp_fgm"),
            pl.col("fg3a").alias("opp_fg3a"),
            pl.col("fg3m").alias("opp_fg3m"),
            pl.col("fta").alias("opp_fta"),
            pl.col("oreb").alias("opp_oreb"),
            pl.col("dreb").alias("opp_dreb"),
            pl.col("tov").alias("opp_tov"),
            pl.col("minutes").alias("opp_minutes"),
        ]
    )
    return grouped.join(opp, on=["gameId", "opponentteamId"], how="inner")


def _league_from_games(games: pl.DataFrame) -> tuple[League, dict]:
    g = games.height
    def mean(col: str) -> float:
        return float(games[col].mean())

    fga = mean("fga")
    fgm = mean("fgm")
    fg3a = mean("fg3a")
    fg3m = mean("fg3m")
    fta = mean("fta")
    ftm = mean("ftm")
    oreb = mean("oreb")
    dreb = mean("dreb")
    tov = mean("tov")
    pts = mean("pts")
    minutes = mean("minutes")
    opp_dreb = mean("opp_dreb")
    opp_oreb = mean("opp_oreb")
    opp_fga = mean("opp_fga")
    opp_fgm = mean("opp_fgm")
    opp_tov = mean("opp_tov")
    opp_minutes = mean("opp_minutes")
    blk = mean("blk")
    stl = mean("stl")
    poss = fga - oreb + tov + 0.44 * fta
    pace = poss * 240.0 / max(minutes, 1.0)
    misses = max(fga - fgm, 1.0)
    opp_misses = max(opp_fga - opp_fgm, 1.0)
    opp_poss = opp_fga - opp_oreb + opp_tov + 0.44 * mean("opp_fta")
    live_oreb = oreb + opp_dreb
    live_dreb = dreb + opp_oreb
    fg2a = max(fga - fg3a, 1.0)
    fg2m = fgm - fg3m
    home = games.filter(pl.col("home") == 1)
    if home.height:
        home_court = float((home["pts"] - home["opp_pts"]).mean())
    else:
        home_court = 1.8
    league = League(
        ppg=pts,
        poss=pace,
        ortg=pts / poss * 100.0,
        fga=fga * 240.0 / max(minutes, 1.0),
        fg3a=fg3a * 240.0 / max(minutes, 1.0),
        fta=fta * 240.0 / max(minutes, 1.0),
        tov=tov * 240.0 / max(minutes, 1.0),
        oreb=oreb * 240.0 / max(minutes, 1.0),
        fg2=fg2m / fg2a,
        fg3=fg3m / max(fg3a, 1.0),
        ft=ftm / max(fta, 1.0),
        p3=fg3a / max(fga, 1.0),
        oreb_share=oreb / max(live_oreb, 1.0),
        dreb_share=dreb / max(live_dreb, 1.0),
        blk_per_miss=blk / opp_misses,
        steal_share=stl / max(tov, 1.0),
        tov_forced=opp_tov / max(opp_poss, 1.0),
        home_court=home_court,
        score_sd=12.0,
    )
    raw = {
        "poss": poss,
        "trips": poss + oreb,
        "fga": fga,
        "fg3a": fg3a,
        "fta": fta,
        "tov": tov,
        "oreb": oreb,
        "minutes": minutes,
        "pts": pts,
        "opp_pts": mean("opp_pts"),
        "opp_poss": opp_poss,
        "opp_minutes": opp_minutes,
    }
    return league, raw


def _weighted_team_row(rows: list[dict], league: League, league_raw: dict) -> dict:
    rows = sorted(rows, key=lambda r: r["gameDateTimeEst"], reverse=True)
    weights = [DECAY ** i for i in range(len(rows))]
    wsum = sum(weights) or 1.0

    def wmean(key: str) -> float:
        return sum(w * float(r[key]) for w, r in zip(weights, rows)) / wsum

    minutes = wmean("minutes")
    scale = 240.0 / max(minutes, 1.0)
    fga = wmean("fga") * scale
    fgm = wmean("fgm") * scale
    fg3a = wmean("fg3a") * scale
    fg3m = wmean("fg3m") * scale
    fta = wmean("fta") * scale
    ftm = wmean("ftm") * scale
    oreb = wmean("oreb") * scale
    dreb = wmean("dreb") * scale
    tov = wmean("tov") * scale
    pts = wmean("pts") * scale
    stl = wmean("stl") * scale
    blk = wmean("blk") * scale
    opp_pts = wmean("opp_pts") * (240.0 / max(wmean("opp_minutes"), 1.0))
    opp_fga = wmean("opp_fga") * scale
    opp_fgm = wmean("opp_fgm") * scale
    opp_oreb = wmean("opp_oreb") * scale
    opp_dreb = wmean("opp_dreb") * scale
    opp_tov = wmean("opp_tov") * scale
    opp_fta = wmean("opp_fta") * scale
    poss = fga - oreb + tov + 0.44 * fta
    opp_poss = opp_fga - opp_oreb + opp_tov + 0.44 * opp_fta
    k = TEAM_PRIOR_GAMES
    poss_s = _shrink(poss, 1.0, league.poss, k)  # already per game; mix k games
    # _shrink(value, sample=1, prior, k) treats the weighted mean as one observation.
    # Use the real game count as the sample size.
    n = float(len(rows))
    poss_s = _shrink(poss * n, n, league.poss, k) / 1.0
    poss_s = (poss * n + league.poss * k) / (n + k)
    pts_s = (pts * n + league.ppg * k) / (n + k)
    opp_pts_s = (opp_pts * n + league.ppg * k) / (n + k)
    opp_poss_s = (opp_poss * n + league.poss * k) / (n + k)
    fga_s = (fga * n + league.fga * k) / (n + k)
    fg3a_s = (fg3a * n + league.fg3a * k) / (n + k)
    fta_s = (fta * n + league.fta * k) / (n + k)
    tov_s = (tov * n + league.tov * k) / (n + k)
    oreb_s = (oreb * n + league.oreb * k) / (n + k)
    live_oreb = oreb + opp_dreb
    live_dreb = dreb + opp_oreb
    oreb_share = _shrink(oreb, live_oreb, league.oreb_share, 80.0)
    dreb_share = _shrink(dreb, live_dreb, league.dreb_share, 80.0)
    misses_against = max(opp_fga - opp_fgm, 1.0)
    blk_per_miss = _shrink(blk, misses_against, league.blk_per_miss, 40.0)
    # Share of opponent turnovers that become steals.
    steal_share = _shrink(stl, max(opp_tov, 1.0), league.steal_share, 30.0)
    tov_forced = (opp_tov * n + league.tov_forced * league.poss * k) / max(opp_poss * n + league.poss * k, 1.0)
    trips = poss_s + oreb_s
    # Shooting-foul trips are the FTA that are not and-1s. About 3 and-1s
    # per game leave the rest as 2-shot (or rare 3-shot) trips.
    sfl_trips = max((fta_s - 3.0) / 2.08, 1.0)
    p_tov = tov_s / trips
    p_sfl = sfl_trips / trips
    p_shot = fga_s / trips
    total = p_tov + p_sfl + p_shot
    return {
        "ortg": pts_s / poss_s * 100.0,
        "drtg": opp_pts_s / opp_poss_s * 100.0,
        "pace": poss_s,
        "p_tov": p_tov / total,
        "p_sfl": p_sfl / total,
        "p_shot": p_shot / total,
        "oreb_share": oreb_share,
        "dreb_share": dreb_share,
        "blk_per_miss": blk_per_miss,
        "steal_share": steal_share,
        "tov_forced": tov_forced,
        "games": len(rows),
        "fg2": (fgm - fg3m) / max(fga - fg3a, 1.0),
        "fg3": fg3m / max(fg3a, 1.0),
        "ft": ftm / max(fta, 1.0),
        "p3": fg3a / max(fga, 1.0),
        "city": rows[0]["playerteamCity"],
        "name": rows[0]["playerteamName"],
    }


def _player_pool(
    logs: pl.DataFrame,
    team_id: int,
    roster_rows: list[dict],
    info: dict,
    career: dict[int, dict],
    held_out: set[int] | None = None,
    cleared: set[int] | None = None,
) -> list[Player]:
    held_out = held_out or set()
    cleared = cleared or set()
    team_logs = logs.filter(pl.col("playerteamId") == team_id).sort("gameDateTimeEst")
    games = (
        team_logs.select(["gameId", "gameDateTimeEst"])
        .unique()
        .sort("gameDateTimeEst")
    )
    game_ids = games["gameId"].to_list()
    last_ids = set(game_ids[-8:])
    by_player: dict[int, list[dict]] = {}
    for row in team_logs.iter_rows(named=True):
        by_player.setdefault(int(row["personId"]), []).append(row)

    # A player ruled back in may have missed the last game entirely.
    present = {int(row["personId"]) for row in roster_rows}
    roster = list(roster_rows)
    for person_id, games_played in by_player.items():
        if person_id in cleared and person_id not in present and person_id not in held_out:
            roster.append(games_played[-1])

    players: list[Player] = []
    for row in roster:
        pid = int(row["personId"])
        if pid in held_out:
            continue
        if _is_out(row["comment"]) and pid not in cleared:
            continue
        season_logs = by_player.get(pid, [])
        rate_num = {k: 0.0 for k in ("min", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "pts")}
        rate_w = 0.0
        minute_samples: list[float] = []
        # Walk team games from newest so the decay matches the team model.
        order = {gid: i for i, gid in enumerate(reversed(game_ids))}
        for game in season_logs:
            if game["gameId"] not in last_ids and float(game["numMinutes"]) <= 0:
                continue
            if game["gameId"] in last_ids and not _is_out(game["comment"]):
                minute_samples.append(float(game["numMinutes"]))
            if float(game["numMinutes"]) <= 0 or _is_out(game["comment"]):
                continue
            w = DECAY ** order.get(game["gameId"], 8)
            rate_w += w
            mins = float(game["numMinutes"])
            rate_num["min"] += w * mins
            for src, dst in (
                ("fieldGoalsAttempted", "fga"),
                ("fieldGoalsMade", "fgm"),
                ("threePointersAttempted", "fg3a"),
                ("threePointersMade", "fg3m"),
                ("freeThrowsAttempted", "fta"),
                ("freeThrowsMade", "ftm"),
                ("reboundsOffensive", "oreb"),
                ("reboundsDefensive", "dreb"),
                ("assists", "ast"),
                ("steals", "stl"),
                ("blocks", "blk"),
                ("turnovers", "tov"),
                ("foulsPersonal", "pf"),
                ("points", "pts"),
            ):
                rate_num[dst] += w * float(game[src])
        if pid in cleared and not minute_samples:
            for game in reversed(season_logs):
                if float(game["numMinutes"]) > 0 and not _is_out(game["comment"]):
                    minute_samples.append(float(game["numMinutes"]))
                if len(minute_samples) >= 8:
                    break
        if not minute_samples and float(row["numMinutes"]) <= 0:
            continue
        target = sum(minute_samples) / len(minute_samples) if minute_samples else float(row["numMinutes"])
        minutes = rate_num["min"]
        started = str(row.get("startingPosition") or "").strip() != ""
        if pid in cleared and not started:
            for game in reversed(season_logs):
                if float(game["numMinutes"]) > 0 and not _is_out(game["comment"]):
                    started = str(game.get("startingPosition") or "").strip() != ""
                    break
        players.append(
            _make_player(row, info, rate_num, minutes, target, started, career.get(pid))
        )
    return players


def _make_player(row: dict, info: dict, num: dict, minutes: float, target_min: float, started: bool, skill: dict | None) -> Player:
    fga = num["fga"]
    fg3a = num["fg3a"]
    fta = num["fta"]
    fg2a = max(fga - fg3a, 0.0)
    fg2m = num["fgm"] - num["fg3m"]
    usage_events = fga + 0.44 * fta + num["tov"]
    # A veteran shrinks toward his own prior seasons. Usage uses a shorter
    # memory than shooting, because role changes faster than touch.
    if skill is not None:
        prior = skill
        minute_n = 480.0
        usage_n = 160.0
        n_p3, n_fg2, n_fg3, n_ft = 150.0, 180.0, 120.0, 80.0
    else:
        prior = info["priors"]
        minute_n = PLAYER_PRIOR_MIN
        usage_n = PLAYER_PRIOR_MIN
        n_p3, n_fg2, n_fg3, n_ft = 30.0, 60.0, 40.0, 25.0

    def per_minute(total: float, key: str, anchor: float) -> float:
        return (total + anchor * prior[key]) / (minutes + anchor)

    return Player(
        person_id=int(row["personId"]),
        first=str(row["firstName"]),
        last=str(row["lastName"]),
        jersey="",
        usage=max(0.02, per_minute(usage_events, "usage", usage_n)),
        p3=_shrink(fg3a, fga, prior["p3"], n_p3),
        fg2=_shrink(fg2m, fg2a, prior["fg2"], n_fg2),
        fg3=_shrink(num["fg3m"], fg3a, prior["fg3"], n_fg3),
        ft=_shrink(num["ftm"], fta, prior["ft"], n_ft),
        oreb=max(0.01, per_minute(num["oreb"], "oreb", minute_n)),
        dreb=max(0.02, per_minute(num["dreb"], "dreb", minute_n)),
        ast=max(0.01, per_minute(num["ast"], "ast", minute_n)),
        stl=max(0.01, per_minute(num["stl"], "stl", minute_n)),
        blk=max(0.005, per_minute(num["blk"], "blk", minute_n)),
        pf=max(0.02, per_minute(num["pf"], "pf", minute_n)),
        target_sec=max(0.0, target_min * 60.0),
        starter=started,
        guard=_flag(row.get("_guard")),
        center=_flag(row.get("_center")),
    )


def _finish_roster(players: list[Player], jerseys: dict[int, str], names: dict[int, tuple[str, bool, bool]]) -> tuple[list[Player], list[int]]:
    for player in players:
        first, guard, center = names.get(player.person_id, (player.first, player.guard, player.center))
        player.first = first or player.first
        player.guard = guard
        player.center = center
        player.jersey = jerseys.get(player.person_id, "")
    starters = [p for p in players if p.starter]
    if len(starters) < 5:
        rest = sorted((p for p in players if not p.starter), key=lambda p: p.target_sec, reverse=True)
        starters = starters + rest[: 5 - len(starters)]
    if len(starters) > 5:
        starters = sorted(starters, key=lambda p: p.target_sec, reverse=True)[:5]
    starter_ids = {p.person_id for p in starters}
    for player in players:
        player.starter = player.person_id in starter_ids
    pool = [p for p in players if p.starter or p.target_sec >= 4 * 60]
    if len(pool) < 8:
        pool = sorted(players, key=lambda p: p.target_sec, reverse=True)[:8]
    if len(pool) < 5:
        raise RuntimeError("Rotation has fewer than 5 available players")
    if len(pool) > 11:
        must = [p for p in pool if p.starter]
        bench = sorted((p for p in pool if not p.starter), key=lambda p: p.target_sec, reverse=True)
        pool = must + bench[: 11 - len(must)]
    # Injured stars leave a minutes hole. Spread it across the rotation.
    total = sum(p.target_sec for p in pool)
    if total <= 0:
        for player in pool:
            player.target_sec = 240.0 * 60.0 / len(pool)
    else:
        scale = (240.0 * 60.0) / total
        for player in pool:
            player.target_sec *= scale
    cap = 38.0 * 60.0
    for _ in range(8):
        over = [player for player in pool if player.target_sec > cap + 0.5]
        if not over:
            break
        extra = sum(player.target_sec - cap for player in over)
        for player in over:
            player.target_sec = cap
        under = [player for player in pool if player.target_sec < cap - 1.0]
        room = sum(cap - player.target_sec for player in under)
        if not under or room <= 0:
            break
        for player in under:
            player.target_sec += extra * (cap - player.target_sec) / room
    used: set[str] = set()
    for player in pool:
        jersey = player.jersey
        if not jersey or jersey in used:
            n = player.person_id % 98 + 1
            while str(n) in used:
                n = n % 99 + 1
            jersey = str(n)
        player.jersey = jersey
        used.add(jersey)

    def starter_key(player: Player):
        role = 0 if player.guard and not player.center else (2 if player.center and not player.guard else 1)
        return (role, -player.usage)

    starter_players = sorted((p for p in pool if p.starter), key=starter_key)
    bench_players = sorted((p for p in pool if not p.starter), key=lambda p: p.target_sec, reverse=True)
    ordered = starter_players + bench_players
    index = {p.person_id: i for i, p in enumerate(ordered)}
    starter_idx = [index[p.person_id] for p in starter_players]
    return ordered, starter_idx


def build_world(raw: Path, day: date | None = None) -> World:
    day = day or datetime.now().astimezone().date()
    directory = load_directory(raw)
    schedule = [g for g in load_schedule(raw, day) if g.home_id in directory and g.away_id in directory]
    logs = _read_player_logs(raw, day)
    games = _team_games(logs)
    league, league_raw = _league_from_games(games)
    from simbball.matchups import build_matchup_book

    league.matchups = build_matchup_book(raw)
    history = _history_priors(raw, league)
    players_file = pl.read_csv(
        raw / "Players.csv",
        schema_overrides={
            "personId": pl.Int64,
            "jersey": pl.Utf8,
            "guard": pl.Utf8,
            "forward": pl.Utf8,
            "center": pl.Utf8,
            "firstName": pl.Utf8,
        },
        ignore_errors=True,
    )
    jersey_map: dict[int, str] = {}
    name_map: dict[int, tuple[str, bool, bool]] = {}
    for row in players_file.iter_rows(named=True):
        pid = int(row["personId"])
        jersey_map[pid] = _clean_jersey(row.get("jersey"))
        name_map[pid] = (str(row.get("firstName") or ""), _flag(row.get("guard")), _flag(row.get("center")))

    needed = {g.home_id for g in schedule} | {g.away_id for g in schedule}
    injury_path = raw / "injuries.txt"
    injuries_by_team: dict[str, list[Injury]] = {}
    if injury_path.exists():
        for injury in load_injuries(injury_path):
            injuries_by_team.setdefault(canonical_team(injury.team), []).append(injury)
    names_by_team: dict[int, list[tuple[int, str, str]]] = {}
    for row in logs.select(["playerteamId", "personId", "firstName", "lastName"]).unique().iter_rows(named=True):
        names_by_team.setdefault(int(row["playerteamId"]), []).append(
            (int(row["personId"]), str(row["firstName"] or ""), str(row["lastName"] or ""))
        )
    ruled_out: list[str] = []
    ruled_in: list[str] = []
    unmatched: list[str] = []
    # Ratings for every NBA team keep an opponent lookup available even if
    # the slate is short. Building all 30 is cheap once the logs are in memory.
    if not needed:
        needed = set(directory)
    models: dict[int, TeamModel] = {}
    team_ids = games["playerteamId"].unique().to_list()
    rows_by_team: dict[int, list[dict]] = {}
    for row in games.iter_rows(named=True):
        rows_by_team.setdefault(int(row["playerteamId"]), []).append(row)

    for team_id in team_ids:
        team_id = int(team_id)
        if team_id not in directory and team_id not in needed:
            continue
        info_row = _weighted_team_row(rows_by_team[team_id], league, league_raw)
        info = TeamInfo(
            team_id=team_id,
            city=str(info_row["city"]),
            name=str(info_row["name"]),
            abbrev=directory[team_id].abbrev if team_id in directory else "NBA",
            primary=directory[team_id].primary if team_id in directory else (40, 70, 110),
            secondary=directory[team_id].secondary if team_id in directory else (200, 200, 200),
            ink=directory[team_id].ink if team_id in directory else (255, 255, 255),
        )
        if team_id not in needed:
            continue
        last_game = max(rows_by_team[team_id], key=lambda r: r["gameDateTimeEst"])["gameId"]
        roster_rows = logs.filter(
            (pl.col("playerteamId") == team_id) & (pl.col("gameId") == last_game)
        ).to_dicts()
        for row in roster_rows:
            meta = name_map.get(int(row["personId"]))
            if meta:
                row["_guard"] = "1" if meta[1] else "0"
                row["_center"] = "1" if meta[2] else "0"
            else:
                row["_guard"] = "0"
                row["_center"] = "0"
        info_row["priors"] = _league_player_priors(league)
        team_key = canonical_team(f"{info.city} {info.name}")
        matched, missed = match_players(injuries_by_team.get(team_key, []), names_by_team.get(team_id, []))
        held_out = {pid for pid, injury in matched.items() if injury.unavailable}
        cleared = {pid for pid, injury in matched.items() if not injury.unavailable}
        for injury in missed:
            unmatched.append(f"{info.abbrev} {injury.name}")
        on_last = {int(row["personId"]): row for row in roster_rows}
        roster = _player_pool(logs, team_id, roster_rows, info_row, history, held_out, cleared)
        ordered, starter_idx = _finish_roster(roster, jersey_map, name_map)
        playing_ids = {player.person_id for player in ordered}
        for pid, injury in matched.items():
            label = f"{info.abbrev} {injury.name} ({injury.designation or injury.status})"
            row = on_last.get(pid)
            played_last = row is not None and float(row["numMinutes"] or 0) > 0
            was_out = row is None or _is_out(row["comment"])
            if injury.unavailable and played_last:
                ruled_out.append(label)
            elif not injury.unavailable and was_out and pid in playing_ids:
                ruled_in.append(label)
        last_names: dict[str, int] = {}
        for player in ordered:
            last_names[player.last] = last_names.get(player.last, 0) + 1
        for player in ordered:
            if last_names[player.last] > 1 and player.first:
                player.last = f"{player.first[:1]}. {player.last}"
        models[team_id] = TeamModel(
            team_id=team_id,
            city=info.city,
            name=info.name,
            abbrev=info.abbrev,
            primary=info.primary,
            secondary=info.secondary,
            ink=info.ink,
            players=ordered,
            starter_idx=starter_idx,
            ortg=info_row["ortg"],
            drtg=info_row["drtg"],
            pace=info_row["pace"],
            p_tov=info_row["p_tov"],
            p_sfl=info_row["p_sfl"],
            p_shot=info_row["p_shot"],
            oreb_share=info_row["oreb_share"],
            dreb_share=info_row["dreb_share"],
            blk_per_miss=info_row["blk_per_miss"],
            steal_share=info_row["steal_share"],
            tov_forced=info_row["tov_forced"],
            games=info_row["games"],
        )
    if ruled_out or ruled_in or unmatched:
        print("Injury report  out and doubtful sit; questionable and probable play", flush=True)
        if ruled_out:
            print("  sitting   " + ", ".join(ruled_out), flush=True)
        if ruled_in:
            print("  playing   " + ", ".join(ruled_in), flush=True)
        if unmatched:
            print("  unmatched " + ", ".join(unmatched), flush=True)
    return World(as_of=day, league=league, teams=models, games=schedule)
