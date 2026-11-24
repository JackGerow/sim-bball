"""Publish the player board and injury list that the site reads for a slate."""

from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path

import polars as pl

from simbball.props import _positions
from simbball.ratings import _read_player_logs
from simbball.reports import canonical_team, load_injuries, resolve_team
from simbball.teams import load_directory

_BODY = re.compile(r"\(([^)]+)\)")
_STATUS = {
    "out": "Out",
    "doubtful": "Doubtful",
    "questionable": "Questionable",
    "probable": "Probable",
}


def _slug(game: dict) -> str:
    folder = str(game.get("video") or "").replace("\\", "/").split("/")
    if len(folder) >= 2 and folder[-2]:
        return folder[-2]
    return f"{game['away']['abbrev']}_at_{game['home']['abbrev']}"


def _read_box(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows: dict[int, dict] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["person_id"])] = row
    return rows


def _line(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {
        "minutes": float(row["minutes"]),
        "pts": float(row["pts"]),
        "reb": float(row["reb"]),
        "ast": float(row["ast"]),
        "stl": float(row["stl"]),
        "blk": float(row["blk"]),
        "tov": float(row["tov"]),
        "pf": float(row["pf"]),
        "fgm": float(row["fgm"]),
        "fga": float(row["fga"]),
        "fg3m": float(row["fg3m"]),
        "fg3a": float(row["fg3a"]),
        "ftm": float(row["ftm"]),
        "fta": float(row["fta"]),
    }


def _abbrev(directory: dict, city: str, name: str) -> str:
    key = canonical_team(f"{city} {name}")
    for team in directory.values():
        if canonical_team(f"{team.city} {team.name}") == key:
            return team.abbrev
    return (name or city or "")[:3].upper()


def _recent(logs: pl.DataFrame, directory: dict) -> dict[int, list[dict]]:
    played = logs.filter(pl.col("numMinutes") > 0).sort("gameDateTimeEst")
    columns = [
        "personId",
        "gameDateTimeEst",
        "opponentteamCity",
        "opponentteamName",
        "numMinutes",
        "points",
        "reboundsTotal",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "foulsPersonal",
        "fieldGoalsMade",
        "fieldGoalsAttempted",
        "threePointersMade",
        "threePointersAttempted",
        "freeThrowsMade",
        "freeThrowsAttempted",
    ]
    by_player: dict[int, list[dict]] = {}
    for row in played.select(columns).iter_rows(named=True):
        by_player.setdefault(int(row["personId"]), []).append(row)
    recent: dict[int, list[dict]] = {}
    for person_id, games in by_player.items():
        last = games[-5:]
        last.reverse()
        recent[person_id] = [
            {
                "date": str(game["gameDateTimeEst"])[:10],
                "opponent": _abbrev(directory, str(game["opponentteamCity"] or ""), str(game["opponentteamName"] or "")),
                "minutes": round(float(game["numMinutes"] or 0), 1),
                "pts": int(game["points"] or 0),
                "reb": int(game["reboundsTotal"] or 0),
                "ast": int(game["assists"] or 0),
                "stl": int(game["steals"] or 0),
                "blk": int(game["blocks"] or 0),
                "tov": int(game["turnovers"] or 0),
                "pf": int(game["foulsPersonal"] or 0),
                "fgm": int(game["fieldGoalsMade"] or 0),
                "fga": int(game["fieldGoalsAttempted"] or 0),
                "fg3m": int(game["threePointersMade"] or 0),
                "fg3a": int(game["threePointersAttempted"] or 0),
                "ftm": int(game["freeThrowsMade"] or 0),
                "fta": int(game["freeThrowsAttempted"] or 0),
            }
            for game in last
        ]
    return recent


def _body(comment: str) -> str:
    match = _BODY.search(comment or "")
    if not match:
        return "Undisclosed"
    return match.group(1).strip().capitalize()


def _status(designation: str, status: str) -> str:
    label = (designation or status or "").strip().lower()
    if label in _STATUS:
        return _STATUS[label]
    text = (status or designation or "Out").strip()
    return text[:1].upper() + text[1:]


def _injuries(raw: Path, games: list[dict]) -> list[dict]:
    path = raw / "injuries.txt"
    if not path.exists():
        return []
    slate: dict[str, tuple[str, str, str]] = {}
    names: list[str] = []
    for game in games:
        slug = _slug(game)
        for side, opponent in (
            (game["home"], game["away"]["abbrev"]),
            (game["away"], game["home"]["abbrev"]),
        ):
            full = f"{side['city']} {side['name']}"
            names.append(full)
            slate[canonical_team(full)] = (side["abbrev"], opponent, slug)
    rows = []
    for injury in load_injuries(path):
        key = canonical_team(injury.team)
        found = slate.get(key)
        if found is None:
            resolved = resolve_team(injury.team, names)
            found = slate.get(canonical_team(resolved)) if resolved else None
        if found is None:
            continue
        team, opponent, slug = found
        rows.append(
            {
                "name": injury.name,
                "position": injury.position,
                "team": injury.team,
                "teamAbbr": team,
                "opponentAbbr": opponent,
                "slug": slug,
                "injury": _body(injury.comment),
                "status": _status(injury.designation, injury.status),
                "returnDate": injury.return_date,
                "detail": injury.comment,
            }
        )
    rows.sort(key=lambda row: (row["teamAbbr"], row["status"], row["name"]))
    return rows


def write_board(raw: Path, day: date, out_dir: Path) -> None:
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    games = summary.get("games") or []
    directory = load_directory(raw)
    logs = _read_player_logs(raw, day)
    positions = _positions(raw, logs)
    recent = _recent(logs, directory)
    players = []
    for game in games:
        slug = _slug(game)
        folder = out_dir / slug
        mean = _read_box(folder / "boxscore_mean.csv")
        mode_path = folder / "boxscore_mode.csv"
        common_kind = "average" if mode_path.exists() else "filmed"
        common = _read_box(mode_path if common_kind == "average" else folder / "boxscore_selected.csv")
        home = game["home"]["abbrev"]
        away = game["away"]["abbrev"]
        for row in mean.values():
            if float(row["minutes"]) < 8:
                continue
            person_id = int(row["person_id"])
            made = float(row["fgm"])
            attempted = float(row["fga"])
            team = row["team"]
            opponent = home if team == away else away
            players.append(
                {
                    "person_id": person_id,
                    "name": row["name"],
                    "team": team,
                    "opponent": opponent,
                    "slug": slug,
                    "game_id": game.get("game_id") or "",
                    "starter": str(row["starter"]) == "1",
                    "position": positions.get(person_id, ""),
                    "jersey": row["jersey"],
                    "fg_pct": round(made / attempted, 4) if attempted else 0.0,
                    "last5": recent.get(person_id, []),
                    "mean": _line(row),
                    "common": _line(common.get(person_id)),
                    "common_kind": common_kind,
                    "common_n": int(game.get("mode_count") or 0),
                }
            )
    players.sort(key=lambda row: (-row["fg_pct"], -row["mean"]["pts"], row["name"]))
    payload = {
        "date": day.isoformat(),
        "players": players,
        "injuries": _injuries(raw, games),
    }
    (out_dir / "board.json").write_text(json.dumps(payload), encoding="utf-8")
    above = sum(1 for row in players if row["fg_pct"] > 0.49)
    print(f"Player board  {day.isoformat()}  {len(players)} projected  {above} above 49% FG", flush=True)
