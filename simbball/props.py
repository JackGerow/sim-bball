"""Prop lines from the filmed box score, the season average, and the depth chart."""

from __future__ import annotations

import csv
import json
import math
from datetime import date
from pathlib import Path

import polars as pl

from simbball.ratings import _read_player_logs
from simbball.reports import canonical_team, load_depth_charts, match_person, resolve_team
from simbball.teams import load_directory

STAT_FIELDS = (
    ("pts", "points", "Points"),
    ("reb", "reboundsTotal", "Rebounds"),
    ("ast", "assists", "Assists"),
    ("fg3m", "threePointersMade", "Threes"),
    ("stl", "steals", "Steals"),
    ("blk", "blocks", "Blocks"),
)
POSITION_ORDER = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4}
GUARDS = {"PG", "SG"}
WINGS = {"PG", "SG", "SF"}


def _snap(projection: float, cushion: float, start: int, step: int) -> int | None:
    """Largest line that sits at least `cushion` under the projection."""
    room = projection - cushion
    if room < start:
        return None
    steps = math.floor((room - start) / step)
    return start + steps * step


def _line(stat: str, projection: float, position: str) -> int | None:
    if stat == "pts":
        return _snap(projection, 5, 5, 5)
    if stat == "reb":
        if position == "C":
            return _snap(projection, 3, 4, 2)
        if position == "PF":
            return 4 if projection - 4 >= 3 else None
        return None
    if stat == "ast":
        if position not in GUARDS:
            return None
        return _snap(projection, 3, 2, 2)
    if stat == "fg3m":
        if position not in WINGS:
            return None
        return _snap(projection, 1, 1, 1)
    if stat == "stl":
        if position not in WINGS:
            return None
        return _snap(projection, 1, 1, 1)
    if stat == "blk":
        if position != "C":
            return None
        return _snap(projection, 1, 1, 1)
    return None


def _positions(raw: Path, logs: pl.DataFrame) -> dict[int, str]:
    """Best depth-chart slot for each player. Box-score flags cover anyone left off the chart."""
    directory = load_directory(raw)
    full_names = [f"{team.city} {team.name}" for team in directory.values()]
    by_team: dict[str, list] = {}
    depth_path = raw / "depth-charts.txt"
    if depth_path.exists():
        for slot in load_depth_charts(depth_path):
            resolved = resolve_team(slot.team, full_names)
            if resolved is None:
                continue
            by_team.setdefault(canonical_team(resolved), []).append(slot)

    people_by_team: dict[int, list[tuple[int, str, str]]] = {}
    for row in logs.select(["playerteamId", "personId", "firstName", "lastName"]).unique().iter_rows(named=True):
        people_by_team.setdefault(int(row["playerteamId"]), []).append(
            (int(row["personId"]), str(row["firstName"] or ""), str(row["lastName"] or ""))
        )

    best: dict[int, tuple[tuple[int, int], str]] = {}
    for team in directory.values():
        key = canonical_team(f"{team.city} {team.name}")
        for slot in by_team.get(key, []):
            person_id = match_person(slot.name, people_by_team.get(team.team_id, []))
            if person_id is None:
                continue
            rank = (slot.rank, POSITION_ORDER.get(slot.position, 9))
            current = best.get(person_id)
            if current is None or rank < current[0]:
                best[person_id] = (rank, slot.position)
    positions = {person_id: position for person_id, (_rank, position) in best.items()}

    players = pl.read_csv(
        raw / "Players.csv",
        schema_overrides={"personId": pl.Int64, "guard": pl.Utf8, "forward": pl.Utf8, "center": pl.Utf8},
        ignore_errors=True,
    )
    for row in players.iter_rows(named=True):
        person_id = int(row["personId"])
        if person_id in positions:
            continue
        guard = str(row.get("guard") or "").strip() in {"1", "1.0"}
        forward = str(row.get("forward") or "").strip() in {"1", "1.0"}
        center = str(row.get("center") or "").strip() in {"1", "1.0"}
        if center and not forward:
            positions[person_id] = "C"
        elif center and forward:
            positions[person_id] = "C"
        elif guard and not forward:
            positions[person_id] = "SG"
        elif forward and not guard:
            positions[person_id] = "SF"
        elif guard and forward:
            positions[person_id] = "SF"
    return positions


def _season(logs: pl.DataFrame) -> dict[int, list[dict]]:
    played = logs.filter(pl.col("numMinutes") > 0).sort("gameDateTimeEst")
    columns = ["personId", "gameDateTimeEst", "numMinutes", *[source for _key, source, _label in STAT_FIELDS]]
    by_player: dict[int, list[dict]] = {}
    for row in played.select(columns).iter_rows(named=True):
        by_player.setdefault(int(row["personId"]), []).append(row)
    return by_player


def _qualifying_line(stat: str, source: str, projection: float, position: str, games: list[dict]) -> dict | None:
    if not games:
        return None
    values = [float(game[source]) for game in games]
    average = sum(values) / len(values)
    if not projection > average:
        return None
    recent = values[-10:]
    hits = sum(1 for value in recent if value >= average)
    if hits < 6:
        return None
    line = _line(stat, projection, position)
    if line is None:
        return None
    return {
        "line": line,
        "season_avg": round(average, 1),
        "hits": hits,
        "recent": len(recent),
        "minutes": round(sum(float(game["numMinutes"]) for game in games) / len(games), 1),
    }


def write_props(raw: Path, day: date, out_dir: Path) -> list[dict]:
    """Write output/<date>/props.json from every filmed box score in the slate folder."""
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    logs = _read_player_logs(raw, day)
    positions = _positions(raw, logs)
    seasons = _season(logs)
    lines: list[dict] = []
    for game in summary.get("games", []):
        video = str(game.get("video") or "").replace("\\", "/")
        folder_name = video.split("/")[-2] if "/" in video else ""
        folder = out_dir / folder_name
        box_path = folder / "boxscore_selected.csv"
        if not box_path.exists():
            continue
        with box_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        away = game["away"]["abbrev"]
        home = game["home"]["abbrev"]
        for row in rows:
            person_id = int(row["person_id"])
            position = positions.get(person_id, "")
            games = seasons.get(person_id, [])
            for stat, source, label in STAT_FIELDS:
                found = _qualifying_line(stat, source, float(row[stat]), position, games)
                if found is None:
                    continue
                lines.append(
                    {
                        "game_id": game.get("game_id", ""),
                        "tip_est": game.get("tip_est", ""),
                        "slug": folder_name,
                        "away": away,
                        "home": home,
                        "team": row["team"],
                        "opponent": home if row["team"] == away else away,
                        "person_id": person_id,
                        "name": row["name"],
                        "position": position,
                        "starter": str(row.get("starter") or "0") == "1",
                        "stat": stat,
                        "label": label,
                        "projection": round(float(row[stat]), 1),
                        **found,
                    }
                )
    stat_order = {stat: index for index, (stat, _source, _label) in enumerate(STAT_FIELDS)}
    lines.sort(key=lambda line: (line["tip_est"], line["slug"], line["team"] != line["away"], line["name"], stat_order[line["stat"]]))
    payload = {"date": day.isoformat(), "lines": lines}
    (out_dir / "props.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Props  {len(lines)} lines", flush=True)
    for line in lines:
        role = "starter" if line["starter"] else "bench"
        print(
            f"  {line['team']} {line['name']:<22} {line['label']} {line['line']}+"
            f"  avg {line['season_avg']:.1f}  proj {line['projection']:.1f}"
            f"  {line['hits']}/{line['recent']}  {role}  {line['minutes']:.1f} mpg",
            flush=True,
        )
    return lines
