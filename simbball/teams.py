"""Team colors and the current NBA directory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# primary, secondary, text-on-primary
COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "ATL": ((225, 58, 62), (196, 214, 0), (255, 255, 255)),
    "BOS": ((0, 122, 51), (186, 195, 201), (255, 255, 255)),
    "BKN": ((18, 18, 18), (255, 255, 255), (255, 255, 255)),
    "CHA": ((29, 17, 96), (0, 140, 158), (255, 255, 255)),
    "CHI": ((206, 17, 65), (6, 25, 34), (255, 255, 255)),
    "CLE": ((134, 0, 56), (253, 187, 48), (255, 255, 255)),
    "DAL": ((0, 83, 188), (187, 196, 202), (255, 255, 255)),
    "DEN": ((13, 34, 64), (255, 198, 39), (255, 255, 255)),
    "DET": ((200, 16, 46), (29, 66, 138), (255, 255, 255)),
    "GSW": ((29, 66, 138), (255, 199, 44), (255, 255, 255)),
    "HOU": ((206, 17, 65), (6, 25, 34), (255, 255, 255)),
    "IND": ((0, 45, 98), (253, 187, 48), (255, 255, 255)),
    "LAC": ((200, 16, 46), (29, 66, 138), (255, 255, 255)),
    "LAL": ((85, 37, 130), (253, 185, 39), (255, 255, 255)),
    "MEM": ((93, 118, 169), (18, 23, 63), (255, 255, 255)),
    "MIA": ((152, 0, 46), (249, 160, 27), (255, 255, 255)),
    "MIL": ((0, 71, 27), (240, 235, 210), (255, 255, 255)),
    "MIN": ((12, 35, 64), (120, 190, 32), (255, 255, 255)),
    "NOP": ((0, 22, 65), (225, 198, 128), (255, 255, 255)),
    "NYK": ((0, 107, 182), (245, 132, 38), (255, 255, 255)),
    "OKC": ((0, 125, 195), (239, 59, 36), (255, 255, 255)),
    "ORL": ((0, 125, 197), (196, 206, 212), (255, 255, 255)),
    "PHI": ((0, 107, 182), (237, 23, 76), (255, 255, 255)),
    "PHX": ((229, 95, 32), (29, 17, 96), (255, 255, 255)),
    "POR": ((224, 58, 62), (6, 25, 34), (255, 255, 255)),
    "SAC": ((91, 43, 130), (188, 149, 86), (255, 255, 255)),
    "SAS": ((196, 206, 211), (6, 25, 34), (20, 20, 20)),
    "SAN": ((196, 206, 211), (6, 25, 34), (20, 20, 20)),
    "TOR": ((206, 17, 65), (6, 25, 34), (255, 255, 255)),
    "UTA": ((79, 16, 122), (249, 160, 27), (255, 255, 255)),
    "WAS": ((0, 43, 92), (227, 24, 55), (255, 255, 255)),
}


@dataclass(frozen=True)
class TeamInfo:
    team_id: int
    city: str
    name: str
    abbrev: str
    primary: tuple[int, int, int]
    secondary: tuple[int, int, int]
    ink: tuple[int, int, int]


def luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def load_directory(raw: Path) -> dict[int, TeamInfo]:
    hist = pl.read_csv(raw / "TeamHistories.csv")
    current = hist.filter(
        (pl.col("league") == "NBA") & (pl.col("seasonActiveTill") >= 2026)
    ).sort("seasonActiveTill", descending=True)
    seen: dict[int, TeamInfo] = {}
    for row in current.iter_rows(named=True):
        team_id = int(row["teamId"])
        if team_id in seen:
            continue
        abbrev = str(row["teamAbbrev"]).strip()
        primary, secondary, ink = COLORS.get(
            abbrev, ((40, 70, 110), (200, 200, 200), (255, 255, 255))
        )
        seen[team_id] = TeamInfo(
            team_id=team_id,
            city=str(row["teamCity"]),
            name=str(row["teamName"]),
            abbrev=abbrev,
            primary=primary,
            secondary=secondary,
            ink=ink,
        )
    return seen


def colors_apart(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    return math.dist(a, b) >= 80
