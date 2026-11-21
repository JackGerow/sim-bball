"""Parse the injury report and depth charts in data/raw."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# The pasted report says "LA Clippers" where the team directory says
# "Los Angeles Clippers".
_TEAM_ALIASES = {
    "la clippers": "los angeles clippers",
    "la lakers": "los angeles lakers",
}

_SLOT = re.compile(r"^(PG|SG|SF|PF|C)(\d+)\s+-\s+(.+)$")
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


@dataclass(frozen=True)
class Injury:
    team: str
    name: str
    position: str
    return_date: str
    status: str
    comment: str

    @property
    def designation(self) -> str:
        """Finer tag pulled from the note: out, doubtful, questionable, probable, or blank."""
        text = self.comment.lower()
        if "probable" in text:
            return "probable"
        if "questionable" in text:
            return "questionable"
        if "doubtful" in text:
            return "doubtful"
        for phrase in (
            "ruled out",
            "listed as out",
            "downgraded to out",
            "is out",
            "remains out",
            "won't play",
            "will not play",
            "wont play",
        ):
            if phrase in text:
                return "out"
        return ""

    @property
    def unavailable(self) -> bool:
        status = self.status.strip().lower()
        if status in {"out", "doubtful"}:
            return True
        return self.designation in {"out", "doubtful"}


@dataclass(frozen=True)
class DepthSlot:
    team: str
    position: str
    rank: int
    name: str
    injured: bool


def canonical_team(label: str) -> str:
    key = " ".join(label.lower().replace(".", "").split())
    return _TEAM_ALIASES.get(key, key)


def resolve_team(label: str, full_names: list[str]) -> str | None:
    """Match a report label to a directory name like 'Boston Celtics'."""
    named = {canonical_team(name): name for name in full_names}
    key = canonical_team(label)
    if key in named:
        return named[key]
    hits = [name for canon, name in named.items() if canon.startswith(key + " ") or canon.endswith(" " + key)]
    if len(hits) == 1:
        return hits[0]
    return None


def parse_injuries(text: str) -> list[Injury]:
    injuries: list[Injury] = []
    team = ""
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if "\t" not in line:
            team = line
            continue
        if line.startswith("NAME\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            raise ValueError(f"injuries.txt line {line_no} has {len(parts)} columns, expected 5")
        if not team:
            raise ValueError(f"injuries.txt line {line_no} has a player before a team name")
        name, position, return_date, status, comment = parts[0], parts[1], parts[2], parts[3], "\t".join(parts[4:])
        injuries.append(
            Injury(
                team=team,
                name=name.strip(),
                position=position.strip(),
                return_date=return_date.strip(),
                status=status.strip(),
                comment=comment.strip(),
            )
        )
    if not injuries:
        raise ValueError("injuries.txt did not contain any players")
    return injuries


def parse_depth_charts(text: str) -> list[DepthSlot]:
    slots: list[DepthSlot] = []
    team = ""
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        match = _SLOT.match(line)
        if match is None:
            team = line
            continue
        if not team:
            raise ValueError(f"depth-charts.txt line {line_no} has a slot before a team name")
        name = match.group(3).strip()
        injured = name.endswith("(IL)")
        if injured:
            name = name[:-4].strip()
        slots.append(
            DepthSlot(
                team=team,
                position=match.group(1),
                rank=int(match.group(2)),
                name=name,
                injured=injured,
            )
        )
    if not slots:
        raise ValueError("depth-charts.txt did not contain any slots")
    return slots


def load_injuries(path: Path) -> list[Injury]:
    return parse_injuries(path.read_text(encoding="utf-8"))


def load_depth_charts(path: Path) -> list[DepthSlot]:
    return parse_depth_charts(path.read_text(encoding="utf-8"))


def _person_key(name: str) -> str:
    text = name.lower().replace(".", " ").replace("'", "").replace("-", " ")
    parts = [part for part in text.split() if part not in _SUFFIXES]
    return " ".join(parts)


def _build_person_index(
    people: list[tuple[int, str, str]],
) -> tuple[dict[str, set[int]], dict[str, set[int]], dict[str, set[int]]]:
    exact: dict[str, set[int]] = {}
    by_initial_last: dict[str, set[int]] = {}
    by_last: dict[str, set[int]] = {}
    for person_id, first, last in people:
        first_key = _person_key(first)
        last_key = _person_key(last)
        full_key = _person_key(f"{first} {last}")
        exact.setdefault(full_key, set()).add(person_id)
        if first_key and last_key:
            by_initial_last.setdefault(f"{first_key[0]} {last_key}", set()).add(person_id)
        if last_key:
            by_last.setdefault(last_key, set()).add(person_id)
    return exact, by_initial_last, by_last


def _lookup_person(
    name: str,
    exact: dict[str, set[int]],
    by_initial_last: dict[str, set[int]],
    by_last: dict[str, set[int]],
) -> int | None:
    key = _person_key(name)
    hits = exact.get(key, set())
    if len(hits) != 1:
        parts = key.split()
        if len(parts) >= 2:
            hits = by_initial_last.get(f"{parts[0][0]} {' '.join(parts[1:])}", set())
        if len(hits) != 1 and parts:
            hits = by_last.get(parts[-1], set())
    if len(hits) != 1:
        return None
    return next(iter(hits))


def match_person(name: str, people: list[tuple[int, str, str]]) -> int | None:
    """Match a single report name to a (person_id, first, last) row for one team."""
    exact, by_initial_last, by_last = _build_person_index(people)
    return _lookup_person(name, exact, by_initial_last, by_last)


def match_players(injuries: list[Injury], people: list[tuple[int, str, str]]) -> tuple[dict[int, Injury], list[Injury]]:
    """Match report names to (person_id, first, last) rows for one team.

    Returns the matched injuries and the names that matched nobody.
    """
    exact, by_initial_last, by_last = _build_person_index(people)

    matched: dict[int, Injury] = {}
    missed: list[Injury] = []
    for injury in injuries:
        person_id = _lookup_person(injury.name, exact, by_initial_last, by_last)
        if person_id is None:
            missed.append(injury)
            continue
        # A player can be listed once. Keep the first row if a paste repeats them.
        matched.setdefault(person_id, injury)
    return matched, missed
