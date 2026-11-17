"""Parse data/raw/injuries.txt and data/raw/depth-charts.txt.

Usage:
    python scripts/parse_reports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simbball.reports import load_depth_charts, load_injuries, resolve_team
from simbball.teams import load_directory

RAW = ROOT / "data" / "raw"


def main() -> None:
    injuries = load_injuries(RAW / "injuries.txt")
    charts = load_depth_charts(RAW / "depth-charts.txt")
    directory = load_directory(RAW)
    full_names = [f"{team.city} {team.name}" for team in directory.values()]

    print(f"injuries  {len(injuries)} players  {len({row.team for row in injuries})} teams")
    current = ""
    for row in injuries:
        if row.team != current:
            current = row.team
            print(f"\n{current}")
        ruling = "OUT" if row.unavailable else row.designation or "available"
        print(f"  {ruling:<12} {row.name:<28} {row.status:<12} return {row.return_date}")

    print(f"\ndepth charts  {len(charts)} slots  {len({row.team for row in charts})} teams")
    by_team: dict[str, list] = {}
    for slot in charts:
        by_team.setdefault(slot.team, []).append(slot)
    unresolved = []
    for label, slots in by_team.items():
        resolved = resolve_team(label, full_names)
        if resolved is None:
            unresolved.append(label)
        injured = sorted({slot.name for slot in slots if slot.injured})
        suffix = f"  IL: {', '.join(injured)}" if injured else ""
        print(f"  {label:<16} {len(slots):3} slots{suffix}")
    if unresolved:
        raise SystemExit(f"depth-chart teams not in the directory: {', '.join(unresolved)}")


if __name__ == "__main__":
    main()
