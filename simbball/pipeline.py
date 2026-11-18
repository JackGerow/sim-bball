"""Run the day's slate: 10,000 sims per game, then the most common final score."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path

import numpy as np

from simbball.engine import simulate_many
from simbball.ratings import ScheduledGame, build_world
from simbball.render import render_game

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

def _box_rows(team, box: np.ndarray, seconds: np.ndarray) -> list[dict]:
    rows = []
    for index, player in enumerate(team.players):
        rows.append(
            {
                "team": team.abbrev,
                "person_id": player.person_id,
                "name": f"{player.first} {player.last}",
                "jersey": player.jersey,
                "starter": int(player.starter),
                "minutes": round(float(seconds[index]) / 60.0, 1),
                "pts": round(float(box[index, 13]), 1),
                "reb": round(float(box[index, 6] + box[index, 7]), 1),
                "ast": round(float(box[index, 8]), 1),
                "stl": round(float(box[index, 9]), 1),
                "blk": round(float(box[index, 10]), 1),
                "tov": round(float(box[index, 11]), 1),
                "pf": round(float(box[index, 12]), 1),
                "fgm": round(float(box[index, 0]), 1),
                "fga": round(float(box[index, 1]), 1),
                "fg3m": round(float(box[index, 2]), 1),
                "fg3a": round(float(box[index, 3]), 1),
                "ftm": round(float(box[index, 4]), 1),
                "fta": round(float(box[index, 5]), 1),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_game(home, away, league, game: ScheduledGame, n_sims: int, seed: int, out_dir: Path, max_events: int | None) -> dict:
    label = f"{away.abbrev} @ {home.abbrev}"
    print(f"Simulating {label}  ({n_sims:,} games)", flush=True)
    result = simulate_many(home, away, league, n_sims, seed)
    folder = out_dir / f"{away.abbrev}_at_{home.abbrev}"
    folder.mkdir(parents=True, exist_ok=True)
    _write_csv(folder / "boxscore_mean.csv", _box_rows(home, result.mean_box_home, result.mean_sec_home) + _box_rows(away, result.mean_box_away, result.mean_sec_away))
    selected = result.selected
    _write_csv(
        folder / "boxscore_selected.csv",
        _box_rows(home, selected.box_home, selected.sec_home) + _box_rows(away, selected.box_away, selected.sec_away),
    )
    with (folder / "score_sims.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sim", "home_score", "away_score"])
        for index, (home_score, away_score) in enumerate(result.scores):
            writer.writerow([index, int(home_score), int(away_score)])
    video = folder / "simulation.mp4"
    render_game(
        home,
        away,
        selected.plays or [],
        result.exp_margin,
        selected.periods,
        video,
        stills_dir=folder,
        max_events=max_events,
        arena=game.arena,
        n_sims=n_sims,
        mode_count=result.mode_count,
    )
    home_scores = result.scores[:, 0]
    away_scores = result.scores[:, 1]
    summary = {
        "game_id": game.game_id,
        "tip_est": game.tip_est,
        "arena": game.arena,
        "away": {"abbrev": away.abbrev, "city": away.city, "name": away.name, "ortg": round(away.ortg, 1), "drtg": round(away.drtg, 1), "pace": round(away.pace, 1)},
        "home": {"abbrev": home.abbrev, "city": home.city, "name": home.name, "ortg": round(home.ortg, 1), "drtg": round(home.drtg, 1), "pace": round(home.pace, 1)},
        "sims": n_sims,
        "home_win_prob": round(result.home_win_prob, 4),
        "mean_score": [round(float(away_scores.mean()), 1), round(float(home_scores.mean()), 1)],
        "median_score": [int(np.median(away_scores)), int(np.median(home_scores))],
        "p10_score": [int(np.percentile(away_scores, 10)), int(np.percentile(home_scores, 10))],
        "p90_score": [int(np.percentile(away_scores, 90)), int(np.percentile(home_scores, 90))],
        "selected_score": [selected.away_score, selected.home_score],
        "selected_seed": int(result.seeds[result.selected_index]),
        "mode_count": result.mode_count,
        "criterion": "final score that occurred most often; the filmed game is one of the simulations that ended on that score",
        "video": str(video.relative_to(ROOT)),
        "mean_box": _box_rows(away, result.mean_box_away, result.mean_sec_away) + _box_rows(home, result.mean_box_home, result.mean_sec_home),
    }
    (folder / "summary.json").write_text(json.dumps({k: v for k, v in summary.items() if k != "mean_box"}, indent=2), encoding="utf-8")
    print(
        f"  {label}  mean {summary['mean_score'][0]:.1f}-{summary['mean_score'][1]:.1f}"
        f"  {home.abbrev} win {result.home_win_prob:.1%}"
        f"  replay {selected.away_score}-{selected.home_score} ({result.mode_count} of {n_sims:,})",
        flush=True,
    )
    return summary


def _print_report(day: date, summaries: list[dict]) -> None:
    print()
    print(f"{day.isoformat()}   {len(summaries)} games   {summaries[0]['sims']:,} simulations each")
    print("Replay is a simulated game that ended on the most common final score.")
    print()
    for game in summaries:
        away = game["away"]["abbrev"]
        home = game["home"]["abbrev"]
        mean_a, mean_h = game["mean_score"]
        sel_a, sel_h = game["selected_score"]
        print(
            f"{away} {mean_a:5.1f} @ {home} {mean_h:5.1f}"
            f"   {home} {game['home_win_prob']:.1%}"
            f"   replay {sel_a}-{sel_h} ({game['mode_count']} of {game['sims']:,})"
            f"   p10-p90 {game['p10_score'][0]}-{game['p10_score'][1]} to {game['p90_score'][0]}-{game['p90_score'][1]}"
        )
        print(f"    {game['video']}")
        for row in game["mean_box"]:
            if row["minutes"] < 12:
                continue
            print(
                f"    {row['team']} {row['name']:<22} {row['minutes']:4.0f} min"
                f"  {row['pts']:4.1f} pts  {row['reb']:4.1f} reb  {row['ast']:4.1f} ast"
                f"  {row['fgm']:.1f}-{row['fga']:.1f} fg"
            )
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulate today's NBA slate and render the most representative game.")
    parser.add_argument("--date", help="Slate date YYYY-MM-DD. Defaults to the day the script is run.")
    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--only", help="Limit to games involving this abbreviation, for example LAL.")
    parser.add_argument("--max-events", type=int, default=None, help="Render only the first N plays. Full game when omitted.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    day = date.fromisoformat(args.date) if args.date else datetime.now().astimezone().date()
    world = build_world(RAW, day)
    games = world.games
    if args.only:
        token = args.only.upper()
        games = [
            game
            for game in games
            if world.teams[game.home_id].abbrev == token or world.teams[game.away_id].abbrev == token
        ]
    if not games:
        print(f"No NBA games scheduled on {day.isoformat()}.")
        return
    print(f"{day.isoformat()}  {len(games)} games  {args.sims:,} simulations  {args.workers} workers", flush=True)
    out_dir = ROOT / "output" / day.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, game in enumerate(games):
        jobs.append((game, args.seed + index * 1009))
    summaries = []
    if args.workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            summaries = list(pool.map(_pool_job, [(world, game, seed, args.sims, out_dir, args.max_events) for game, seed in jobs]))
    else:
        for game, seed in jobs:
            summaries.append(
                run_game(
                    world.teams[game.home_id],
                    world.teams[game.away_id],
                    world.league,
                    game,
                    args.sims,
                    seed,
                    out_dir,
                    args.max_events,
                )
            )
    payload = {"date": day.isoformat(), "sims_per_game": args.sims, "games": [{k: v for k, v in g.items() if k != "mean_box"} | {"players": [p for p in g["mean_box"] if p["minutes"] >= 8]} for g in summaries]}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_report(day, summaries)
    if args.max_events is None:
        _publish_results(out_dir)


def _git_executable() -> str:
    """Find git.exe when PowerShell cannot see the Git Bash PATH.

    Python on Windows cannot launch git.cmd, so a PATH entry that points at
    the cmd shim still has to be turned into git.exe.
    """
    found = shutil.which("git")
    if found and found.lower().endswith(".exe"):
        return found
    candidates: list[Path] = []
    if found:
        candidates.append(Path(found).with_suffix(".exe"))
    roots = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("LocalAppData", ""),
    ]
    for root in roots:
        if not root:
            continue
        base = Path(root) / "Git"
        candidates.extend(
            [
                base / "cmd" / "git.exe",
                base / "bin" / "git.exe",
                base / "mingw64" / "bin" / "git.exe",
                Path(root) / "Programs" / "Git" / "cmd" / "git.exe",
            ]
        )
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError("git was not found. Install Git for Windows or add it to PATH.")


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_git_executable(), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _publish_results(out_dir: Path) -> None:
    """Commit this slate's output and push it to origin."""
    rel = out_dir.relative_to(ROOT).as_posix()
    try:
        added = _git(["add", "--", rel])
    except FileNotFoundError as exc:
        print(str(exc), flush=True)
        return
    if added.returncode != 0:
        print(added.stderr.strip() or "Could not stage the sim results.", flush=True)
        return
    staged = _git(["diff", "--cached", "--quiet", "--", rel])
    if staged.returncode == 0:
        print("Sim results already match the repo. Nothing to push.", flush=True)
        return
    message = f"Add {out_dir.name} simulation results."
    committed = _git(["commit", "-m", message])
    if committed.returncode != 0:
        detail = (committed.stderr or committed.stdout).strip()
        print(detail or "Could not commit the sim results.", flush=True)
        return
    pushed = _git(["push", "origin", "HEAD"])
    if pushed.returncode != 0:
        print((pushed.stderr or pushed.stdout).strip() or "Could not push the sim results.", flush=True)
        return
    print(f"Pushed {rel} to origin.", flush=True)


def _pool_job(packed) -> dict:
    world, game, seed, n_sims, out_dir, max_events = packed
    return run_game(
        world.teams[game.home_id],
        world.teams[game.away_id],
        world.league,
        game,
        n_sims,
        seed,
        out_dir,
        max_events,
    )
