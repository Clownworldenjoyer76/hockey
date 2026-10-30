#!/usr/bin/env python3
"""Export all dated NHL schedule, sportsbook, predictions, and picks as one CSV.

Run from any working directory:
    python docs/win/hockey/nhl/scripts/frontend.py

Writes:
    frontend/data/nhl_slate.csv

The CSV contains two row types:
    game: One row for each scheduled game, with sportsbook odds and predictions.
    pick: One row per selected bet, linked by (game_date, game_id).

Pick-specific columns are prefixed with "pick_" to prevent collisions with
schedule/odds/prediction columns. Every source pick is preserved, including
multiple picks for a game. Missing inputs do not remove scheduled games.
The source CSV files are never modified. No JSON is produced.
"""

from __future__ import annotations

import csv
import filecmp
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
NHL = ROOT / "docs/win/hockey/nhl"
OUT = ROOT / "frontend/data/nhl_slate.csv"
DATE_PATTERN = re.compile(r"(\d{4}_\d{2}_\d{2})_nhl_games\.csv")

BASE_COLUMNS = (
    "row_type", "game_date", "game_id",
    "games_status", "sportsbook_status", "predictions_status", "picks_status",
    "has_sportsbook", "has_predictions", "pick_matches_schedule",
)
IDENTITY_FIELDS = {"game_id", "game_date", "sport", "league", "home_team", "away_team"}


def clean(value: str | None) -> str:
    if value is None:
        return ""
    result = value.strip()
    return "" if result.lower() in {"nan", "none", "null"} else result


def read_csv(path: Path, *, required: bool = False) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required input missing: {path}")
        return [], []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not headers or any(not field for field in headers) or len(headers) != len(set(headers)):
            raise ValueError(f"Invalid or duplicate CSV headers in {path}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Malformed CSV row in {path} at line {line_number}")
            rows.append({field: clean(value) for field, value in row.items()})
    return headers, rows


def source_status(path: Path, rows: list[dict[str, str]]) -> str:
    if not path.is_file():
        return "missing"
    return "present" if rows else "empty"


def check_date(row: dict[str, str], date: str, path: Path) -> None:
    if row.get("game_date", "") != date:
        raise ValueError(
            f"Incorrect game_date={row.get('game_date', '')!r}; expected {date} in {path}"
        )


def equal(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
            float(a), float(b), rel_tol=1e-10, abs_tol=1e-8
        )
    except ValueError:
        return False


def merge(game: dict[str, str], row: dict[str, str], path: Path) -> None:
    for field, value in row.items():
        if not value:
            continue
        previous = game.get(field, "")
        if previous and not equal(previous, value):
            if field in IDENTITY_FIELDS:
                raise ValueError(
                    f"Conflicting {field} for game_id={game['game_id']} in {path}"
                )
            print(
                f"WARNING: Conflicting {field} for game_id={game['game_id']} "
                f"in {path.relative_to(ROOT)}; keeping first value"
            )
        elif not previous:
            game[field] = value


def build(date: str) -> tuple[list[str], list[str], list[dict[str, str]], int, int]:
    paths = {
        "games": NHL / f"00_intake/games/{date}_nhl_games.csv",
        "sportsbook": NHL / f"00_intake/sportsbook/NHL_{date}.csv",
        "predictions": NHL / f"00_intake/predictions/hockey_{date}.csv",
        "picks": NHL / f"04_select/{date}_NHL.csv",
    }

    schedule_headers, schedule_rows = read_csv(paths["games"], required=True)
    if not schedule_rows:
        raise ValueError(f"Empty schedule: {paths['games']}")
    if not {"game_id", "game_date", "game_time", "home_team", "away_team"}.issubset(schedule_headers):
        raise ValueError(f"Missing required schedule columns: {paths['games']}")

    games: dict[str, dict[str, str]] = {}
    for row in schedule_rows:
        game_id = row.get("game_id", "")
        if not game_id or game_id in games:
            raise ValueError(f"Blank or duplicate game_id={game_id!r} in {paths['games']}")
        check_date(row, date, paths["games"])
        games[game_id] = row.copy()

    game_columns = list(schedule_headers)
    statuses = {"games_status": "present"}
    availability: dict[str, set[str]] = {}

    for name in ("sportsbook", "predictions"):
        path = paths[name]
        headers, rows = read_csv(path)
        statuses[f"{name}_status"] = source_status(path, rows)
        if headers and not {"game_id", "game_date"}.issubset(headers):
            raise ValueError(f"Missing game_id or game_date column: {path}")
        for field in headers:
            if field not in game_columns:
                game_columns.append(field)

        seen: set[str] = set()
        availability[name] = set()
        for row in rows:
            game_id = row.get("game_id", "")
            if not game_id or game_id in seen:
                raise ValueError(f"Blank or duplicate game_id={game_id!r} in {path}")
            seen.add(game_id)
            check_date(row, date, path)
            game = games.get(game_id)
            if game is None:
                print(f"WARNING: Unmatched game_id={game_id} in {path.relative_to(ROOT)}")
                continue
            merge(game, row, path)
            availability[name].add(game_id)

    picks_path = paths["picks"]
    pick_headers, picks = read_csv(picks_path)
    statuses["picks_status"] = source_status(picks_path, picks)
    if pick_headers and not {"game_id", "game_date"}.issubset(pick_headers):
        raise ValueError(f"Missing game_id or game_date column: {picks_path}")

    picks_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    unmatched_picks: list[dict[str, str]] = []
    for pick in picks:
        check_date(pick, date, picks_path)
        game_id = pick.get("game_id", "")
        if not game_id:
            raise ValueError(f"Blank selected-bet game_id in {picks_path}")
        if game_id in games:
            picks_by_game[game_id].append(pick)
        else:
            print(f"WARNING: Pick for unmatched game_id={game_id} in {picks_path.relative_to(ROOT)}")
            unmatched_picks.append(pick)

    output_rows: list[dict[str, str]] = []

    def emit_pick(pick: dict[str, str], matched: bool) -> None:
        output_rows.append({
            "row_type": "pick",
            "game_date": date,
            "game_id": pick["game_id"],
            "pick_matches_schedule": "1" if matched else "0",
            **{f"pick_{field}": value for field, value in pick.items()
               if field not in {"game_id", "game_date"}},
        })

    ordered_games = sorted(games.values(), key=lambda game: (game.get("game_time", ""), game["game_id"]))
    for game in ordered_games:
        game_id = game["game_id"]
        output_rows.append({
            **game,
            "row_type": "game",
            **statuses,
            "has_sportsbook": "1" if game_id in availability["sportsbook"] else "0",
            "has_predictions": "1" if game_id in availability["predictions"] else "0",
        })
        for pick in picks_by_game.get(game_id, []):
            emit_pick(pick, matched=True)

    for pick in unmatched_picks:
        emit_pick(pick, matched=False)

    return game_columns, [f"pick_{field}" for field in pick_headers
                          if field not in {"game_id", "game_date"}], output_rows, len(games), len(picks)


def main() -> None:
    games_dir = NHL / "00_intake/games"
    if not games_dir.is_dir():
        raise FileNotFoundError(f"Missing schedule directory: {games_dir}")

    dates = sorted({
        match.group(1)
        for path in games_dir.glob("*_nhl_games.csv")
        if (match := DATE_PATTERN.fullmatch(path.name))
    })
    if not dates:
        raise FileNotFoundError(f"No dated NHL schedule CSVs in {games_dir}")

    columns = list(BASE_COLUMNS)
    rows: list[dict[str, str]] = []
    all_game_fields: set[str] = set()
    all_pick_fields: set[str] = set()
    total_games = total_picks = 0
    for date in dates:
        game_columns, pick_columns, date_rows, game_count, pick_count = build(date)
        reserved_collision = (set(game_columns) | set(pick_columns)) & (set(BASE_COLUMNS) - {"game_id", "game_date"})
        source_collision = set(game_columns) & set(pick_columns)
        if reserved_collision or source_collision:
            raise ValueError(
                f"Conflicting output columns for {date}: "
                f"{sorted(reserved_collision | source_collision)}"
            )
        for column in (*game_columns, *pick_columns):
            if column not in columns:
                columns.append(column)
        all_game_fields.update(game_columns)
        all_pick_fields.update(pick_columns)
        rows.extend(date_rows)
        total_games += game_count
        total_picks += pick_count

    cross_date_collision = all_game_fields & all_pick_fields
    if cross_date_collision:
        raise ValueError(f"Conflicting game/pick output columns: {sorted(cross_date_collision)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        if OUT.is_file() and filecmp.cmp(OUT, temporary, shallow=False):
            print(f"UNCHANGED {OUT.relative_to(ROOT)}: {len(dates)} dates")
            return
        temporary.replace(OUT)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        f"WROTE {OUT.relative_to(ROOT)}: {len(dates)} dates, "
        f"{total_games} games, {total_picks} picks, {len(rows)} rows"
    )


if __name__ == "__main__":
    main()
