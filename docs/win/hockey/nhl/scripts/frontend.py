#!/usr/bin/env python3
"""Combine dated NHL schedule, sportsbook, predictions, and picks CSVs.

Run from the repository root or any working directory:
    python docs/win/hockey/nhl/scripts/frontend.py

Writes one file containing every available schedule date:
    frontend/data/nhl_slate.json

The schedule defines the games. Sportsbook and prediction rows join on game_id;
selected bets remain individual rows (multiple picks per game are allowed).
Source CSVs are never modified.
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
NHL = ROOT / "docs/win/hockey/nhl"
OUT = ROOT / "frontend/data/nhl_slate.json"
DATE_PATTERN = re.compile(r"(\d{4}_\d{2}_\d{2})_nhl_games\.csv")
IDENTITY_FIELDS = {"game_id", "game_date", "sport", "league"}


def clean(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    return "" if value.lower() in {"nan", "none", "null"} else value


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
        rows: list[dict[str, str]] = []
        for line_number, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise ValueError(f"Malformed CSV row in {path} at line {line_number}")
            rows.append({field: clean(value) for field, value in raw_row.items()})
    return headers, rows


def index_games(rows: list[dict[str, str]], path: Path, date: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        game_id = row.get("game_id", "")
        if not game_id or game_id in result:
            raise ValueError(f"Blank or duplicate game_id={game_id!r} in {path}")
        check_date(row, date, path)
        result[game_id] = row.copy()
    return result


def check_date(row: dict[str, str], date: str, path: Path) -> None:
    row_date = row.get("game_date", "")
    if row_date != date:
        raise ValueError(f"Incorrect game_date={row_date!r}; expected {date} in {path}")


def status(path: Path, rows: list[dict[str, str]]) -> str:
    if not path.is_file():
        return "missing"
    return "present" if rows else "empty"


def equal(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
            float(a), float(b), rel_tol=1e-10, abs_tol=1e-8
        )
    except ValueError:
        return False


def combine(date: str) -> dict:
    paths = {
        "games": NHL / f"00_intake/games/{date}_nhl_games.csv",
        "sportsbook": NHL / f"00_intake/sportsbook/NHL_{date}.csv",
        "predictions": NHL / f"00_intake/predictions/hockey_{date}.csv",
        "picks": NHL / f"04_select/{date}_NHL.csv",
    }

    game_headers, game_rows = read_csv(paths["games"], required=True)
    if not game_rows:
        raise ValueError(f"Empty schedule: {paths['games']}")
    if not {"game_id", "game_date", "game_time", "home_team", "away_team"}.issubset(game_headers):
        raise ValueError(f"Missing required schedule columns: {paths['games']}")

    games = index_games(game_rows, paths["games"], date)
    fields = list(game_headers)
    source_status = {"games": "present"}

    for source_name in ("sportsbook", "predictions"):
        path = paths[source_name]
        headers, rows = read_csv(path)
        source_status[source_name] = status(path, rows)
        if headers and not {"game_id", "game_date"}.issubset(headers):
            raise ValueError(f"Missing game_id or game_date column: {path}")
        for field in headers:
            if field not in fields:
                fields.append(field)

        seen: set[str] = set()
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

            for field, value in row.items():
                if not value:
                    continue
                existing = game.get(field, "")
                if existing and not equal(existing, value):
                    if field in IDENTITY_FIELDS:
                        raise ValueError(f"Conflicting {field} for game_id={game_id} in {path}")
                    print(
                        f"WARNING: Conflicting {field} for game_id={game_id} "
                        f"in {path.relative_to(ROOT)}; keeping first value"
                    )
                    continue
                if not existing:
                    game[field] = value

    picks_path = paths["picks"]
    pick_headers, picks = read_csv(picks_path)
    source_status["picks"] = status(picks_path, picks)
    if pick_headers and not {"game_id", "game_date"}.issubset(pick_headers):
        raise ValueError(f"Missing game_id or game_date column: {picks_path}")
    for pick in picks:
        check_date(pick, date, picks_path)
        if not pick.get("game_id"):
            raise ValueError(f"Blank selected-bet game_id in {picks_path}")
        if pick["game_id"] not in games:
            print(f"WARNING: Pick for unmatched game_id={pick['game_id']} in {picks_path.relative_to(ROOT)}")

    ordered_games = sorted(games.values(), key=lambda row: (row.get("game_time", ""), row["game_id"]))
    return {
        "sources": source_status,
        "games": [{field: game.get(field, "") for field in fields} for game in ordered_games],
        "picks": picks,
    }


def main() -> None:
    games_dir = NHL / "00_intake/games"
    if not games_dir.is_dir():
        raise FileNotFoundError(f"Missing NHL schedule directory: {games_dir}")

    dates = sorted({
        match.group(1)
        for path in games_dir.glob("*_nhl_games.csv")
        if (match := DATE_PATTERN.fullmatch(path.name))
    })
    if not dates:
        raise FileNotFoundError(f"No dated NHL schedule CSVs in {games_dir}")

    slates = {date: combine(date) for date in dates}
    content = {
        "schema_version": 1,
        "sport": "hockey",
        "league": "nhl",
        "latest_date": dates[-1],
        "slates": slates,
    }

    if OUT.is_file():
        try:
            previous = json.loads(OUT.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
        if isinstance(previous, dict) and {
            key: value for key, value in previous.items() if key != "generated_at_utc"
        } == content:
            print(f"UNCHANGED {OUT.relative_to(ROOT)}: {len(dates)} dates")
            return

    content["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(content, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        temporary.replace(OUT)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"WROTE {OUT.relative_to(ROOT)}: {len(dates)} dates, "
          f"{sum(len(slate['games']) for slate in slates.values())} games, "
          f"{sum(len(slate['picks']) for slate in slates.values())} picks")


if __name__ == "__main__":
    main()
