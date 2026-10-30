#!/usr/bin/env python3
"""Combine NHL games, sportsbook odds, predictions, and picks into one CSV."""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
NHL = ROOT / "docs/win/hockey/nhl"
OUTPUT = ROOT / "frontend/data/nhl_slate.csv"


def read_csv(path):
    if not path.is_file():
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames or [], list(reader)


def main():
    headers = []
    output_rows = []
    games_dir = NHL / "00_intake/games"

    for games_path in sorted(games_dir.glob("????_??_??_nhl_games.csv")):
        date = games_path.name.removesuffix("_nhl_games.csv")
        sources = [
            games_path,
            NHL / f"00_intake/sportsbook/NHL_{date}.csv",
            NHL / f"00_intake/predictions/hockey_{date}.csv",
            NHL / f"04_select/{date}_NHL.csv",
        ]

        files = [read_csv(path) for path in sources]
        for columns, _ in files:
            for column in columns:
                if column not in headers:
                    headers.append(column)

        games = {row["game_id"]: row.copy() for row in files[0][1]}

        for _, source_rows in files[1:3]:
            for row in source_rows:
                game = games.get(row["game_id"])
                if game is not None:
                    for column, value in row.items():
                        if value and not game.get(column):
                            game[column] = value

        picks = defaultdict(list)
        for row in files[3][1]:
            picks[row["game_id"]].append(row)

        for game in games.values():
            selected = picks.get(game["game_id"])
            if not selected:
                output_rows.append(game)
            else:
                for pick in selected:
                    combined = game.copy()
                    for column, value in pick.items():
                        if value and not combined.get(column):
                            combined[column] = value
                    output_rows.append(combined)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
