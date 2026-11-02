#!/usr/bin/env python3
"""Combine NHL slate sources and final-score/graded-bet sources into CSVs."""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
NHL = ROOT / "docs/win/hockey/nhl"
OUTPUT = ROOT / "frontend/data/nhl_slate.csv"
FINAL_OUTPUT = ROOT / "frontend/data/nhl/nhl_final.csv"


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



def build_nhl_final():
    scores_dir = NHL / "05_final_scores/final_scores"
    graded_dir = NHL / "05_final_scores/graded"
    score_suffix = "_NHL_final_scores.csv"
    graded_suffix = "_results_NHL.csv"

    score_files = {
        path.name.removesuffix(score_suffix): path
        for path in scores_dir.glob(f"????_??_??{score_suffix}")
    }
    graded_files = {
        path.name.removesuffix(graded_suffix): path
        for path in graded_dir.glob(f"????_??_??{graded_suffix}")
    }

    headers = []
    output_rows = []

    for date in sorted(score_files.keys() | graded_files.keys()):
        score_columns, scores = read_csv(
            score_files.get(date, scores_dir / f"{date}{score_suffix}")
        )
        graded_columns, graded = read_csv(
            graded_files.get(date, graded_dir / f"{date}{graded_suffix}")
        )

        for column in score_columns + graded_columns:
            if column not in headers:
                headers.append(column)

        grades_by_game = defaultdict(list)
        for bet in graded:
            grades_by_game[bet["game_id"]].append(bet)

        scored_game_ids = set()
        for score in scores:
            game_id = score["game_id"]
            scored_game_ids.add(game_id)
            bets = grades_by_game.get(game_id)
            if not bets:
                output_rows.append(score)
                continue

            for bet in bets:
                combined = score.copy()
                combined.update({key: value for key, value in bet.items() if value})
                output_rows.append(combined)

        # Retain graded bets even if their matching final-scores row is absent.
        for game_id, bets in grades_by_game.items():
            if game_id not in scored_game_ids:
                output_rows.extend(bets)

    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with FINAL_OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
    build_nhl_final()
