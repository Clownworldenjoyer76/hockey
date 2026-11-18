#!/usr/bin/env python3
"""Select NHL player props using actual betting lines and editable markets.yaml.

Default inputs:
  <prop_engine>/00_intake/YYYY_MM_DD_nhl_props_data.csv
  <prop_engine>/01_predictions/YYYY_MM_DD_nhl_player_predictions.csv
  <prop_engine>/config/markets.yaml
  <prop_engine>/models/nhl_8_market_portable_models.json.gz
Default output:
  <prop_engine>/03_select/YYYY_MM_DD_nhl_props.csv

Join odds and predictions on (game_id, official NHL player_id, market).
All model probabilities are calculated for the actual sportsbook line using the
same model_scoring.over_probability and dispersion as predict_player_stats.py.
When no goals_against odds market exists, use the predictions' own threshold
for goalies who appear in the same day's odds file, leaving prices blank.
No projected players absent from the day's odds file are added.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "nhl_8_market_portable_models.json.gz"
DEFAULT_CONFIG = ROOT / "config" / "markets.yaml"
MODEL_SCRIPT_DIR = ROOT / "scripts" / "01_predict"
sys.path.insert(0, str(MODEL_SCRIPT_DIR))
from model_scoring import load_models, over_probability  # noqa: E402

# Betting data names => prediction model names. The eighth model, goals_against,
# has no market in the current ESPN props intake; see synthesized rows below.
PROP_MARKETS = {
    "SOG": "shots_on_goal",
    "Points": "points",
    "Assists": "assists",
    "Goals": "goals",
    "Blocked_Shots": "blocked_shots",
    "PowerPlayPoints": "power_play_points",
    "Saves": "saves",
    "GoalsAgainst": "goals_against",
    "goals_against": "goals_against",
}
MARKETS = (
    "shots_on_goal", "points", "assists", "goals", "blocked_shots",
    "power_play_points", "saves", "goals_against",
)
OUTPUT_FIELDS = (
    "game_date", "game_id", "player_id", "team", "opponent_team",
    "home_away", "game_start_utc", "over_probability", "under_probability",
    "line", "market", "bet_side", "over_odds", "under_odds",
)
ODDS_FIELDS = {"date", "game_id", "player_id", "line", "market", "over_odds", "under_odds"}
PRED_FIELDS = {
    "game_date", "game_id", "player_id", "team", "opponent_team",
    "home_away", "game_start_utc", "market", "predicted_mean", "threshold",
}
PREFERENCES = {"best_prob", "over", "under", "both"}


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        return list(reader)


def ranges(value: object, name: str, *, low: float | None = None,
           high: float | None = None) -> list[tuple[float, float]]:
    """Ranges are inclusive; multiple pairs are OR; [] means unrestricted."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of [minimum, maximum] pairs")
    output: list[tuple[float, float]] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{name}[{index}] must have exactly two endpoints")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in pair):
            raise ValueError(f"{name}[{index}] must contain numeric endpoints")
        minimum, maximum = map(float, pair)
        if (not math.isfinite(minimum) or not math.isfinite(maximum)
                or minimum > maximum or (low is not None and minimum < low)
                or (high is not None and maximum > high)):
            raise ValueError(f"Invalid range {name}[{index}]: {pair}")
        output.append((minimum, maximum))
    return output


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Markets configuration does not exist: {path}")
    with path.open("r", encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) != set(MARKETS):
        raise ValueError(f"markets.yaml must contain exactly these markets: {list(MARKETS)}")
    parsed = {}
    for market in MARKETS:
        opts = config[market]
        if not isinstance(opts, dict) or set(opts) != {"enabled", "pick_preference", "over", "under"}:
            raise ValueError(f"{market} requires enabled, pick_preference, over and under")
        if not isinstance(opts["enabled"], bool):
            raise ValueError(f"{market}.enabled must be true or false")
        preference = opts["pick_preference"]
        if preference not in PREFERENCES:
            raise ValueError(f"{market}.pick_preference must be one of {sorted(PREFERENCES)}")
        sides = {}
        for side in ("over", "under"):
            spec = opts[side]
            if not isinstance(spec, dict) or set(spec) != {"enabled", "line", "prob", "odds"}:
                raise ValueError(f"{market}.{side} requires enabled, line, prob and odds")
            if not isinstance(spec["enabled"], bool):
                raise ValueError(f"{market}.{side}.enabled must be true or false")
            sides[side] = {
                "enabled": spec["enabled"],
                "line": ranges(spec["line"], f"{market}.{side}.line", low=0),
                "prob": ranges(spec["prob"], f"{market}.{side}.prob", low=0, high=1),
                "odds": ranges(spec["odds"], f"{market}.{side}.odds"),
            }
        parsed[market] = {
            "enabled": opts["enabled"],
            "pick_preference": preference,
            **sides,
        }
    return parsed


def valid_id(raw: str) -> bool:
    return bool(re.fullmatch(r"\d+", str(raw or "").strip()))


def date_label(raw: str) -> str:
    try:
        return datetime.strptime(str(raw).replace("_", "-"), "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date: {raw!r}") from exc


def valid_number(raw: object, field: str, *, nonnegative: bool = False) -> float:
    try:
        num = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {raw!r}") from exc
    if not math.isfinite(num) or (nonnegative and num < 0):
        raise ValueError(f"Invalid {field}: {raw!r}")
    return num


def price(raw: str) -> float | None:
    raw = str(raw or "").strip()
    if not raw:
        return None  # An absent price expressly bypasses the odds filter.
    if not re.fullmatch(r"[+-]?\d+", raw):
        raise ValueError(f"Invalid American odds: {raw!r}")
    value = int(raw)
    if -99 <= value <= 99:
        raise ValueError(f"Invalid American odds: {raw!r}")
    return float(value)


def matches(value: float, intervals: list[tuple[float, float]]) -> bool:
    return not intervals or any(a <= value <= b for a, b in intervals)


def probabilities(mean: float, line: float, alpha: float) -> tuple[float, float]:
    """Exact P(X > line) and P(X < line); for integer lines push is distinct."""
    over = float(over_probability([mean], line, alpha)[0])
    under = 1.0 - float(over_probability([mean], math.ceil(line) - 1, alpha)[0])
    if not (0 <= over <= 1 and 0 <= under <= 1):
        raise ValueError(f"Invalid probabilities for predicted_mean={mean}, line={line}")
    return over, under


def choose_sides(cfg: dict, line: float, over_prob: float, under_prob: float,
                 over_odds: str, under_odds: str) -> list[str]:
    if not cfg["enabled"]:
        return []
    eligible = []
    for side, probability, odds in (
        ("over", over_prob, over_odds), ("under", under_prob, under_odds),
    ):
        spec = cfg[side]
        if (spec["enabled"] and matches(line, spec["line"])
                and matches(probability, spec["prob"])):
            value = price(odds)
            if value is None or matches(value, spec["odds"]):
                eligible.append(side)
    preference = cfg["pick_preference"]
    if preference == "both":
        return eligible
    if preference in ("over", "under"):
        return [preference] if preference in eligible else []
    return [max(eligible, key=lambda side: (over_prob if side == "over" else under_prob,
                                            side == "over"))] if eligible else []


def write_atomic(path: Path, data: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="",
                                         suffix=".tmp", dir=path.parent,
                                         delete=False) as stream:
            tmp = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(data)
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def select(day: str, config_path: Path, *, root: Path = ROOT) -> tuple[Path, Counter]:
    display_day = date_label(day)
    stamp = display_day.replace("-", "_")
    props_path = root / "00_intake" / f"{stamp}_nhl_props_data.csv"
    preds_path = root / "01_predictions" / f"{stamp}_nhl_player_predictions.csv"
    output_path = root / "03_select" / f"{stamp}_nhl_props.csv"
    cfg = load_config(config_path)
    odds = read_rows(props_path, ODDS_FIELDS)
    predictions = read_rows(preds_path, PRED_FIELDS)

    pred_by_key = {}
    for row in predictions:
        if date_label(row["game_date"]) != display_day:
            raise ValueError(f"Prediction includes another date: {row['game_date']}")
        game, player, market = (row["game_id"].strip(), row["player_id"].strip(),
                                row["market"].strip())
        if not valid_id(game) or not valid_id(player) or market not in MARKETS:
            raise ValueError(f"Invalid prediction identity: {game}/{player}/{market}")
        key = (game, player, market)
        if key in pred_by_key:
            raise ValueError(f"Duplicate prediction: {key}")
        pred_by_key[key] = row

    counts = Counter()
    observed_keys = set()
    odds_players = set()
    inputs: list[tuple[dict[str, str], dict[str, str], str]] = []
    priced_ga_players = set()
    for row in odds:
        if date_label(row["date"]) != display_day:
            raise ValueError(f"Odds include another date: {row['date']}")
        game = row["game_id"].strip()
        player = row["player_id"].strip()
        if valid_id(game) and valid_id(player):
            odds_players.add((game, player))
        market = PROP_MARKETS.get(row["market"].strip())
        if market is None:
            counts["unknown_market"] += 1
            continue
        if not valid_id(game) or not valid_id(player):
            counts["unmatched"] += 1
            continue
        key = (game, player, market)
        pred = pred_by_key.get(key)
        if pred is None:
            counts["unmatched"] += 1
            continue
        line = valid_number(row["line"], "sportsbook line", nonnegative=True)
        unique = (key, line)
        if unique in observed_keys:
            raise ValueError(f"Duplicate sportsbook player/market/line: {unique}")
        observed_keys.add(unique)
        if market == "goals_against":
            priced_ga_players.add((game, player))
        inputs.append((pred, row, market))
        counts["matched"] += 1

    # The current intake has no goals-against betting market. For a goalie
    # already present in the odds file (e.g. through Saves), include the model's
    # own published goals_against threshold; never invent betting odds.
    for (game, player, market), pred in pred_by_key.items():
        if (market == "goals_against" and (game, player) in odds_players
                and (game, player) not in priced_ga_players):
            inputs.append((pred, {
                "line": pred["threshold"], "over_odds": "", "under_odds": "",
            }, market))
            counts["goals_against_without_odds"] += 1

    models = load_models(root / "models" / "nhl_8_market_portable_models.json.gz")
    output = []
    for pred, prop, market in inputs:
        if not cfg[market]["enabled"]:
            counts["market_disabled"] += 1
            continue
        mean = valid_number(pred["predicted_mean"], "predicted_mean", nonnegative=True)
        line = valid_number(prop["line"], "sportsbook line", nonnegative=True)
        over, under = probabilities(mean, line, models[market].alpha)
        over_odds = str(prop.get("over_odds", "") or "").strip()
        under_odds = str(prop.get("under_odds", "") or "").strip()
        sides = choose_sides(cfg[market], line, over, under, over_odds, under_odds)
        if not sides:
            counts["filtered_out"] += 1
        for side in sides:
            output.append({
                "game_date": pred["game_date"],
                "game_id": pred["game_id"],
                "player_id": pred["player_id"],
                "team": pred["team"],
                "opponent_team": pred["opponent_team"],
                "home_away": pred["home_away"],
                "game_start_utc": pred["game_start_utc"],
                "over_probability": over,
                "under_probability": under,
                "line": prop["line"],
                "market": market,
                "bet_side": side.upper(),
                "over_odds": over_odds,
                "under_odds": under_odds,
            })
    output.sort(key=lambda r: (r["game_id"], r["player_id"], r["market"],
                               float(r["line"]), r["bet_side"]))
    write_atomic(output_path, output)
    counts["selected_bets"] = len(output)
    return output_path, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d"),
                        help="YYYY-MM-DD or YYYY_MM_DD (defaults to today's Eastern date)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        path, counts = select(args.date, args.config)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"WROTE {path}")
    print(" | ".join(f"{name}={counts[name]}" for name in (
        "matched", "unmatched", "unknown_market", "goals_against_without_odds",
        "market_disabled", "filtered_out", "selected_bets",
    )))


if __name__ == "__main__":
    main()
