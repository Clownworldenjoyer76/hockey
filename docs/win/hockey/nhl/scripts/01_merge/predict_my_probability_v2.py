#!/usr/bin/env python3
"""Strictly isolated NHL my_probability v2: market-anchored, prior-only guard.

Dependency: predict_my_probability.py (v1) in this same directory.

No existing pipeline inputs or outputs are modified. All outputs are written
under docs/win/hockey/nhl/my_probability/v2/.

Inputs: existing merged game features, v1's historical comparisons and final
scores. It re-fits v1 for each historical date, using only earlier games.

The raw v1 outcome model is ALWAYS calculated and retained. The published
my_probability_v2 is conservatively blended with the vig-free moneyline.
Blend weights are calibrated on PRIOR out-of-fold predictions from the SAME
season only and revert to 0 unless both prior Brier and log loss improve.

Caveats: 2025-26 bookmaker capture times cannot be authenticated from the
season-master file. Feature snapshot as-of provenance is not established for
all saved 2026-27 merged fields. Backtested ROI is exploratory, not evidence
of a realizable betting edge. No model is wired into bet selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import predict_my_probability as v1

OUT = v1.NHL / "my_probability" / "v2"
VERSION = "my_probability_v2_guard_2026-11-19"
MIN_PAST_OOF = 200
LOOKBACK_OOF = 350
WEIGHTS = np.linspace(0.0, 1.0, 21)  # 0, .05, ..., 1
WEIGHT_PENALTY = 0.002    # conservative regularizer toward bookmaker
LOG_LOSS_MARGIN = 0.0015 # prior-season improvement required
BRIER_MARGIN = 0.0007
EPS = v1.EPS


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=float)
    return float(np.mean(-y * np.log(p) - (1.0 - y) * np.log(1.0 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def market_guard(prior: pd.DataFrame, season: str) -> tuple[float, int, str]:
    # Only previous completed OUT-OF-FOLD predictions, from the same season.
    past = prior.loc[prior["data_period"].eq(season)].tail(LOOKBACK_OOF)
    n = len(past)
    if n < MIN_PAST_OOF:
        return 0.0, n, "insufficient_same_season_prior_validation"

    y = past["target"].to_numpy(float)
    market = past["vig_free_home_prob"].to_numpy(float)
    raw = past["raw_model_probability_home"].to_numpy(float)
    market_ll, market_bs = logloss(y, market), brier(y, market)

    # Weight is found from strictly past data. It is not chosen on current results.
    metrics = []
    for w in WEIGHTS:
        pred = market + float(w) * (raw - market)
        metrics.append((logloss(y, pred) + WEIGHT_PENALTY * float(w)**2,
                        brier(y, pred), float(w), logloss(y, pred)))
    _, selected_brier, weight, selected_ll = min(metrics, key=lambda z: (z[0], z[2]))
    if weight <= 0:
        return 0.0, n, "market_wins_prior_comparison"
    if market_ll - selected_ll < LOG_LOSS_MARGIN or market_bs - selected_brier < BRIER_MARGIN:
        return 0.0, n, "improvement_below_validation_margin"
    return weight, n, "validated_prior_shrinkage"


def make_predictions(model: v1.Predictor, games: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    raw = model.predict(games)
    market = games["vig_free_home_prob"].to_numpy(dtype=float)
    periods = games["data_period"].astype(str).tolist()
    weights, history_counts, states = [], [], []
    for season in periods:
        weight, count, state = market_guard(prior, season)
        weights.append(weight)
        history_counts.append(count)
        states.append(state)
    weight = np.asarray(weights, dtype=float)
    result = games[["game_id", "game_date", "game_time", "home_team", "away_team", "data_period",
                    "home_decimal", "away_decimal", "vig_free_home_prob",
                    "drat_home_win_prob", "sdv_home_win_prob", "pregame_prices_verified"]].copy()
    result["raw_model_probability_home"] = raw
    result["market_blend_weight"] = weight
    result["prior_validated_games_same_season"] = history_counts
    result["adjustment_status"] = states
    result["my_probability_home"] = np.clip(market + weight * (raw - market), EPS, 1.0-EPS)
    result["my_probability_away"] = 1.0 - result["my_probability_home"]
    result["predicted_winner"] = np.where(result["my_probability_home"].to_numpy() >= 0.5,
                                           result["home_team"], result["away_team"])
    result["home_ev_per_unit"] = result["my_probability_home"] * result["home_decimal"] - 1
    result["away_ev_per_unit"] = result["my_probability_away"] * result["away_decimal"] - 1
    result["training_games"] = model.train_rows
    result["training_cutoff_date"] = model.max_training_date
    result["calibration_games"] = model.calibration_rows
    result["model_version"] = VERSION
    result["odds_provenance"] = np.where(
        result["pregame_prices_verified"], "odds_pull_timestamp_pre_game",
        "historical_odds_capture_time_unknown"
    )
    result["all_feature_asof_verified"] = False
    # Verify complement by calculation, not blind assumption.
    if not np.allclose(result.my_probability_home + result.my_probability_away, 1.0, rtol=0, atol=1e-12):
        raise RuntimeError("Probabilities are not complementary")
    return result


def walkforward(history: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for index, day in enumerate(sorted(history["game_date_dt"].drop_duplicates()), 1):
        trained = v1.fit_past(history, day)
        if trained is None:
            continue
        games = history.loc[history["game_date_dt"].eq(day)].copy()
        if games.empty:
            continue
        prior = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["data_period", "target", "vig_free_home_prob", "raw_model_probability_home"]
        )
        prediction = make_predictions(trained, games, prior)
        prediction["target"] = games["target"].to_numpy(float)
        frames.append(prediction)
        if index % 40 == 0:
            print(f"WALKFORWARD {index} days processed")
    if not frames:
        raise ValueError("No eligible out-of-fold forecasts")
    out = pd.concat(frames, ignore_index=True).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return out


def build_summary(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    summaries = []
    for season, subset in [("ALL", pred), *list(pred.groupby("data_period", sort=True))]:
        reference = {
            "my_probability_v2": subset["my_probability_home"].to_numpy(float),
            "raw_my_probability_v1": subset["raw_model_probability_home"].to_numpy(float),
            "vig_free_market": subset["vig_free_home_prob"].to_numpy(float),
            "D_Ratings": subset["drat_home_win_prob"].to_numpy(float),
            "SportsDataverse": subset["sdv_home_win_prob"].to_numpy(float),
        }
        market_score = brier(subset["target"].to_numpy(float), reference["vig_free_market"])
        for label, p in reference.items():
            score = v1.score_one(subset, label, p, threshold)
            score.update({
                "season": season,
                "brier_minus_market": round(brier(subset["target"].to_numpy(float), p)-market_score, 7),
                "pregame_odds_timestamp_coverage": round(float(subset["pregame_prices_verified"].mean()), 5),
                "full_feature_asof_verified": False,
                "roi_is_exploratory": True,
                "market_fallback_games": int((subset["market_blend_weight"] == 0).sum()),
            })
            summaries.append(score)
    return pd.DataFrame(summaries)


def feature_coverage(history: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    reports = []
    # These are the model candidate's input features, not proof of historical as-of snapshots.
    combined = pd.concat([history.assign(snapshot="labeled"), current.assign(snapshot="merged")],
                         ignore_index=True)
    for (period, snapshot), group in combined.groupby(["data_period", "snapshot"], sort=True):
        for field in v1.FEATURES:
            n = int(group[field].notna().sum())
            reports.append({"season": period, "dataset": snapshot, "feature": field,
                            "rows": len(group), "nonmissing": n,
                            "coverage": round(n / len(group), 6),
                            "feature_asof_fully_verified": False})
    return pd.DataFrame(reports)


def calibration_table(pred: pd.DataFrame) -> pd.DataFrame:
    data = []
    for season, group in [("ALL", pred), *list(pred.groupby("data_period", sort=True))]:
        for label, column in [("my_probability_v2", "my_probability_home"),
                              ("vig_free_market", "vig_free_home_prob")]:
            for bucket in range(10):
                rows = group.loc[(group[column] >= bucket/10) &
                                 (group[column] < (bucket+1)/10 if bucket<9 else group[column] <= 1)]
                if rows.empty:
                    continue
                data.append({"season": season, "model": label,
                             "probability_bucket": f"{bucket/10:.1f}-{(bucket+1)/10:.1f}",
                             "games": len(rows), "mean_predicted": float(rows[column].mean()),
                             "actual_home_win_rate": float(rows["target"].mean()),
                             "calibration_gap": float(rows["target"].mean() - rows[column].mean())})
    return pd.DataFrame(data)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"WROTE {path} | rows={len(df)}")


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--only", choices=["both", "backtest", "predict"], default="both")
    cli.add_argument("--date", help="Merged data date to predict, YYYY_MM_DD; default latest")
    cli.add_argument("--ev-threshold", type=float, default=0.03)
    args = cli.parse_args()
    if not (0 <= args.ev_threshold <= 0.5):
        cli.error("--ev-threshold must be between 0 and .5")
    history, current = v1.load_history()
    # Rebuild historical validation every run; never trust a stale cached score file.
    scored = walkforward(history)
    scored["game_date_dt"] = v1.date_col(scored["game_date"]).to_numpy()
    if args.only in ("backtest", "both"):
        write_csv(scored, OUT / "reports" / "walkforward_predictions.csv")
        summary = build_summary(scored, args.ev_threshold)
        write_csv(summary, OUT / "reports" / "walkforward_summary.csv")
        write_csv(calibration_table(scored), OUT / "reports" / "calibration.csv")
        print(summary[["season", "model", "games", "brier", "log_loss", "roi",
                       "market_fallback_games"]].to_string(index=False))
    write_csv(feature_coverage(history, current), OUT / "reports" / "feature_coverage.csv")
    if args.only in ("predict", "both"):
        day = (pd.to_datetime(args.date.replace("_", "-"), format="%Y-%m-%d")
               if args.date else current["game_date_dt"].max())
        games = current.loc[current["game_date_dt"].eq(day)]
        if games.empty:
            raise ValueError(f"No merged games on {day.date()}")
        fit = v1.fit_past(history, day)
        if fit is None:
            raise ValueError(f"Insufficient prior history for date {day.date()}")
        prior = scored.loc[scored["game_date_dt"] < day].copy()
        pred = make_predictions(fit, games, prior)
        run_at = datetime.now(timezone.utc)
        pred["generated_at_utc"] = run_at.isoformat()
        game_local = pd.to_datetime(
            pred["game_date"].astype(str).str.replace("_", "-", regex=False)
            + " " + pred["game_time"].astype(str), errors="raise"
        ).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        if game_local.isna().any():
            raise ValueError("Could not safely identify NHL puck-drop time")
        pred["forecast_type"] = np.where(
            game_local.dt.tz_convert("UTC") > pd.Timestamp(run_at),
            "future_fixture_unverified_latest_price", "post_start_reconstruction"
        )
        out = OUT / "predictions" / f"{day.strftime('%Y_%m_%d')}_NHL_my_probability_v2.csv"
        write_csv(pred, out)
        print(pred[["game_date", "home_team", "away_team", "my_probability_home",
                    "my_probability_away", "market_blend_weight", "adjustment_status"]].to_string(index=False))
    print("NOTE: No confirmed predictive edge; keep separate from live bet selection.")
    print("NOTE: 2025-26 moneyline timestamp unverified; all feature as-of provenance unverified.")
    print("NOTE: A past-date forecast is historical reconstruction, not a timely betting signal.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
