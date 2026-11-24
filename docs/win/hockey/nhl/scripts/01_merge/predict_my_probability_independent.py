#!/usr/bin/env python3
"""Standalone independent NHL outcome model: historical scores and pregame source signals.

Install beside predict_my_probability.py and predict_my_probability_v3.py.
Only writes to: NHL/my_probability/independent_v4/

DESIGN:
- No sportsbook odds enter the trained probability model. No fallback to market.
- Features: chronological Elo (derived from past scores), past-date-only team
  rest/form, optional past-date-only xG, D-Ratings and SportsDataverse forecasts.
- Team-only and sources-only models are reported as honest ablations.
- Hyperparameters are chosen using ONLY an earlier 2025-26 train/validation
  split. The 2026-27 season is a separate, chronological forward holdout.
- Each holdout date's model is fitted to results strictly before that date.
- Sportsbook odds are held out from fitting and used only to compare forecasts
  or calculate exploratory ROI. The 2025-26 capture timestamps are unverified.
- Historical D-Ratings and SDV prediction capture times are also not fully
  independently authenticated; score-derived Elo/rest/form use strictly past dates.
- No production betting pipeline, daily intake, or earlier model is modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

import predict_my_probability as v1
import predict_my_probability_v3 as v3

BASE = v1.NHL
OUT = BASE / "my_probability" / "independent_v4"
VERSION = "independent_v4_historical_2026_11_19"
EPS = 1e-7
PENALTIES = (2.0, 15.0, 60.0, 200.0)
MODEL_VARIANTS = ("historical_plus_sources", "team_history_only", "sources_only")
TEAM_FEATURES = (
    "elo_diff_asof", "rest_diff_asof", "back_to_back_diff_asof",
    "games_in_four_diff_asof", "win_rate_last10_diff_asof",
    "goals_diff_last10_diff_asof", "goals_for_last10_diff_asof",
    "goals_against_last10_diff_asof", "prior_games_min_asof",
)
SOURCES = ("logit_drat", "logit_sdv")


def write(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    print(f"WROTE {target} | rows={len(frame)}", flush=True)


def brier(y, p) -> float:
    a = np.asarray(y, float); q = np.asarray(p, float)
    return float(np.mean((a - q) ** 2))


def log_loss(y, p) -> float:
    a = np.asarray(y, float); q = np.clip(np.asarray(p, float), EPS, 1. - EPS)
    return float(np.mean(-a * np.log(q) - (1. - a) * np.log1p(-q)))


def elo_prior_results(historical: pd.DataFrame, scheduled: pd.DataFrame,
                      k: float = 20., offseason_carry: float = .45,
                      home_advantage: float = 65.) -> pd.DataFrame:
    """Generate each day's Elo BEFORE updating from that day's final scores."""
    columns = ["game_id", "game_date_dt", "data_period", "home_team_full",
               "away_team_full", "home_score", "away_score", "target"]
    combined = (pd.concat([historical[columns], scheduled[columns]], ignore_index=True)
                .drop_duplicates("game_id")
                .sort_values(["game_date_dt", "game_id"]))
    ratings = defaultdict(lambda: 1500.)
    previous_season = None
    records = []
    for game_day, day_games in combined.groupby("game_date_dt", sort=True):
        season_names = day_games["data_period"].unique()
        if len(season_names) != 1:
            raise ValueError("Multiple seasons on one calendar date: Elo as-of ambiguous")
        season = str(season_names[0])
        if previous_season is not None and season != previous_season:
            ratings = defaultdict(lambda: 1500., {
                team: 1500. + offseason_carry * (value - 1500.)
                for team, value in ratings.items()})
        previous_season = season
        for game in day_games.itertuples(index=False):
            diff = ratings[game.home_team_full] - ratings[game.away_team_full]
            records.append({
                "game_id": game.game_id,
                "elo_diff_asof": diff / 400.,
                "elo_games_asof_rule": "final_game_date_strictly_less_than_target_date",
            })
        for game in day_games.itertuples(index=False):
            if not np.isfinite(game.target):
                continue
            diff = ratings[game.home_team_full] - ratings[game.away_team_full]
            expected = 1. / (1. + 10. ** (-(diff + home_advantage) / 400.))
            mov = abs(float(game.home_score) - float(game.away_score))
            change = k * (float(game.target) - expected) * (1. + .13 * max(mov - 1., 0.))
            ratings[game.home_team_full] += change
            ratings[game.away_team_full] -= change
    out = pd.DataFrame(records)
    if out["game_id"].duplicated().any() or len(out) != len(combined):
        raise ValueError("Elo feature join cardinality failure")
    return out


@dataclass
class RidgeLogit:
    features: tuple[str, ...]
    penalty: float
    median: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    coefficients: np.ndarray | None = None

    def design(self, data: pd.DataFrame, *, training: bool) -> np.ndarray:
        values = data.loc[:, self.features].to_numpy(float)
        missing = ~np.isfinite(values)
        if training:
            self.median = np.asarray([
                np.median(values[~missing[:, j], j]) if (~missing[:, j]).any() else 0.
                for j in range(values.shape[1])], float)
        completed = np.where(missing, self.median, values)
        if training:
            self.means = completed.mean(axis=0)
            self.scales = np.where(completed.std(axis=0) > 1e-8,
                                   completed.std(axis=0), 1.)
        return np.column_stack([
            np.ones(len(data)), (completed - self.means) / self.scales,
            missing.astype(float)
        ])

    def fit(self, data: pd.DataFrame) -> "RidgeLogit":
        if len(data) < 250 or data["target"].nunique() != 2:
            raise ValueError("Insufficient completed historical training games")
        x = self.design(data, training=True)
        y = data["target"].to_numpy(float)
        def loss(beta):
            z = np.clip(x @ beta, -35., 35.)
            return (float(np.logaddexp(0., z).sum() - np.dot(y, z)
                          + .5 * self.penalty * np.dot(beta[1:], beta[1:])),
                    x.T @ (expit(z) - y) + np.r_[0., self.penalty * beta[1:]])
        initial = np.zeros(x.shape[1], float)
        initial[0] = logit(np.clip(y.mean(), .01, .99))
        result = minimize(loss, initial, jac=True, method="L-BFGS-B",
                          options={"maxiter": 180, "ftol": 1e-9})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError("Independent logistic model fit failed")
        self.coefficients = result.x
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model must be trained before prediction")
        return np.clip(expit(self.design(data, training=False) @ self.coefficients),
                       EPS, 1. - EPS)


def feature_groups(use_xg: bool) -> dict[str, tuple[str, ...]]:
    team = TEAM_FEATURES + ((v3.XG_FEATURE,) if use_xg else ())
    return {
        "historical_plus_sources": SOURCES + team,
        "team_history_only": team,
        "sources_only": SOURCES,
    }


def train_validate(historical: pd.DataFrame, groups: dict[str, tuple[str, ...]]):
    first = historical.loc[historical["data_period"].eq("2025_2026")].copy()
    unique_days = sorted(first["game_date_dt"].unique())
    if len(unique_days) < 90:
        raise ValueError("Not enough distinct historical dates for time validation")
    cut = pd.Timestamp(unique_days[int(len(unique_days) * .65)])
    train = first.loc[first["game_date_dt"] < cut]
    valid = first.loc[first["game_date_dt"] >= cut]
    if len(train) < 500 or len(valid) < 200:
        raise ValueError("Historical validation split has too few observations")
    scored = []
    chosen = {}
    for model_name, features in groups.items():
        for penalty in PENALTIES:
            fit = RidgeLogit(features, penalty).fit(train)
            pred = fit.predict(valid)
            scored.append({
                "model": model_name, "penalty": penalty,
                "train_games": len(train), "validation_games": len(valid),
                "last_training_date": train["game_date_dt"].max().date().isoformat(),
                "first_validation_date": cut.date().isoformat(),
                "brier": brier(valid["target"], pred),
                "log_loss": log_loss(valid["target"], pred),
            })
    report = pd.DataFrame(scored).sort_values(["model", "brier", "log_loss", "penalty"])
    for model_name in groups:
        chosen[model_name] = float(report.loc[report.model.eq(model_name)].iloc[0]["penalty"])
    return report, chosen, cut


def process_holdout(historical: pd.DataFrame, groups, chosen):
    targets = historical.loc[historical["data_period"].eq("2026_2027")].copy()
    if targets.empty:
        raise ValueError("No completed 2026-27 results for forward holdout")
    frames = []
    for n, (day, rows) in enumerate(targets.groupby("game_date_dt", sort=True), 1):
        prior = historical.loc[historical["game_date_dt"] < day]
        if (prior["game_date_dt"] >= day).any():
            raise ValueError("Outcome date leakage")
        forecast = rows[["game_id", "game_date", "game_date_dt", "data_period",
                         "home_team", "away_team", "target", "vig_free_home_prob",
                         "drat_home_win_prob", "sdv_home_win_prob", "home_decimal",
                         "away_decimal", "pregame_prices_verified"]].copy()
        for name, features in groups.items():
            model = RidgeLogit(features, chosen[name]).fit(prior)
            forecast[name] = model.predict(rows)
        forecast["train_games_before_target"] = len(prior)
        forecast["max_training_game_date"] = prior["game_date_dt"].max().date().isoformat()
        if forecast["max_training_game_date"].ge(day.date().isoformat()).any():
            raise RuntimeError("Training date not strictly earlier than target date")
        frames.append(forecast)
        if n % 20 == 0:
            print(f"FORWARD TEST: {n} historical target dates processed", flush=True)
    out = pd.concat(frames, ignore_index=True).sort_values(["game_date_dt", "game_id"])
    return out


def bootstrap_interval(data: pd.DataFrame, label: str, repeats: int = 2500):
    y = data["target"].to_numpy(float)
    market = data["vig_free_home_prob"].to_numpy(float)
    model = data[label].to_numpy(float)
    diff = (model - y) ** 2 - (market - y) ** 2
    days = [values.index.to_numpy() for _, values in data.reset_index(drop=True).groupby("game_date_dt", sort=True)]
    rng = np.random.default_rng(20261119)
    vals = np.empty(repeats, float)
    for j in range(repeats):
        idx = np.concatenate([days[k] for k in rng.integers(0, len(days), len(days))])
        vals[j] = diff[idx].mean()
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def stats_table(predictions: pd.DataFrame, threshold: float):
    y = predictions["target"].to_numpy(float)
    rows = []
    benchmark = brier(y, predictions["vig_free_home_prob"])
    for name, col in [
        ("my_probability_odds_free", "historical_plus_sources"),
        ("team_results_only", "team_history_only"),
        ("providers_only_calibrated", "sources_only"),
        ("vig_free_market", "vig_free_home_prob"),
        ("D_Ratings", "drat_home_win_prob"),
        ("SportsDataverse", "sdv_home_win_prob"),
    ]:
        p = predictions[col].to_numpy(float)
        n = 0; units = 0.
        checked = predictions.loc[predictions["pregame_prices_verified"]]
        if not checked.empty:
            q = predictions.loc[predictions["pregame_prices_verified"], col].to_numpy(float)
            h = checked["home_decimal"].to_numpy(float)
            a = checked["away_decimal"].to_numpy(float)
            ev_h = q * h - 1.
            ev_a = (1. - q) * a - 1.
            take = np.maximum(ev_h, ev_a) > threshold
            bet_home = ev_h >= ev_a
            won = np.where(bet_home, checked["target"].to_numpy(float) == 1,
                           checked["target"].to_numpy(float) == 0)
            decimal = np.where(bet_home, h, a)
            n = int(take.sum())
            units = float(np.where(won, decimal - 1., -1.)[take].sum())
        low, high = (np.nan, np.nan) if col == "vig_free_home_prob" else bootstrap_interval(predictions, col)
        rows.append({
            "model": name, "games": len(predictions),
            "brier": round(brier(y, p), 7),
            "log_loss": round(log_loss(y, p), 7),
            "accuracy": round(float(np.mean((p >= .5) == (y == 1.))), 7),
            "brier_minus_market": round(brier(y, p) - benchmark, 7),
            "paired_date_bootstrap_low": round(low, 7),
            "paired_date_bootstrap_high": round(high, 7),
            "verified_odds_games": int(predictions["pregame_prices_verified"].sum()),
            "research_bets_at_fixed_ev": n,
            "research_unit_profit": round(units, 5) if n else np.nan,
            "research_roi": round(units / n, 6) if n else np.nan,
            "roi_is_exploratory_not_forward_betting": True,
        })
    return pd.DataFrame(rows)


def features_coef(historical: pd.DataFrame, groups, chosen):
    rows = []
    for name, features in groups.items():
        model = RidgeLogit(features, chosen[name]).fit(historical)
        for feature, beta in zip(["intercept", *features, *[f"{f}__missing" for f in features]],
                                 model.coefficients):
            rows.append({"model": name, "feature": feature, "standardized_coefficient": float(beta),
                         "regularization": chosen[name], "train_games": len(historical)})
    return pd.DataFrame(rows)


def forecast(historical, current, groups, chosen, date: pd.Timestamp):
    day_games = current.loc[current["game_date_dt"].eq(date)].copy()
    if day_games.empty:
        raise ValueError(f"No merged target games for {date.date()}")
    prior = historical.loc[historical["game_date_dt"] < date]
    if len(prior) < 250:
        raise ValueError("Insufficient past results to forecast target date")
    out = day_games[["game_id", "game_date", "game_time", "home_team", "away_team",
                     "home_decimal", "away_decimal", "vig_free_home_prob",
                     "drat_home_win_prob", "sdv_home_win_prob", "pregame_prices_verified"]].copy()
    for name, features in groups.items():
        out[name] = RidgeLogit(features, chosen[name]).fit(prior).predict(day_games)
    out["my_probability_home"] = out["historical_plus_sources"]
    out["my_probability_away"] = 1. - out["my_probability_home"]
    out["pure_team_history_probability_home"] = out["team_history_only"]
    out["my_probability_minus_market"] = out["my_probability_home"] - out["vig_free_home_prob"]
    out["home_model_ev_research_only"] = out["my_probability_home"] * out["home_decimal"] - 1.
    out["away_model_ev_research_only"] = out["my_probability_away"] * out["away_decimal"] - 1.
    out["predicted_winner"] = np.where(out["my_probability_home"] >= .5,
                                         out["home_team"], out["away_team"])
    out["max_training_game_date"] = prior["game_date_dt"].max().date().isoformat()
    out["train_games_before_target"] = len(prior)
    out["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["forecast_type"] = v3.forecast_time_status(day_games)
    out["used_sportsbook_as_model_input"] = False
    out["used_market_fallback"] = False
    out["historical_provider_capture_verified"] = False
    out["model_version"] = VERSION
    if not np.allclose(out["my_probability_home"] + out["my_probability_away"], 1., atol=1e-12):
        raise ValueError("Home and away probabilities do not sum to one")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="YYYY_MM_DD; default latest date in 01_merge")
    ap.add_argument("--skip-xg", action="store_true", help="Ignore optional locally reconstructed historical xG")
    ap.add_argument("--ev-threshold", type=float, default=.03)
    args = ap.parse_args()
    if not 0 <= args.ev_threshold <= 0.5:
        ap.error("EV threshold outside 0..0.5")
    historical, current = v1.load_history()
    historical, current, used_v3_features, coverage, xg_status = v3.assemble(historical, current, args.skip_xg)
    elo = elo_prior_results(historical, current)
    historical = historical.merge(elo, on="game_id", validate="one_to_one")
    current = current.merge(elo, on="game_id", validate="one_to_one")
    if historical["elo_diff_asof"].isna().any() or current["elo_diff_asof"].isna().any():
        raise RuntimeError("Elo features missing")
    groups = feature_groups(v3.XG_FEATURE in used_v3_features)
    report, selected, cutoff = train_validate(historical, groups)
    holdout = process_holdout(historical, groups, selected)
    summary = stats_table(holdout, args.ev_threshold)
    path = OUT / "reports"
    write(report, path / "historical_2025_hyperparameter_validation.csv")
    write(pd.DataFrame([{"model": name, "selected_penalty": selected[name],
                         "selected_using": "2025_26_only_late_validation", "holdout": "2026_27_chronological"}
                        for name in groups]), path / "model_selection.csv")
    write(holdout.drop(columns=["game_date_dt"]), path / "holdout_2026_predictions.csv")
    write(summary, path / "holdout_2026_summary.csv")
    write(features_coef(historical, groups, selected), path / "model_coefficients.csv")
    write(coverage, path / "feature_coverage.csv")
    write(xg_status, path / "xg_source_status.csv")
    day = pd.Timestamp(args.date.replace("_", "-")) if args.date else current["game_date_dt"].max()
    out = forecast(historical, current, groups, selected, day)
    outfile = OUT / "predictions" / f"{day.strftime('%Y_%m_%d')}_NHL_my_probability_independent.csv"
    write(out, outfile)
    print("2025-26 hyperparameter selection:", selected, flush=True)
    print("2026-27 CHRONOLOGICAL HOLDOUT (strictly pre-date models):", flush=True)
    print(summary[["model", "games", "brier", "log_loss", "brier_minus_market",
                   "paired_date_bootstrap_low", "paired_date_bootstrap_high", "research_roi"]].to_string(index=False))
    print("FORECAST - ODDS-FREE MY PROBABILITY; NO MARKET FALLBACK:", flush=True)
    print(out[["game_date", "away_team", "home_team", "my_probability_home",
               "my_probability_away", "vig_free_home_prob", "my_probability_minus_market"]].to_string(index=False))
    print("RESEARCH ONLY. No production selection changes; odds are benchmarks, not model features.")
    print("2025-26 odds capture timestamps AND provider prediction capture timestamps remain unauthenticated.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
        raise SystemExit(f"ERROR: {error}") from error
