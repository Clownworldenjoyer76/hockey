#!/usr/bin/env python3
"""Standalone NHL game-outcome probability model; no existing pipeline stages modified.

Install via the user's PowerShell command into:
G:\\hockey\\docs\\win\\hockey\\nhl\\scripts\\01_merge\\predict_my_probability.py

Inputs (relative to repository NHL root):
  season_master/2025_2026/sportsbook.csv
  research/sdv_challenger/season_2025/standalone_comparison.csv
  01_merge/*_NHL_merged.csv
  05_final_scores/final_scores/*_NHL_final_scores.csv

Outputs, all new and isolated under NHL root:
  my_probability/predictions/<YYYY_MM_DD>_NHL_my_probability.csv
  my_probability/reports/walkforward_predictions.csv
  my_probability/reports/walkforward_summary.csv
  my_probability/reports/training_coverage.csv

No packages beyond pandas, numpy, scipy. Historical 2025-26 sportsbook snapshots
lack a verified pregame capture timestamp: reported ROI is provisional.
All training and calibration splits exclude the prediction date and later.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

NHL = Path(__file__).resolve().parents[2]
MERGED = NHL / "01_merge"
HISTORICAL = NHL / "research" / "sdv_challenger" / "season_2025" / "standalone_comparison.csv"
PRICES = NHL / "season_master" / "2025_2026" / "sportsbook.csv"
FINAL = NHL / "05_final_scores" / "final_scores"
OUT = NHL / "my_probability"
MODEL_VERSION = "my_probability_v1"
MIN_HISTORY = 350
MIN_FIT = 240
MIN_CALIBRATION = 70
REGULARIZATION = 20.0
EPS = 1e-6

# Historical 2025-26 files lack the xG, fatigue, goalie and lineup columns.
# These features are included when present, otherwise imputed strictly from
# earlier training games with explicit missingness indicators.
FEATURES = (
    "logit_market", "logit_drat", "logit_sdv",
    "drat_exp_margin", "sdv_exp_margin", "drat_exp_total", "sdv_exp_total",
    "adj_xg_net_differential", "adj_xgf_differential", "adj_xga_differential",
    "rest_differential", "starter_gsax_differential",
    "skater_rapm_differential", "skater_war_differential",
    "home_back_to_back", "away_back_to_back",
)
EXTRA_2026 = (
    "adj_xg_net_differential", "adj_xgf_differential", "adj_xga_differential",
    "rest_differential", "starter_gsax_differential",
    "skater_rapm_differential", "skater_war_differential",
    "home_back_to_back", "away_back_to_back",
)
REQUIRED_CURRENT = (
    "game_id", "game_date", "game_time", "home_team", "away_team",
    "home_prob_moneyline", "sdv_home_win_prob", "sdv_exp_margin",
    "sdv_exp_total", "home_projected_goals", "away_projected_goals",
    "total_projected_goals", "home_dk_moneyline_american",
    "away_dk_moneyline_american", "pulled_at",
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input unavailable: {path}")
    return pd.read_csv(path, dtype={"game_id": "string"}, low_memory=False)


def assert_cols(frame: pd.DataFrame, names, label: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing required columns {missing}")


def validate_ids(df: pd.DataFrame, label: str) -> None:
    ids = df["game_id"].astype("string")
    if ids.isna().any() or not ids.str.fullmatch(r"\d{10}").all():
        raise ValueError(f"{label}: missing or noncanonical game_id")
    if ids.duplicated().any():
        raise ValueError(f"{label}: duplicate game_id")


def date_col(values) -> pd.Series:
    result = pd.to_datetime(
        pd.Series(values, copy=False).astype("string").str.replace("_", "-", regex=False),
        errors="coerce", format="%Y-%m-%d",
    )
    if result.isna().any():
        raise ValueError("Invalid or empty game_date")
    return result.dt.normalize()


def decimal_from_american(series: pd.Series) -> pd.Series:
    a = pd.to_numeric(series, errors="coerce").astype(float)
    if (a == 0).any():
        raise ValueError("Moneyline contains zero American odds")
    return pd.Series(np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / a.abs()), index=series.index)


def add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date_dt"] = date_col(df["game_date"]).to_numpy()
    df["home_decimal"] = decimal_from_american(df["home_dk_moneyline_american"])
    df["away_decimal"] = decimal_from_american(df["away_dk_moneyline_american"])
    if df[["home_decimal", "away_decimal"]].isna().any().any():
        raise ValueError("Missing home/away moneyline in training or target data")
    if (df[["home_decimal", "away_decimal"]] <= 1).any().any():
        raise ValueError("Invalid decimal moneyline")
    implied_home = 1 / df["home_decimal"]
    implied_away = 1 / df["away_decimal"]
    df["vig_free_home_prob"] = implied_home / (implied_home + implied_away)
    df["logit_market"] = logit(df["vig_free_home_prob"].clip(EPS, 1 - EPS))
    for kind, source in [("drat", "drat_home_win_prob"), ("sdv", "sdv_home_win_prob")]:
        value = pd.to_numeric(df[source], errors="coerce")
        if value.notna().any() and not value.dropna().between(0, 1).all():
            raise ValueError(f"Invalid probability in {source}")
        df[f"logit_{kind}"] = logit(value.clip(EPS, 1 - EPS))
    for f in FEATURES:
        if f not in df.columns:
            df[f] = np.nan
        df[f] = pd.to_numeric(df[f], errors="coerce")
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
    if df["logit_market"].isna().any():
        raise ValueError("Invalid vig-free market probability")
    return df


def check_current_timestamps(df: pd.DataFrame) -> None:
    # Require pregame odds and observed feature timestamps where available.
    local_date = df["game_date"].astype("string").str.replace("_", "-", regex=False)
    naive = pd.to_datetime(local_date + " " + df["game_time"].astype("string"), errors="coerce")
    if naive.isna().any():
        raise ValueError("Missing/invalid scheduled game_time; cannot verify pregame capture")
    start = naive.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    if start.isna().any():
        raise ValueError("Ambiguous/nonexistent NHL game start timezone")
    for column in ("pulled_at", "home_goalie_status_observed_at", "away_goalie_status_observed_at",
                   "home_lineup_observed_at", "away_lineup_observed_at"):
        if column not in df.columns:
            continue
        raw = df[column].astype("string").str.strip()
        parsed = pd.to_datetime(raw, utc=True, errors="coerce", format="mixed")
        if column == "pulled_at" and parsed.isna().any():
            raise ValueError("Missing odds pull timestamp; cannot validate pregame sportsbook prices")
        bad = parsed.notna() & (parsed > start)
        if bad.any():
            ids = df.loc[bad, "game_id"].astype(str).tolist()[:5]
            raise ValueError(f"{column} occurs after puck drop: game_id={ids}")


def load_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison = read_csv(HISTORICAL)
    assert_cols(comparison, ["game_id", "game_date", "drat_home_win_prob", "sdv_home_win_prob",
                             "drat_exp_margin", "sdv_exp_margin", "drat_exp_total", "sdv_exp_total",
                             "actual_home_win"], str(HISTORICAL))
    validate_ids(comparison, "2025-26 comparison")
    sportsbook = read_csv(PRICES)
    assert_cols(sportsbook, ["game_id", "home_dk_moneyline_american", "away_dk_moneyline_american"], str(PRICES))
    validate_ids(sportsbook, "2025-26 sportsbook")
    prev = comparison.merge(sportsbook[["game_id", "home_dk_moneyline_american",
                                       "away_dk_moneyline_american"]], on="game_id",
                            validate="one_to_one", how="left")
    if prev[["home_dk_moneyline_american", "away_dk_moneyline_american"]].isna().any().any():
        raise ValueError("Historical comparison contains games without moneyline prices")
    prev["data_period"] = "2025_2026"
    prev["pregame_prices_verified"] = False  # No historical timestamp in season_master.
    prev["target"] = pd.to_numeric(prev["actual_home_win"], errors="raise")
    if not prev["target"].isin([0, 1]).all():
        raise ValueError("Historical target labels are not binary")
    prev = add_common_features(prev)

    paths = sorted(MERGED.glob("*_NHL_merged.csv"))
    if not paths:
        raise FileNotFoundError(f"No current season merged files in {MERGED}")
    cur = pd.concat([read_csv(path) for path in paths], ignore_index=True)
    assert_cols(cur, REQUIRED_CURRENT, "Current 01_merge inputs")
    validate_ids(cur, "2026-27 merged")
    cur["drat_home_win_prob"] = cur["home_prob_moneyline"]
    cur["drat_exp_margin"] = pd.to_numeric(cur["home_projected_goals"], errors="coerce") - pd.to_numeric(cur["away_projected_goals"], errors="coerce")
    cur["drat_exp_total"] = cur["total_projected_goals"]
    check_current_timestamps(cur)
    cur["data_period"] = "2026_2027"
    cur["pregame_prices_verified"] = True  # Pulled_at checked against scheduled start.
    cur = add_common_features(cur)

    finals = sorted(FINAL.glob("*_NHL_final_scores.csv"))
    if not finals:
        raise FileNotFoundError(f"No current final scores in {FINAL}")
    score = pd.concat([read_csv(path) for path in finals], ignore_index=True)
    assert_cols(score, ["game_id", "home_score", "away_score"], "Current final scores")
    validate_ids(score, "2026-27 final scores")
    cur = cur.merge(score[["game_id", "home_score", "away_score"]], on="game_id", how="left",
                    validate="one_to_one")
    home_score = pd.to_numeric(cur["home_score"], errors="coerce")
    away_score = pd.to_numeric(cur["away_score"], errors="coerce")
    incomplete = home_score.isna() | away_score.isna()
    if (home_score[~incomplete] == away_score[~incomplete]).any():
        raise ValueError("Final NHL result contains equal scores")
    cur["target"] = np.where(incomplete, np.nan, (home_score > away_score).astype(float))
    # Restrict to the official NHL regular-season game code: yyyy02xxxx.
    prev = prev.loc[prev["game_id"].str.slice(4, 6).eq("02")].copy()
    cur = cur.loc[cur["game_id"].str.slice(4, 6).eq("02")].copy()
    history = pd.concat([prev, cur.loc[cur["target"].notna()]], ignore_index=True)
    if history["game_id"].duplicated().any():
        raise ValueError("Duplicate game_id across training seasons")
    history = history.sort_values(["game_date_dt", "game_id"]).reset_index(drop=True)
    return history, cur


@dataclass
class Scaler:
    median: np.ndarray
    average: np.ndarray
    std: np.ndarray


def fit_transform(train: pd.DataFrame) -> tuple[Scaler, np.ndarray]:
    raw = train.loc[:, FEATURES].to_numpy(dtype=float)
    nanmask = ~np.isfinite(raw)
    medians = np.array([float(np.median(raw[~nanmask[:, j], j]))
                        if (~nanmask[:, j]).any() else 0.0
                        for j in range(raw.shape[1])])
    filled = np.where(nanmask, medians, raw)
    averages = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x = np.column_stack(((filled - averages) / std, nanmask.astype(float)))
    return Scaler(medians, averages, std), x


def apply_transform(scaler: Scaler, frame: pd.DataFrame) -> np.ndarray:
    raw = frame.loc[:, FEATURES].to_numpy(dtype=float)
    nanmask = ~np.isfinite(raw)
    filled = np.where(nanmask, scaler.median, raw)
    return np.column_stack(((filled - scaler.average) / scaler.std, nanmask.astype(float)))


def optimize_logistic(x: np.ndarray, y: np.ndarray, penalty: float, initial=None, prior=None) -> np.ndarray:
    design = np.column_stack((np.ones(len(x)), x))
    if initial is None:
        initial = np.zeros(design.shape[1])
        initial[0] = logit(np.clip(float(y.mean()), EPS, 1 - EPS))
    if prior is None:
        prior = np.zeros(design.shape[1])
    penalize = np.ones(design.shape[1]); penalize[0] = 0.0

    def cost_grad(b):
        z = np.clip(design @ b, -35, 35)
        probs = expit(z)
        diff = b - prior
        loss = float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * penalty * np.sum(penalize * diff * diff))
        gradient = design.T @ (probs - y) + penalty * penalize * diff
        return loss, gradient

    result = minimize(cost_grad, initial, jac=True, method="L-BFGS-B", options={"maxiter": 150, "ftol": 1e-9})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"Model optimization failed: {result.message}")
    return result.x


@dataclass
class Predictor:
    scaler: Scaler
    beta: np.ndarray
    cal_beta: np.ndarray
    train_rows: int
    fit_rows: int
    calibration_rows: int
    max_training_date: str

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = apply_transform(self.scaler, df)
        raw = expit(np.column_stack((np.ones(len(x)), x)) @ self.beta)
        z = logit(np.clip(raw, EPS, 1 - EPS))
        result = expit(self.cal_beta[0] + self.cal_beta[1] * z)
        return np.clip(result, EPS, 1 - EPS)


def fit_past(history: pd.DataFrame, day: pd.Timestamp) -> Predictor | None:
    past = history.loc[history["game_date_dt"] < day].copy()
    if len(past) < MIN_HISTORY or past["target"].nunique() < 2:
        return None
    dates = past["game_date_dt"].drop_duplicates().sort_values().to_numpy()
    # Calibrator uses the most recent 20% of DISTINCT historical dates.
    # Holding out whole dates prevents same-day training leakage.
    split_ix = max(1, int(len(dates) * 0.8))
    cutoff = pd.Timestamp(dates[min(split_ix, len(dates) - 1)])
    fit = past.loc[past["game_date_dt"] < cutoff]
    cal = past.loc[past["game_date_dt"] >= cutoff]
    if len(fit) < MIN_FIT or len(cal) < MIN_CALIBRATION or fit["target"].nunique() < 2:
        return None
    scaler, x_fit = fit_transform(fit)
    y_fit = fit["target"].to_numpy(dtype=float)
    beta = optimize_logistic(x_fit, y_fit, REGULARIZATION)
    x_cal = apply_transform(scaler, cal)
    p_cal = expit(np.column_stack((np.ones(len(x_cal)), x_cal)) @ beta)
    z = logit(np.clip(p_cal, EPS, 1 - EPS))
    y_cal = cal["target"].to_numpy(dtype=float)
    # Platt scaling: slope is regularized toward 1, not free to drift wildly.
    cal_beta = optimize_logistic(z[:, None], y_cal, penalty=5.0,
                                 initial=np.array([0.0, 1.0]), prior=np.array([0.0, 1.0]))
    return Predictor(scaler, beta, cal_beta, len(past), len(fit), len(cal),
                     past["game_date_dt"].max().date().isoformat())


def predict_rows(predictor: Predictor, rows: pd.DataFrame) -> pd.DataFrame:
    out = rows[["game_id", "game_date", "home_team", "away_team", "data_period",
                "home_decimal", "away_decimal", "vig_free_home_prob",
                "drat_home_win_prob", "sdv_home_win_prob", "pregame_prices_verified"]].copy()
    out["my_probability_home"] = predictor.predict(rows)
    out["my_probability_away"] = 1.0 - out["my_probability_home"]
    out["predicted_winner"] = np.where(out["my_probability_home"] >= 0.5,
                                        out["home_team"], out["away_team"])
    out["home_ev_per_unit"] = out["my_probability_home"] * out["home_decimal"] - 1
    out["away_ev_per_unit"] = out["my_probability_away"] * out["away_decimal"] - 1
    out["training_games"] = predictor.train_rows
    out["training_cutoff_date"] = predictor.max_training_date
    out["calibration_games"] = predictor.calibration_rows
    out["model_version"] = MODEL_VERSION
    return out


def pnl_for_prob(df: pd.DataFrame, home_prob: np.ndarray, ev_threshold: float):
    home_ev = home_prob * df["home_decimal"].to_numpy(float) - 1
    away_ev = (1 - home_prob) * df["away_decimal"].to_numpy(float) - 1
    choose_home = home_ev >= away_ev
    max_ev = np.maximum(home_ev, away_ev)
    bet = max_ev > ev_threshold
    home_win = df["target"].to_numpy(float) == 1.0
    win = np.where(choose_home, home_win, ~home_win)
    payouts = np.where(choose_home, df["home_decimal"].to_numpy(float),
                       df["away_decimal"].to_numpy(float))
    profit = np.where(win, payouts - 1, -1)
    return bet, float(np.sum(profit[bet]))


def score_one(df: pd.DataFrame, name: str, p: np.ndarray, threshold: float) -> dict:
    y = df["target"].to_numpy(dtype=float)
    p = np.clip(p.astype(float), EPS, 1 - EPS)
    if not np.isfinite(p).all():
        raise ValueError(f"{name} produced missing probabilities")
    bet, units = pnl_for_prob(df, p, threshold)
    return {"model": name, "games": len(df), "brier": round(float(np.mean((p - y) ** 2)), 8),
            "log_loss": round(float(np.mean(-y * np.log(p) - (1-y) * np.log(1-p))), 8),
            "win_accuracy": round(float(np.mean((p >= 0.5) == (y == 1))), 8),
            "fixed_ev_threshold": threshold, "one_unit_bets": int(bet.sum()),
            "one_unit_profit": round(units, 6),
            "roi": round(units / int(bet.sum()), 8) if bet.sum() else np.nan}


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"WROTE {path} | rows={len(df)}")


def evaluate(history: pd.DataFrame, threshold: float) -> None:
    frames = []
    dates = sorted(history["game_date_dt"].drop_duplicates())
    total = len(dates)
    for index, day in enumerate(dates, 1):
        model = fit_past(history, day)
        if model is None:
            continue
        rows = history.loc[history["game_date_dt"] == day]
        pred = predict_rows(model, rows)
        pred["target"] = rows["target"].to_numpy(dtype=float)
        frames.append(pred)
        if index % 30 == 0:
            print(f"BACKTEST {index}/{total} days")
    if not frames:
        raise ValueError("No walk-forward predictions: insufficient prior data to fit and calibrate")
    scored = pd.concat(frames, ignore_index=True).sort_values(["game_date", "game_id"])
    # Fold predictions were generated from strictly earlier completed game dates.
    write_csv(scored, OUT / "reports" / "walkforward_predictions.csv")
    summaries = []
    for season, group in [("ALL", scored), *list(scored.groupby("data_period", sort=True))]:
        baselines = {
            "my_probability": group["my_probability_home"].to_numpy(float),
            "vig_free_market": group["vig_free_home_prob"].to_numpy(float),
            "D_Ratings": group["drat_home_win_prob"].to_numpy(float),
            "SportsDataverse": group["sdv_home_win_prob"].to_numpy(float),
        }
        for name, probs in baselines.items():
            entry = score_one(group, name, probs, threshold)
            entry["season"] = season
            entry["historical_odds_timestamp_verified"] = bool(group["pregame_prices_verified"].all())
            summaries.append(entry)
    report = pd.DataFrame(summaries)
    write_csv(report, OUT / "reports" / "walkforward_summary.csv")
    print(report.to_string(index=False))


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--date", help="Predict specific current merged date, format YYYY_MM_DD; default latest")
    cli.add_argument("--only", choices=["both", "predict", "backtest"], default="both")
    cli.add_argument("--ev-threshold", type=float, default=0.03,
                     help="Fixed historical ROI evaluation threshold; not optimized")
    args = cli.parse_args()
    if not (0 <= args.ev_threshold <= 0.5):
        cli.error("--ev-threshold must be between 0 and 0.5")
    history, current = load_history()
    coverage = []
    for period, frame in history.groupby("data_period"):
        for f in FEATURES:
            coverage.append({"season": period, "feature": f, "training_games": len(frame),
                             "nonmissing_games": int(frame[f].notna().sum()),
                             "nonmissing_fraction": round(float(frame[f].notna().mean()), 5)})
    write_csv(pd.DataFrame(coverage), OUT / "reports" / "training_coverage.csv")
    print(f"TRAINING HISTORY: {len(history)} completed games; current merged: {len(current)} games")
    if args.only in ("both", "backtest"):
        evaluate(history, args.ev_threshold)
    if args.only in ("both", "predict"):
        day = pd.to_datetime(args.date.replace("_", "-")) if args.date else current["game_date_dt"].max()
        rows = current.loc[current["game_date_dt"] == day].copy()
        if rows.empty:
            raise ValueError(f"No merged games for requested date {day.date()}")
        model = fit_past(history, day)
        if model is None:
            raise ValueError(f"Insufficient earlier labeled history for {day.date()}")
        pred = predict_rows(model, rows)
        pred["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        pred["forecast_type"] = ("historical_reconstruction" if day.date() < datetime.now(ZoneInfo("America/New_York")).date()
                                 else "pregame_or_current")
        key = day.strftime("%Y_%m_%d")
        write_csv(pred, OUT / "predictions" / f"{key}_NHL_my_probability.csv")
        print(pred[["game_date", "away_team", "home_team", "my_probability_home",
                    "my_probability_away", "predicted_winner"]].to_string(index=False))
    print("NOTE: Historical 2025-26 sportsbook timestamps are unavailable; historical ROI is provisional.")
    print("NOTE: Historical xG/rest/goalie/lineup features absent; their incremental gain is unproven.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"ERROR: {exc}")
