#!/usr/bin/env python3
"""Standalone my_probability v3: pregame-safe schedule/form features and guarded market residual.

Repository location (user installs via PowerShell):
  G:\\hockey\\docs\\win\\hockey\\nhl\\scripts\\01_merge\\predict_my_probability_v3.py
Dependencies: predict_my_probability.py (v1) in this same folder, pandas/numpy/scipy.

ALL writes are isolated to: G:\\hockey\\docs\\win\\hockey\\nhl\\my_probability\\v3\\
Existing production intake, merge, odds, selection and model files are never edited.

Evidence and limitations:
- Historical 2025-26 results come from the repo's SDV challenger comparison; names
  and schedule dates from season_master/2025_2026/games.csv. Current season result
  labels and odds are loaded by the existing v1 reader.
- Rest and rolling scored-form use ONLY completed game outcomes with source date
  STRICTLY before the predicted game's calendar date. Same-day games never leak.
- Optional xG reconstruction uses the repo's SportsDataverse P5 Parquet store and
  repo's preexisting, date-gated SDV-P6 rating calculations. It never reads final
  target-game xG. No P5 data are downloaded. Missing seasons remain unavailable.
- Historical 2025-26 moneyline capture times cannot be verified from the season
  master and MUST NOT be counted as proven historical betting prices. Missing
  historical pregame goalie identities cannot be inferred from actual starters.
- 2026-27 timestamps are checked by the v1 loader; no new timestamp is invented.
- This is a research backtest. No bet-selection integration is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import importlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

import predict_my_probability as v1

NHL = v1.NHL
OUT = NHL / "my_probability" / "v3"
MASTER = NHL / "season_master" / "2025_2026" / "games.csv"
P5 = NHL / "research" / "sdv_history"
VERSION = "v3_historical_asof_form_2026-11-19"
EPS = 1e-6
MIN_TRAIN = 350
MIN_TRAIN_FIT = 260
MIN_CAL = 80
RIDGE = 60.0
CAL_WEIGHTS = np.array([0., .15, .30, .45, .60, .75, .90, 1.00], dtype=float)
BRIER_MARGIN = 0.0004
LOSS_MARGIN = 0.0008
GUARD_N = 180
GUARD_WINDOW = 350

BASE_FEATURES = (
    "drat_market_logit_delta",
    "sdv_market_logit_delta",
    "rest_diff_asof",
    "back_to_back_diff_asof",
    "games_in_four_diff_asof",
    "win_rate_last10_diff_asof",
    "goals_diff_last10_diff_asof",
    "goals_for_last10_diff_asof",
    "goals_against_last10_diff_asof",
    "prior_games_min_asof",
)
XG_FEATURE = "xg_net_diff_prior_asof"


def required(df: pd.DataFrame, names: list[str], source: str) -> None:
    absent = sorted(set(names) - set(df.columns))
    if absent:
        raise ValueError(f"{source} missing columns: {absent}")


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"WROTE {path} | rows={len(df)}", flush=True)


def scores_and_names(history: pd.DataFrame, current: pd.DataFrame):
    """2025 names are obtained by exact NHL game_id joins (not fuzzy matching)."""
    master = v1.read_csv(MASTER)
    required(master, ["game_id", "home_team", "away_team", "game_date"], str(MASTER))
    master = master.loc[master["game_id"].str.slice(4, 6).eq("02")].copy()
    v1.validate_ids(master, "2025 season master")

    history = history.copy()
    current = current.copy()
    is_prev = history["data_period"].eq("2025_2026")
    hist_prev = history.loc[is_prev].copy()
    if hist_prev.empty:
        raise ValueError("No historical 2025-26 match history")
    lookup = master.set_index("game_id")
    expected_dates = hist_prev["game_id"].map(lookup["game_date"])
    if not expected_dates.astype("string").str.replace("_", "-", regex=False).eq(
        hist_prev["game_date"].astype("string").str.replace("_", "-", regex=False)).all():
        raise ValueError("2025-26 season master dates conflict with historical comparison dates")
    if not hist_prev["game_id"].isin(lookup.index).all():
        raise ValueError("2025-26 results cannot all be matched to the official season master")
    hist_prev["home_team_full"] = hist_prev["game_id"].map(lookup["home_team"])
    hist_prev["away_team_full"] = hist_prev["game_id"].map(lookup["away_team"])
    if hist_prev[["home_team_full", "away_team_full"]].isna().any().any():
        raise ValueError("Missing 2025-26 names after game_id join")

    # Exact historical mapping for optional xG joins. Fails on conflicting names.
    aliases = pd.concat([
        hist_prev[["home_team_full", "home_team"]].rename(columns={"home_team_full": "name", "home_team": "abbr"}),
        hist_prev[["away_team_full", "away_team"]].rename(columns={"away_team_full": "name", "away_team": "abbr"}),
    ], ignore_index=True).drop_duplicates()
    ambiguity = aliases.groupby("name")["abbr"].nunique()
    if (ambiguity > 1).any():
        raise ValueError("Historical NHL team name maps to conflicting abbreviations")
    abbreviations = aliases.set_index("name")["abbr"].to_dict()

    # Note: current-season outcome rows come from date-stamped official scores.
    cur = history.loc[~is_prev].copy()
    cur["home_team_full"] = cur["home_team"]
    cur["away_team_full"] = cur["away_team"]
    history = pd.concat([hist_prev, cur], ignore_index=True)
    current["home_team_full"] = current["home_team"]
    current["away_team_full"] = current["away_team"]

    for frame in (history, current):
        if frame[["home_team_full", "away_team_full"]].isna().any().any():
            raise ValueError("Missing canonical team name")
        for side in ("home", "away"):
            frame[f"{side}_team_abbr"] = frame[f"{side}_team_full"].map(abbreviations)
    history = history.sort_values(["game_date_dt", "game_id"]).reset_index(drop=True)
    current = current.sort_values(["game_date_dt", "game_id"]).reset_index(drop=True)
    return history, current, abbreviations


def build_pregame_form(history: pd.DataFrame, current: pd.DataFrame):
    """Snapshot team form BEFORE ingesting any results for a target day."""
    completed = history.copy()
    if completed[["home_score", "away_score"]].isna().any().any():
        raise ValueError("Completed results missing home or away score")
    for col in ["home_score", "away_score"]:
        completed[col] = pd.to_numeric(completed[col], errors="raise")
    if ((completed["home_score"] - completed["away_score"]) == 0).any():
        raise ValueError("Final games with tied score cannot be labeled binary")

    all_targets = pd.concat([
        history[["game_id", "game_date_dt", "data_period", "home_team_full", "away_team_full"]],
        current[["game_id", "game_date_dt", "data_period", "home_team_full", "away_team_full"]],
    ], ignore_index=True).drop_duplicates(subset=["game_id"], keep="first")
    if all_targets["game_id"].duplicated().any():
        raise ValueError("Duplicate target game IDs")

    # Each season's recorded recent form is reset at season boundary.
    # No same-day outcomes are consumed by the feature generator.
    ledger = defaultdict(lambda: deque(maxlen=12))
    records = []
    for season, season_targets in all_targets.groupby("data_period", sort=True):
        ledger.clear()
        completed_season = completed.loc[completed["data_period"].eq(season)].copy()
        completed_by_day = {day: frame for day, frame in completed_season.groupby("game_date_dt", sort=True)}
        for target_day, targets in season_targets.groupby("game_date_dt", sort=True):
            for row in targets.itertuples(index=False):
                home = ledger[row.home_team_full]
                away = ledger[row.away_team_full]

                def snapshot(past):
                    recent = [g for g in past if g[0] < target_day]
                    last10 = recent[-10:]
                    last_day = recent[-1][0] if recent else None
                    # Game-dates from completed games strictly before target day.
                    rest = (target_day - last_day).days - 1 if last_day is not None else np.nan
                    last4 = sum((target_day - g[0]).days <= 4 for g in recent)
                    if last10:
                        win_rate = float(np.mean([g[1] for g in last10]))
                        gf = float(np.mean([g[2] for g in last10]))
                        ga = float(np.mean([g[3] for g in last10]))
                    else:
                        win_rate = gf = ga = np.nan
                    return dict(rest=rest, b2b=int(rest == 0) if np.isfinite(rest) else np.nan,
                                games4=last4, win=win_rate, gf=gf, ga=ga,
                                gd=gf - ga if np.isfinite(gf) else np.nan,
                                n=len(recent))

                h, a = snapshot(home), snapshot(away)
                def diff(k):
                    if not np.isfinite(h[k]) or not np.isfinite(a[k]):
                        return np.nan
                    return h[k] - a[k]
                records.append({
                    "game_id": row.game_id,
                    "form_source_season": season,
                    "form_cutoff_exclusive": target_day.date().isoformat(),
                    "rest_diff_asof": diff("rest"),
                    "back_to_back_diff_asof": diff("b2b"),
                    "games_in_four_diff_asof": diff("games4"),
                    "win_rate_last10_diff_asof": diff("win"),
                    "goals_diff_last10_diff_asof": diff("gd"),
                    "goals_for_last10_diff_asof": diff("gf"),
                    "goals_against_last10_diff_asof": diff("ga"),
                    "prior_games_min_asof": min(h["n"], a["n"]),
                    "form_complete_both_teams": int(h["n"] > 0 and a["n"] > 0),
                })
            # Advance team game histories only AFTER every target on this date is snapshotted.
            for past in completed_by_day.get(target_day, pd.DataFrame()).itertuples(index=False):
                date = past.game_date_dt
                hg = float(past.home_score)
                ag = float(past.away_score)
                ledger[past.home_team_full].append((date, int(hg > ag), hg, ag))
                ledger[past.away_team_full].append((date, int(ag > hg), ag, hg))
    form = pd.DataFrame(records)
    if form["game_id"].duplicated().any() or len(form) != len(all_targets):
        raise ValueError("Form feature cardinality mismatch")
    return form


def optional_xg_features(frames: list[pd.DataFrame], abbreviations: dict, skip: bool):
    """Use only local P5 history, and strictly past-date prior_adjusted_ratings."""
    out = {str(gid): np.nan for f in frames for gid in f["game_id"]}
    status = []
    if skip:
        return out, [{"season": "ALL", "status": "user_skipped", "rows_filled": 0}]
    paths_available = {}
    for season in ["2025_2026", "2026_2027"]:
        year = int(season[:4])
        schedules = P5 / "schedules" / f"season_{year}.parquet"
        pbp = P5 / "play_by_play" / f"season_{year}.parquet"
        paths_available[season] = schedules.is_file() and pbp.is_file()
        if not paths_available[season]:
            status.append({"season": season, "status": "P5_schedule_or_pbp_missing", "rows_filled": 0,
                           "schedule_path": str(schedules), "pbp_path": str(pbp)})
    if not any(paths_available.values()):
        return out, status
    research_path = NHL / "scripts" / "research"
    if not research_path.is_dir():
        raise FileNotFoundError(f"SDV research scripts missing: {research_path}")
    if str(research_path) not in sys.path:
        sys.path.insert(0, str(research_path))
    try:
        challenger = importlib.import_module("build_sdv_challenger")
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"P5 exists but cannot import SDV challenger: {exc}") from exc

    for season in ["2025_2026", "2026_2027"]:
        if not paths_available[season]:
            continue
        year = int(season[:4])
        print(f"BUILDING AS-OF xG from local P5 data | season {season}", flush=True)
        _, game_rates = challenger.load_p5_game_rates(year)
        date_to_games = {}
        targets = pd.concat([
            f.loc[f["data_period"].eq(season), ["game_id", "game_date_dt", "home_team_full", "away_team_full"]]
            for f in frames
        ], ignore_index=True).drop_duplicates("game_id")
        for day, same_day in targets.groupby("game_date_dt", sort=True):
            date_to_games[day] = same_day
        filled = 0
        for day, games in date_to_games.items():
            prior = challenger.ratings_from_prior_games(game_rates, day.date())
            if prior.is_empty():
                continue
            rating = prior.select(["team", "adj_xg_net"]).to_pandas()
            idx = rating.set_index("team")["adj_xg_net"].to_dict()
            for row in games.itertuples(index=False):
                ha = abbreviations.get(row.home_team_full)
                aa = abbreviations.get(row.away_team_full)
                if ha in idx and aa in idx:
                    out[str(row.game_id)] = float(idx[ha]) - float(idx[aa])
                    filled += 1
        status.append({"season": season, "status": "computed_asof_prior_games",
                       "rows_filled": filled,
                       "schedule_path": str(P5 / "schedules" / f"season_{year}.parquet"),
                       "pbp_path": str(P5 / "play_by_play" / f"season_{year}.parquet")})
    return out, status


def assemble(history: pd.DataFrame, current: pd.DataFrame, skip_xg: bool):
    history, current, abbreviations = scores_and_names(history, current)
    form = build_pregame_form(history, current)
    frames = [history, current]
    xg_map, xg_status = optional_xg_features(frames, abbreviations, skip_xg)
    for i, frame in enumerate(frames):
        frame = frame.merge(form, on="game_id", how="left", validate="one_to_one")
        frame[XG_FEATURE] = frame["game_id"].astype(str).map(xg_map).astype(float)
        if frame["form_cutoff_exclusive"].isna().any():
            raise ValueError("Unmatched reconstructed as-of form features")
        frames[i] = frame
    history, current = frames
    # Include xG only when BOTH seasons have >50% coverage among labeled rows.
    # Otherwise a season-shift missingness artifact could dominate backtesting.
    ratios = history.groupby("data_period")[XG_FEATURE].apply(lambda s: float(s.notna().mean()))
    use_xg = all(float(ratios.get(s, 0.0)) >= 0.5 for s in ["2025_2026", "2026_2027"])
    features = BASE_FEATURES + ((XG_FEATURE,) if use_xg else ())
    for frame in (history, current):
        frame["drat_market_logit_delta"] = frame["logit_drat"] - frame["logit_market"]
        frame["sdv_market_logit_delta"] = frame["logit_sdv"] - frame["logit_market"]
        for col in features:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    coverage = []
    for name, frame in [("historical_scored", history), ("current_merged", current)]:
        for season, part in frame.groupby("data_period", sort=True):
            for feat in [*BASE_FEATURES, XG_FEATURE, "form_complete_both_teams"]:
                coverage.append({"dataset": name, "season": season, "feature": feat,
                                 "games": len(part), "nonmissing": int(part[feat].eq(1).sum()) if feat == "form_complete_both_teams" else int(part[feat].notna().sum()),
                                 "coverage": float(part[feat].eq(1).mean()) if feat == "form_complete_both_teams" else float(part[feat].notna().mean()),
                                 "selected_for_model": feat in features})
    return history, current, features, pd.DataFrame(coverage), pd.DataFrame(xg_status)


@dataclass
class OffsetModel:
    med: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    beta: np.ndarray
    names: tuple[str, ...]

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        x = transform(data, self.names, self.med, self.mean, self.sd)
        logit_market = data["logit_market"].to_numpy(float)
        return np.clip(expit(logit_market + np.column_stack([np.ones(len(x)), x]) @ self.beta), EPS, 1-EPS)


def matrix(frame: pd.DataFrame, names: tuple[str, ...]):
    values = frame[list(names)].to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    med = np.array([np.median(values[~missing[:, j], j]) if (~missing[:, j]).any() else 0.0
                    for j in range(values.shape[1])], dtype=float)
    full = np.where(missing, med, values)
    mean = full.mean(axis=0)
    sd = full.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    x = np.column_stack(((full - mean)/sd, missing.astype(float)))
    return med, mean, sd, x


def transform(frame, names, med, mean, sd):
    values = frame[list(names)].to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    full = np.where(missing, med, values)
    return np.column_stack(((full-mean)/sd, missing.astype(float)))


def fit_offset(frame: pd.DataFrame, names: tuple[str, ...]) -> OffsetModel:
    med, mu, sd, x = matrix(frame, names)
    y = frame["target"].to_numpy(float)
    lm = frame["logit_market"].to_numpy(float)
    design = np.column_stack([np.ones(len(x)), x])
    def objective(beta):
        z = np.clip(lm + design @ beta, -30., 30.)
        p = expit(z)
        penalty = np.r_[0.5, np.ones(len(beta)-1)] * RIDGE
        obj = float(np.sum(np.logaddexp(0, z)-y*z) + 0.5*np.sum(penalty*beta**2))
        grad = design.T @ (p-y) + penalty*beta
        return obj, grad
    initial = np.zeros(design.shape[1], dtype=float)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": 120, "ftol": 1e-8})
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Regularized market residual fit failed: {result.message}")
    return OffsetModel(med, mu, sd, result.x, names)


def brier(y, p):
    return float(np.mean((np.asarray(p, float)-np.asarray(y, float))**2))


def logloss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1-EPS)
    y = np.asarray(y, float)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))


def choose_weight(frame: pd.DataFrame, raw: np.ndarray, *, threshold_n: int, margin: bool):
    n = len(frame)
    if n < threshold_n:
        return 0., "insufficient_past_calibration"
    y = frame["target"].to_numpy(float)
    market = frame["vig_free_home_prob"].to_numpy(float)
    best = min(CAL_WEIGHTS, key=lambda w: (logloss(y, market + w*(raw-market)) + .0015*w*w, float(w)))
    best = float(best)
    if best <= 0:
        return 0., "market_better_on_past_calibration"
    p = market + best*(raw-market)
    if margin and (brier(y, market)-brier(y, p) < BRIER_MARGIN or
                   logloss(y, market)-logloss(y, p) < LOSS_MARGIN):
        return 0., "improvement_below_safety_margin"
    return best, "past_calibration_supported_adjustment"


def fit_predict_prior(history: pd.DataFrame, day: pd.Timestamp, names: tuple[str, ...]):
    past = history.loc[history["game_date_dt"] < day].copy()
    if len(past) < MIN_TRAIN or past["target"].nunique() < 2:
        return None
    days = sorted(past["game_date_dt"].unique())
    cut = pd.Timestamp(days[int(len(days)*.8)])
    train = past.loc[past["game_date_dt"] < cut].copy()
    cal = past.loc[past["game_date_dt"] >= cut].copy()
    if len(train) < MIN_TRAIN_FIT or len(cal) < MIN_CAL or train["target"].nunique() < 2:
        return None
    temp = fit_offset(train, names)
    raw_cal = temp.predict(cal)
    cal_weight, cal_status = choose_weight(cal, raw_cal, threshold_n=MIN_CAL, margin=True)
    # Calibrator is selected only on past out-of-time holdout. Base model
    # is then refit on all past data for target date.
    full = fit_offset(past, names)
    return full, len(past), cal_weight, cal_status, past["game_date_dt"].max()


def prior_guard(previous: pd.DataFrame, season: str):
    subset = previous.loc[previous["data_period"].eq(season)].tail(GUARD_WINDOW)
    if len(subset) < GUARD_N:
        return 0., "insufficient_out_of_fold_validation"
    p = subset["raw_v3_probability_home"].to_numpy(float)
    w, state = choose_weight(subset, p, threshold_n=GUARD_N, margin=True)
    return w, state


def row_predictions(games: pd.DataFrame, model_info, past_oof: pd.DataFrame):
    model, n, cal_weight, cal_status, maxday = model_info
    raw = model.predict(games)
    market = games["vig_free_home_prob"].to_numpy(float)
    guard = {season: prior_guard(past_oof, season) for season in games["data_period"].unique()}
    weights, flags = [], []
    for season in games["data_period"].astype(str):
        check_weight, state = guard[season]
        weight = min(cal_weight, check_weight)
        weights.append(weight)
        flags.append(state if weight == 0 else "prior_calibration_and_oof_supported")
    weights = np.asarray(weights, float)
    pred = np.clip(market + weights*(raw-market), EPS, 1-EPS)
    out = games[["game_id", "game_date", "data_period", "home_team", "away_team",
                 "home_decimal", "away_decimal", "vig_free_home_prob",
                 "drat_home_win_prob", "sdv_home_win_prob", "pregame_prices_verified",
                 "form_cutoff_exclusive", "form_complete_both_teams", XG_FEATURE]].copy()
    out["raw_v3_probability_home"] = raw
    out["my_probability_home"] = pred
    out["my_probability_away"] = 1.0-pred
    out["market_blend_weight"] = weights
    out["adjustment_status"] = flags
    out["calibration_status"] = cal_status
    out["prior_history_games"] = n
    out["max_source_result_date"] = maxday.date().isoformat()
    out["predicted_winner"] = np.where(pred >= .5, out["home_team"], out["away_team"])
    out["home_expected_value"] = pred*out["home_decimal"].to_numpy(float)-1
    out["away_expected_value"] = (1.-pred)*out["away_decimal"].to_numpy(float)-1
    out["feature_asof_verified_form_only"] = True
    out["historical_odds_timestamps_verified"] = out["pregame_prices_verified"]
    out["historical_goalie_starter_verified"] = False
    out["model_version"] = VERSION
    if not np.allclose(out.my_probability_home+out.my_probability_away,1.,atol=1e-12,rtol=0):
        raise ValueError("Complement probabilities do not sum to one")
    return out


def walkforward(history: pd.DataFrame, names: tuple[str, ...]):
    frames=[]
    # Prior OOF guard uses only earlier date's predictions; no current outcomes.
    prior = pd.DataFrame(columns=["data_period","target","vig_free_home_prob","raw_v3_probability_home"])
    days = sorted(history["game_date_dt"].unique())
    for count, day in enumerate(days,1):
        day = pd.Timestamp(day)
        fitted = fit_predict_prior(history, day, names)
        if fitted is None:
            continue
        games=history.loc[history["game_date_dt"].eq(day)].copy()
        scored=row_predictions(games,fitted,prior)
        scored["target"]=games["target"].to_numpy(float)
        frames.append(scored)
        prior=scored.copy() if prior.empty else pd.concat([prior,scored],ignore_index=True)
        if count%40==0:
            print(f"WALKFORWARD {count}/{len(days)} dates",flush=True)
    if not frames:
        raise ValueError("Insufficient history for walk-forward evaluation")
    return pd.concat(frames,ignore_index=True).sort_values(["game_date","game_id"]).reset_index(drop=True)


def ROI(df,p,threshold):
    # Include ONLY games with a verified pregame price timestamp.
    sub=df.loc[df["pregame_prices_verified"]].copy()
    if sub.empty:
        return 0, np.nan, np.nan
    mask=df["pregame_prices_verified"].to_numpy(bool)
    p=np.asarray(p,float)[mask]
    he=p*sub["home_decimal"].to_numpy(float)-1
    ae=(1-p)*sub["away_decimal"].to_numpy(float)-1
    bet=(np.maximum(he,ae)>threshold)
    choose_home=(he>=ae)
    wins=sub["target"].to_numpy(float)>0.5
    ywin=np.where(choose_home,wins,~wins)
    payout=np.where(choose_home,sub["home_decimal"].to_numpy(float),sub["away_decimal"].to_numpy(float))
    units=float(np.sum(np.where(ywin,payout-1,-1)[bet]))
    n=int(bet.sum())
    return n,round(units,6),round(units/n,7) if n else np.nan


def summary_table(scored: pd.DataFrame, threshold: float):
    metrics=[]
    for period, grp in [("ALL",scored),*list(scored.groupby("data_period",sort=True))]:
        y=grp["target"].to_numpy(float)
        for label,key in [("my_probability_v3","my_probability_home"),
                          ("raw_v3","raw_v3_probability_home"),
                          ("vig_free_market","vig_free_home_prob"),
                          ("D_Ratings","drat_home_win_prob"),
                          ("SportsDataverse","sdv_home_win_prob")]:
            p=grp[key].to_numpy(float)
            n,units,roi=ROI(grp,p,threshold)
            metrics.append({"season":period,"model":label,"games":len(grp),
                            "brier":round(brier(y,p),8),"log_loss":round(logloss(y,p),8),
                            "accuracy":round(float(np.mean((p>=0.5)==(y==1))),7),
                            "verified_pregame_odds_rows":int(grp["pregame_prices_verified"].sum()),
                            "eligible_roi_bets":n,"verified_timestamp_units":units,"verified_timestamp_roi":roi,
                            "historical_2025_odds_unverified":bool((grp["data_period"]=="2025_2026").any()),
                            "nonmarket_probability_rows":int((grp["market_blend_weight"]>0).sum()),
                            "roi_is_exploratory":True})
    return pd.DataFrame(metrics)


def forecast_time_status(games: pd.DataFrame):
    now=datetime.now(timezone.utc)
    naive=pd.to_datetime(games["game_date"].str.replace("_","-",regex=False)+" "+games["game_time"],errors="raise")
    tz=naive.dt.tz_localize("America/New_York",ambiguous="NaT",nonexistent="NaT").dt.tz_convert("UTC")
    if tz.isna().any():
        raise ValueError("Invalid target fixture timezone")
    return np.where(tz>pd.Timestamp(now),"future_fixture_unverified_latest_price","historical_reconstruction")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only",choices=["both","backtest","predict"],default="both")
    parser.add_argument("--date",help="YYYY_MM_DD, defaults to latest merged game date")
    parser.add_argument("--skip-xg",action="store_true",help="Skip optional P5 local Parquet xG reconstruction")
    parser.add_argument("--ev-threshold",type=float,default=.03)
    args=parser.parse_args()
    if not (0. <= args.ev_threshold <= .5):
        parser.error("EV threshold outside 0..0.5")
    history,current=v1.load_history()
    hist,cur,cols,coverage,xg_status=assemble(history,current,args.skip_xg)
    write_csv(coverage,OUT/"reports"/"feature_coverage.csv")
    write_csv(xg_status,OUT/"reports"/"xg_source_status.csv")
    scheduled = v1.read_csv(MASTER)
    schedule_regular = scheduled["game_id"].str.slice(4, 6).eq("02")
    hist2025 = hist.loc[hist["data_period"].eq("2025_2026")]
    audit = pd.DataFrame([
        {"item": "2025_26_regular_scheduled", "count": int(schedule_regular.sum()),
         "status": "historical_aggregated_schedule_no_observation_timestamp"},
        {"item": "2025_26_completed_matched_labels", "count": len(hist2025),
         "status": "validated_postgame_scores_not_same_day_features"},
        {"item": "2025_26_regular_schedule_not_in_training_labels",
         "count": int(schedule_regular.sum()) - len(hist2025),
         "status": "historical_form_may_under_count_missing_result_games"},
        {"item": "2025_26_training_odds_without_pregame_capture_timestamp",
         "count": int((~hist2025["pregame_prices_verified"]).sum()),
         "status": "do_not_use_for_verified_roi"},
        {"item": "2026_27_training_odds_pulled_before_scheduled_start",
         "count": int(hist.loc[hist["data_period"].eq("2026_2027"),"pregame_prices_verified"].sum()),
         "status": "odds_pull_only_other_feature_asof_provenance_not_authenticated"},
        {"item": "historical_pregame_goalie_starters_authenticated", "count": 0,
         "status": "unavailable_no_actual_starter_leakage_allowed"},
    ])
    write_csv(audit, OUT/"reports"/"provenance_audit.csv")
    print(f"HISTORICAL {len(hist)} games; CURRENT {len(cur)} rows; FEATURES {len(cols)}",flush=True)
    scores=walkforward(hist,cols)
    if args.only in ("both","backtest"):
        write_csv(scores,OUT/"reports"/"walkforward_predictions.csv")
        score_table=summary_table(scores,args.ev_threshold)
        write_csv(score_table,OUT/"reports"/"walkforward_summary.csv")
        print(score_table[["season","model","games","brier","log_loss","eligible_roi_bets","verified_timestamp_roi"]].to_string(index=False))
    if args.only in ("both","predict"):
        day=pd.Timestamp(args.date.replace("_","-")) if args.date else cur["game_date_dt"].max()
        games=cur.loc[cur["game_date_dt"].eq(day)].copy()
        if games.empty:
            raise ValueError(f"No merged fixtures on {day.date()}")
        model=fit_predict_prior(hist,day,cols)
        if model is None:
            raise ValueError("Insufficient past games for target date")
        past=scores.loc[v1.date_col(scores["game_date"])<day].copy()
        pred=row_predictions(games,model,past)
        pred["generated_at_utc"]=datetime.now(timezone.utc).isoformat()
        pred["forecast_type"]=forecast_time_status(games)
        path=OUT/"predictions"/f"{day.strftime('%Y_%m_%d')}_NHL_my_probability_v3.csv"
        write_csv(pred,path)
        print(pred[["game_date","away_team","home_team","my_probability_home",
                    "my_probability_away","market_blend_weight","adjustment_status"]].to_string(index=False))
    print("SAFETY: No live bet-selection integration. 2025-26 historical odds timestamps unverified.")
    print("SAFETY: Historical starter identities not verifiably pregame. No starter-derived training feature.")
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,ImportError,KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
