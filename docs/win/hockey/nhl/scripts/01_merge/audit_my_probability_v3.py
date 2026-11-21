#!/usr/bin/env python3
"""Read-only diagnostic of my_probability v3. No retraining or production changes.

Copy via PowerShell into:
 G:\\hockey\\docs\\win\\hockey\\nhl\\scripts\\01_merge\\audit_my_probability_v3.py

Inputs: my_probability/v3/reports/{walkforward_predictions,feature_coverage,
       xg_source_status,provenance_audit}.csv
Outputs: my_probability/v4/reports/* (only new diagnostic artifacts).

The expected-value ROI is EXPLORATORY and restricted to rows whose odds capture
preceded the scheduled game start in the source. A timestamp from the source is
not independent proof of the historical quote or its availability.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "my_probability" / "v3" / "reports"
OUTPUT = ROOT / "my_probability" / "v4" / "reports"
EPS = 1e-8
MODEL_COLUMNS = {"market": "vig_free_home_prob", "raw_v3": "raw_v3_probability_home",
                 "guarded_v3": "my_probability_home", "D_Ratings": "drat_home_win_prob",
                 "SportsDataverse": "sdv_home_win_prob"}
NEEDED = {"game_id", "game_date", "data_period", "target", "pregame_prices_verified",
          "home_decimal", "away_decimal", "market_blend_weight", "adjustment_status",
          "form_complete_both_teams", "form_cutoff_exclusive", "max_source_result_date", "my_probability_away",
          *MODEL_COLUMNS.values()}


def load(name: str) -> pd.DataFrame:
    path = INPUT / name
    if not path.is_file():
        raise FileNotFoundError(f"Required v3 artifact not found: {path}")
    return pd.read_csv(path, dtype={"game_id": "string"})


def save(frame: pd.DataFrame, name: str) -> None:
    out = OUTPUT / name
    frame.to_csv(out, index=False)
    print(f"WROTE {out} | rows={len(frame)}")


def loss(y: np.ndarray, pred: np.ndarray, metric: str) -> np.ndarray:
    p = np.clip(pred.astype(float), EPS, 1 - EPS)
    if metric == "brier":
        return (p - y)**2
    if metric == "log_loss":
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    raise ValueError(metric)


def stats(group: pd.DataFrame, name: str, *, threshold: float) -> dict:
    y = group["target"].to_numpy(float)
    p = group[MODEL_COLUMNS[name]].to_numpy(float)
    eligible = group.loc[group["pregame_prices_verified"]].copy()
    quote_n = len(eligible)
    chosen_n, units, roi = 0, np.nan, np.nan
    if quote_n:
        z = eligible[MODEL_COLUMNS[name]].to_numpy(float)
        home_dec = eligible["home_decimal"].to_numpy(float)
        away_dec = eligible["away_decimal"].to_numpy(float)
        home_ev = z * home_dec - 1.0
        away_ev = (1-z) * away_dec - 1.0
        bet = np.maximum(home_ev, away_ev) > threshold
        select_home = home_ev >= away_ev
        selected_dec = np.where(select_home, home_dec, away_dec)
        win = np.where(select_home, eligible["target"].to_numpy(float) == 1,
                       eligible["target"].to_numpy(float) == 0)
        chosen_n = int(np.sum(bet))
        units = float(np.sum(np.where(win, selected_dec-1, -1)[bet])) if chosen_n else np.nan
        roi = float(units/chosen_n) if chosen_n else np.nan
    return {"model": name, "games": len(group),
            "brier": float(np.mean(loss(y, p, "brier"))),
            "log_loss": float(np.mean(loss(y,p,"log_loss"))),
            "accuracy": float(np.mean((p>=0.5)==(y==1))),
            "mean_probability_home": float(np.mean(p)),
            "mean_outcome_home": float(np.mean(y)),
            "odds_timestamp_pre_start_rows": quote_n,
            "ev_threshold": threshold, "eligible_bets": chosen_n,
            "one_unit_profit": units, "roi": roi,
            "roi_provisional": True}


def model_table(group: pd.DataFrame, *, period: str, bucket: str, threshold: float) -> list[dict]:
    return [{"period": period, "bucket": bucket, **stats(group, label, threshold=threshold)}
            for label in MODEL_COLUMNS]


def paired_bootstrap(group: pd.DataFrame, *, metric: str, draws: int, seed: int) -> dict:
    """Resample whole game dates; positive difference means raw V3 underperformed market."""
    y = group["target"].to_numpy(float)
    diff = (loss(y, group[MODEL_COLUMNS["raw_v3"]].to_numpy(float),metric)
            - loss(y,group[MODEL_COLUMNS["market"]].to_numpy(float),metric))
    labels, index = np.unique(group["game_date"].to_numpy(str), return_inverse=True)
    sums = np.bincount(index, weights=diff)
    sizes = np.bincount(index)
    if len(labels)<2:
        return {"metric":metric,"days":len(labels),"mean_raw_minus_market":float(diff.mean()),
                "ci_low":np.nan,"ci_high":np.nan,"bootstrap_draws":0}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for i in range(draws):
        ix=rng.integers(0,len(labels),size=len(labels))
        estimates[i] = sums[ix].sum()/sizes[ix].sum()
    low,high = np.quantile(estimates, [.025,.975])
    return {"metric":metric,"days":len(labels),"mean_raw_minus_market":float(diff.mean()),
            "ci_low":float(low),"ci_high":float(high),"bootstrap_draws":draws}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ev-threshold",type=float,default=.03)
    parser.add_argument("--bootstrap-draws",type=int,default=1500)
    opts=parser.parse_args()
    if not(0 <= opts.ev_threshold < 1) or not(100<=opts.bootstrap_draws<=10000):
        parser.error("EV threshold must be 0..1; bootstrap draws must be 100..10000")
    data=load("walkforward_predictions.csv")
    coverage=load("feature_coverage.csv")
    xg=load("xg_source_status.csv")
    provenance=load("provenance_audit.csv")
    missing=sorted(NEEDED-set(data.columns))
    if missing:
        raise ValueError(f"Missing v3 prediction columns: {missing}")
    if data.empty or data["game_id"].duplicated().any():
        raise ValueError("Empty or duplicated v3 predictions")
    if not data["target"].isin([0,1]).all():
        raise ValueError("Labels must be binary")
    if not set(data["data_period"].unique()) <= {"2025_2026","2026_2027"}:
        raise ValueError("Unknown historical period")
    for col in MODEL_COLUMNS.values():
        vals=pd.to_numeric(data[col],errors="coerce")
        if vals.isna().any() or not vals.between(0,1).all():
            raise ValueError(f"Missing or out-of-range probabilities: {col}")
        data[col] = vals
    for col in ("home_decimal","away_decimal"):
        data[col]=pd.to_numeric(data[col],errors="coerce")
        if data[col].isna().any() or (data[col]<=1).any():
            raise ValueError(f"Missing or invalid decimal odds: {col}")
    if not (data["my_probability_home"]-(1-data["my_probability_away"])).abs().lt(1e-6).all():
        raise ValueError("V3 probabilities do not sum to 1")
    data["pregame_prices_verified"] = data["pregame_prices_verified"].astype("string").str.lower().eq("true")
    # Validate historical results cannot be used on or after the target prediction date.
    train_date = pd.to_datetime(data["max_source_result_date"],errors="coerce")
    pred_date = pd.to_datetime(data["game_date"].str.replace("_","-",regex=False),errors="coerce")
    form_date = pd.to_datetime(data["form_cutoff_exclusive"],errors="coerce")
    if train_date.isna().any() or pred_date.isna().any() or form_date.isna().any():
        raise ValueError("Missing source or prediction date; cannot audit leakage")
    if (train_date >= pred_date).any():
        raise ValueError("Date leakage: max_source_result_date >= game date")
    if not (form_date == pred_date).all():
        raise ValueError("Form cutoff differs from prediction date")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    checks=pd.DataFrame([
        {"check":"unique_game_ids","status":"PASS","count":len(data)},
        {"check":"all_prior_results_strictly_before_target","status":"PASS","count":len(data)},
        {"check":"form_cutoff_is_target_exclusive","status":"PASS","count":len(data)},
        {"check":"2025_2026_odds_capture_times_unverified","status":"WARNING","count":int((data.data_period.eq('2025_2026')&~data.pregame_prices_verified).sum())},
        {"check":"2026_2027_odds_timestamp_pre_start_only","status":"LIMITATION","count":int((data.data_period.eq('2026_2027')&data.pregame_prices_verified).sum())},
        {"check":"verified_historical_goalie_starter_features","status":"UNAVAILABLE","count":0},
    ])
    save(checks,"validation_checks.csv")
    mainrows=[]
    all_groups=[("ALL",data),*list(data.groupby("data_period",sort=True))]
    for period,g in all_groups:
        mainrows.extend(model_table(g,period=period,bucket="ALL",threshold=opts.ev_threshold))
    mainframe=pd.DataFrame(mainrows)
    save(mainframe,"season_summary.csv")
    monthly=[]
    for (period,month),g in data.groupby(["data_period",data["game_date"].str.slice(0,7)],sort=True):
        monthly.extend(model_table(g,period=period,bucket=month,threshold=opts.ev_threshold))
    save(pd.DataFrame(monthly),"monthly_summary.csv")
    data["favourite_status"] = np.where(data.vig_free_home_prob>=.5,"home_favourite","home_underdog")
    data["magnitude"] = np.where((data.raw_v3_probability_home-data.vig_free_home_prob).abs()>=.07,"abs_adjustment_ge_7pt","abs_adjustment_lt_7pt")
    data["raw_adjustment_direction"] = np.where(
        (data.raw_v3_probability_home-.5).abs() < (data.vig_free_home_prob-.5).abs(),
        "toward_50pct","away_from_50pct")
    data["market_probability_bin"] = pd.cut(data.vig_free_home_prob,
        [0,.35,.45,.50,.55,.65,1], include_lowest=True).astype(str)
    segments=[]
    for period,g in data.groupby("data_period",sort=True):
        for dimension in ["favourite_status","magnitude","raw_adjustment_direction",
                          "form_complete_both_teams"]:
            for value,s in g.groupby(dimension,dropna=False,sort=True):
                segments.extend([{"dimension":dimension,"level":str(value),**row}
                                 for row in model_table(s,period=period,bucket=str(value),threshold=opts.ev_threshold)])
    save(pd.DataFrame(segments),"segments.csv")
    calrows=[]
    for period,g in data.groupby("data_period",sort=True):
        for b,s in g.groupby("market_probability_bin",sort=True):
            calrows.append({"period":period,"market_probability_bin":b,"games":len(s),
                            "realized_home_win_rate":s.target.mean(),
                            "market_avg":s.vig_free_home_prob.mean(),
                            "raw_v3_avg":s.raw_v3_probability_home.mean(),
                            "guarded_v3_avg":s.my_probability_home.mean(),
                            "raw_adjustment_mean":(s.raw_v3_probability_home-s.vig_free_home_prob).mean()})
    save(pd.DataFrame(calrows),"calibration_bins.csv")
    ci=[]
    for period,g in all_groups:
        for m in ["brier","log_loss"]:
            ci.append({"period":period,"comparison":"raw_v3 minus vig_free_market",
                       **paired_bootstrap(g,metric=m,draws=opts.bootstrap_draws,seed=20261119)})
    save(pd.DataFrame(ci),"paired_date_bootstrap.csv")
    save(coverage,"copied_feature_coverage.csv")
    save(xg,"copied_xg_source_status.csv")
    save(provenance,"copied_provenance_audit.csv")
    report=[]
    report.append("my_probability v3 read-only diagnostic (not a new model)")
    report.append("Positive raw-minus-market loss means v3 is WORSE. Lower loss is better.")
    report.append("A 95% bootstrap interval including zero does not show a reliable difference.")
    report.append("2025-26 odds timestamps unknown: ROI for that season is not validated.")
    report.append("Current-season odds timestamps precede scheduled start according to source, not independently authenticated.")
    report.append("No v3 probabilities or pipeline configuration are modified.")
    report.append("")
    for period,g in all_groups:
        rec=mainframe.loc[mainframe.period.eq(period)].set_index("model")
        effect=rec.loc["raw_v3","brier"]-rec.loc["market","brier"]
        raw_adjustment=(g.raw_v3_probability_home-g.vig_free_home_prob)
        report.extend([f"{period}: {len(g)} games; raw-market Brier difference={effect:+.6f}",
                       f"  mean raw home-prob shift={raw_adjustment.mean():+.4f}; mean abs shift={raw_adjustment.abs().mean():.4f}",
                       f"  market guard fallback count={(g.market_blend_weight==0).sum()}/{len(g)}",
                       f"  form complete count={(g.form_complete_both_teams==1).sum()}/{len(g)}"])
    report.append("Recommendation: retain separate model; require prospective timestamped data, new pre-registered season splits, and evidence over market before deployment.")
    text="\n".join(report)+"\n"
    dest=OUTPUT/"diagnosis.txt"
    dest.write_text(text,encoding="utf-8")
    print(text)
    print(f"WROTE {dest}")
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise SystemExit(f"DIAGNOSTIC ERROR: {exc}") from exc
