#!/usr/bin/env python3
"""Publish V8's actual xG probabilities; compare against D-Ratings and SDV.

Reads existing, already-computed V8 artifacts. Does not train, adjust,
select on holdouts, or touch live bet selection. The market appears only
in reports as a reference and is not used to produce probabilities.

Usage: python publish_my_probability_v8_independent.py [--date YYYY_MM_DD]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "v8_independent_direct_publisher_20261120"
REQUIRED_REPLAY = {"game_id", "game_date", "data_period", "target", "v8_xg_home",
                   "v5_base_home", "vig_free_home_prob", "drat_home_win_prob", "sdv_home_win_prob"}
REQUIRED_FORECAST = {"game_id", "game_date", "home_team", "away_team", "v8_xg_experimental_home",
                     "v5_baseline_home"}
BENCHMARKS = [
    ("V8 independent xG", "v8_xg_home"),
    ("V5 historical baseline", "v5_base_home"),
    ("D-Ratings", "drat_home_win_prob"),
    ("SportsDataverse", "sdv_home_win_prob"),
    ("No-vig market", "vig_free_home_prob"),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f'WROTE {path} | rows={len(df)}')


def atomic_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)
    print(f'WROTE {path}')


def load_replay(path):
    require(path.is_file(), f'V8 historical replay missing: {path}')
    df = pd.read_csv(path, dtype={'game_id': 'string'}, low_memory=False)
    require(REQUIRED_REPLAY.issubset(df.columns),
            f'V8 replay missing columns: {sorted(REQUIRED_REPLAY - set(df.columns))}')
    require(not df.game_id.isna().any(), 'Replay contains null game IDs')
    require(df.game_id.str.fullmatch(r'\d{10}').all(), 'Replay contains noncanonical game IDs')
    require(not df.game_id.duplicated().any(), 'Replay contains duplicate game IDs')
    df['day'] = pd.to_datetime(df.game_date.astype(str).str.replace('_', '-', regex=False), errors='coerce')
    require(df.day.notna().all(), 'Replay contains invalid calendar dates')
    require(df.data_period.isin(['2025_2026', '2026_2027']).all(),
            'Replay contains an unexpected season')
    require(set(df.data_period.unique()) == {'2025_2026', '2026_2027'},
            'Replay does not cover both expected seasons')
    for col in ['target'] + [c for _, c in BENCHMARKS]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    require(df.target.isin([0.0, 1.0]).all(), 'Replay has missing/non-binary results')
    require(df.v8_xg_home.notna().all(), 'Missing independent V8 probabilities')
    require(df.v8_xg_home.between(0, 1, inclusive='neither').all(),
            'Independent V8 has invalid probabilities')
    return df


def paired_interval(df, model, reference, draws=2000, seed=20261120):
    rows = df[['day', 'target', model, reference]].dropna().copy()
    require(len(rows) >= 25 and rows.day.nunique() >= 15, 'Insufficient paired dates')
    rows['excess'] = (rows[model] - rows.target)**2 - (rows[reference] - rows.target)**2
    groups = [x.excess.to_numpy(float) for _, x in rows.groupby('day', sort=True)]
    # Preserve within-day pairing and the number of games when resampling dates.
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for j in range(draws):
        idx = rng.integers(0, len(groups), size=len(groups))
        out[j] = sums[idx].sum() / counts[idx].sum()
    return (float(rows.excess.mean()), float(np.quantile(out, .025)),
            float(np.quantile(out, .975)), len(rows), rows.day.nunique())


def compare(df):
    result = []
    for period, period_rows in df.groupby('data_period', sort=True):
        # One complete-case sample per season for fair four-way comparison.
        use = period_rows.dropna(subset=[c for _, c in BENCHMARKS]).copy()
        require(len(use) >= 200, f'Too few complete observations in {period}: {len(use)}')
        for name, col in BENCHMARKS:
            q = use[col]
            require(q.between(0, 1, inclusive='neither').all(),
                    f'Invalid {name} probability for {period}')
            y = use.target
            loss = float(((q - y)**2).mean())
            p = q.clip(1e-12, 1-1e-12)
            log_loss = float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
            result.append({'season':period, 'model':name, 'games':len(use),
                           'brier':loss, 'log_loss':log_loss,
                           'paired_sample':'all_five_models_complete'})
    return pd.DataFrame(result)


def comparisons(df):
    result = []
    for period, period_rows in df.groupby('data_period', sort=True):
        use = period_rows.dropna(subset=[c for _, c in BENCHMARKS]).copy()
        for name, col in BENCHMARKS[1:]:
            mean, lo, hi, n, ndays = paired_interval(use, 'v8_xg_home', col)
            result.append({'season':period, 'V8_vs':name, 'games':n, 'dates':ndays,
                           'V8_brier_minus_comparator':mean, 'paired_date_95ci_low':lo,
                           'paired_date_95ci_high':hi,
                           'interval_excludes_zero':bool(lo > 0 or hi < 0),
                           'caution':'Repeatedly examined historical seasons; exploratory, not fresh holdouts'})
    return pd.DataFrame(result)


def get_forecast(source_dir, date_arg):
    forecast_dir = source_dir / 'predictions'
    if date_arg:
        require(bool(re.fullmatch(r'\d{4}_\d{2}_\d{2}', date_arg)),
                'Use --date YYYY_MM_DD')
        selected = forecast_dir / f'{date_arg}_NHL_my_probability_independent_v8.csv'
    else:
        possibilities = sorted(forecast_dir.glob('*_NHL_my_probability_independent_v8.csv'))
        require(bool(possibilities), f'No saved V8 forecasts in {forecast_dir}')
        selected = possibilities[-1]
    require(selected.is_file(), f'V8 forecast missing: {selected}')
    pred = pd.read_csv(selected, dtype={'game_id': 'string'}, low_memory=False)
    require(REQUIRED_FORECAST.issubset(pred.columns),
            f'V8 forecast missing columns: {sorted(REQUIRED_FORECAST - set(pred.columns))}')
    require(not pred.empty, 'V8 forecast is empty')
    require(not pred.game_id.isna().any() and not pred.game_id.duplicated().any(),
            'V8 forecast has missing/duplicate game IDs')
    require(pred.game_id.str.fullmatch(r'\d{10}').all(), 'Forecast has invalid game IDs')
    d = pred.game_date.astype(str)
    require(d.nunique() == 1, 'V8 forecast contains multiple game dates')
    require(bool(re.fullmatch(r'\d{4}_\d{2}_\d{2}', d.iloc[0])), 'Invalid forecast game date')
    p = pd.to_numeric(pred.v8_xg_experimental_home, errors='coerce')
    require(p.notna().all() and p.between(0, 1, inclusive='neither').all(),
            'V8 forecast xG probabilities missing or invalid')
    out = pred[['game_id','game_date','home_team','away_team']].copy()
    out['my_probability_home'] = p
    out['my_probability_away'] = 1 - p
    out['model'] = 'independent_v8_xg'
    out['model_inputs_include_market_odds'] = False
    out['market_fallback_used'] = False
    out['research_only'] = True
    out['forecast_status'] = 'historical_saved_forecast_not_new_live_prediction'
    if 'v5_baseline_home' in pred:
        out['v5_baseline_home_reference'] = pd.to_numeric(pred.v5_baseline_home, errors='coerce')
    if 'vig_free_home_prob' in pred:
        out['vig_free_market_home_reference'] = pd.to_numeric(pred.vig_free_home_prob, errors='coerce')
    return selected, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--date', default=None, help='Existing V8 forecast date YYYY_MM_DD')
    ap.add_argument('--nhl-root', type=Path, default=Path(__file__).resolve().parents[2],
                    help='Test override: root directory containing my_probability')
    args = ap.parse_args()
    src = args.nhl_root / 'my_probability' / 'independent_v8_xg'
    dest = args.nhl_root / 'my_probability' / 'independent_v13_v8_direct'
    replay_path = src / 'reports' / '2025_2027_chronological_replay.csv'
    replay = load_replay(replay_path)
    summary = compare(replay)
    pairs = comparisons(replay)
    forecast_path, forecast = get_forecast(src, args.date)
    today = forecast.game_date.iloc[0]
    atomic_csv(summary, dest / 'reports' / 'direct_model_comparison.csv')
    atomic_csv(pairs, dest / 'reports' / 'paired_date_comparisons.csv')
    atomic_csv(forecast, dest / 'predictions' / f'{today}_NHL_my_probability_independent.csv')
    atomic_json({'version': VERSION, 'published_at_utc':datetime.now(timezone.utc).isoformat(),
                 'inputs_sha256':{str(replay_path):sha256(replay_path),
                                  str(forecast_path):sha256(forecast_path)},
                 'probability_source':'V8 xG experimental, NOT V5 fallback',
                 'odds_as_features':False, 'historical_tests_repeatedly_inspected':True,
                 '2025_odds_timestamp_verified':False,
                 'production_modified':False, 'fitting_or_selection_performed':False},
                dest / 'reports' / 'run_manifest.json')
    print('Brier comparison (lower is better; same games within each season):')
    print(summary[['season','model','games','brier']].to_string(index=False))
    print('Research-only V8 xG probabilities published DIRECTLY, not V5 fallback.')
    print('Holdout periods were repeatedly examined; no predictive-edge claim.')
    print('No betting or production pipeline changes.')

if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as e:
        sys.exit(f'ERROR: {e}')
