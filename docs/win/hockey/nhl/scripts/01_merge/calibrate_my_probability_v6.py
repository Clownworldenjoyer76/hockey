#!/usr/bin/env python3
"""V6: independently calibrate V5 using 2024-25 only; test untouched 2025-27.

Run alongside predict_my_probability_v5_extended.py. Never changes V5 or the
production pipeline. Uses official game results to build 2024-25 out-of-training
predictions and selects a single probability-shrinkage coefficient using ONLY
those labels. All 2025-26 and 2026-27 observations are read after selection,
for evaluation only. Does not train on odds, provider predictions, or future
outcomes. Market prices are a research benchmark, not part of the model.

2024-25 was ALSO used to select V5's base features/regularization. Therefore
calibration selection on that season is exploratory and may itself overfit.
Only the 2025-27 chronological tests are out-of-selection evidence.

Outputs: NHL/my_probability/independent_v6_calibration/reports/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

import predict_my_probability_v5_extended as v5
from predict_my_probability_independent import RidgeLogit, brier, log_loss

OUT = v5.NHL / 'my_probability' / 'independent_v6_calibration'
REPORTS = OUT / 'reports'
V5_REPORTS = v5.OUT / 'reports'
ALPHAS = (1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30)
EPS = 1e-7
MODEL_ID = 'independent_v6_2024_only_shrinkage_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f'WROTE {path} | rows={len(frame)}', flush=True)


def shrink(raw, alpha, home_baseline):
    """Odds-free, strictly monotonic temperature shrink toward prior home rate."""
    raw = np.asarray(raw, dtype=float)
    require(np.isfinite(raw).all() and ((raw > 0) & (raw < 1)).all(), 'Invalid raw independent probabilities')
    require(0. < alpha <= 1., 'Alpha outside the predeclared conservative range')
    require(0. < home_baseline < 1., 'Invalid prior-season home-win baseline')
    base_logit = float(logit(home_baseline))
    return np.clip(expit(base_logit + alpha * (logit(raw) - base_logit)), EPS, 1.-EPS)


def baseline_model_selection():
    path = V5_REPORTS / 'model_selection_2024_validation.csv'
    require(path.is_file(), f'Missing locked V5 selection: {path}')
    rows = pd.read_csv(path)
    for field in ('model', 'penalty', 'brier', 'log_loss'):
        require(field in rows.columns, f'V5 selection missing column {field}')
    require(len(rows) == len(v5.FEATURE_SETS) * len(v5.PENALTIES), 'V5 selection has unexpected candidate count')
    rows = rows.sort_values(['brier', 'log_loss', 'model', 'penalty'])
    best = rows.iloc[0]
    name, penalty = str(best['model']), float(best['penalty'])
    require(name in v5.FEATURE_SETS and penalty in v5.PENALTIES, 'V5 winner unknown or altered')
    require(rows['brier'].notna().all(), 'Invalid V5 validation scores')
    return name, penalty, float(best['brier'])


def build_2024_calibration(name, penalty, expected_brier):
    # Offline cache only: no NHL requests and no alteration of older V5 data.
    season_frames=[]
    for year in (2022, 2023, 2024):
        path=v5.RAW / f'season_{year}_regular_games.csv'
        require(path.is_file(), f'Missing existing official historical cache: {path}')
        season_frames.append(v5.load_old_season(None, year, offline=True))
    old=pd.concat(season_frames, ignore_index=True)
    history, current, local = v5.historical_and_current(old)
    train=history.loc[history.data_period.isin(['2022_2023', '2023_2024'])].copy()
    valid=history.loc[history.data_period.eq('2024_2025')].copy()
    require(len(train) >= 2200 and len(valid)>=1100, 'Not enough earlier completed historical games')
    require(train.game_date_dt.max()<valid.game_date_dt.min(), 'Base-training outcomes overlap calibration dates')
    require(train.target.isin([0.,1.]).all() and valid.target.isin([0.,1.]).all(), 'Nonbinary official outcomes')
    base=float(train.target.mean())
    require(.40 <= base <= .65, f'Unusual past-only home-win baseline {base}')
    raw=RidgeLogit(v5.FEATURE_SETS[name], penalty).fit(train).predict(valid)
    reproduced=float(brier(valid.target,raw))
    require(abs(reproduced-expected_brier)<2e-5,
            f'V5 2024 selection cannot be reproduced: selection Brier={expected_brier:.7f}, rebuilt={reproduced:.7f}')
    candidate_rows=[]
    for alpha in ALPHAS:
        p=shrink(raw, alpha, base)
        candidate_rows.append({'alpha':alpha,'2024_validation_games':len(valid),
                               'brier':float(brier(valid.target,p)),
                               'log_loss':float(log_loss(valid.target,p)),
                               'home_baseline_2022_2024':base,
                               'model_features':name,'model_penalty':penalty,
                               'odds_used_to_select':False})
    trials=pd.DataFrame(candidate_rows).sort_values(['brier','log_loss','alpha'], ascending=[True,True,False])
    alpha=float(trials.iloc[0].alpha)
    preview=valid[['game_id','game_date','data_period','target']].copy()
    preview['v5_home_prob']=raw
    preview['v6_home_prob']=shrink(raw,alpha,base)
    preview['calibration_fit_label']='2024_2025_selection_only_not_held_out'
    return trials, preview, alpha, base


def load_holdout():
    path=V5_REPORTS/'chronological_2025_2026_evaluation.csv'
    require(path.is_file(), f'Missing locked V5 chronological predictions: {path}')
    frame=pd.read_csv(path,dtype={'game_id':'string'})
    required=('game_id','game_date','data_period','target','my_probability_home',
              'vig_free_home_prob','last_training_game_date')
    for f in required:
        require(f in frame, f'V5 chronological results missing {f}')
    require(len(frame)>=1500, 'Insufficient V5 holdout rows')
    require(frame.game_id.notna().all() and not frame.game_id.duplicated().any(), 'Duplicate/missing V5 game IDs')
    require(set(frame.data_period.unique())=={'2025_2026','2026_2027'}, 'Unexpected V5 holdout seasons')
    require(frame.target.isin([0.,1.]).all(), 'V5 holdout has incomplete results')
    day=pd.to_datetime(frame.game_date.str.replace('_','-',regex=False),errors='raise')
    prior=pd.to_datetime(frame.last_training_game_date,errors='raise')
    require((prior<day).all(), 'V5 holdout training dates leak target game day')
    raw=frame.my_probability_home.to_numpy(float)
    require(np.isfinite(raw).all() and ((raw>0)&(raw<1)).all(), 'V5 holdout has invalid probabilities')
    return frame


def bootstrap_by_day(frame, left_col, right_col, repeats=1500, seed=20261120):
    paired=frame.loc[frame[[left_col,right_col]].notna().all(axis=1)].reset_index(drop=True)
    if len(paired)==0:
        return float('nan'),float('nan'),float('nan'),0
    y=paired.target.to_numpy(float)
    diff=(paired[left_col].to_numpy(float)-y)**2-(paired[right_col].to_numpy(float)-y)**2
    date_key=paired.game_date.astype(str).to_numpy()
    unique_dates=np.unique(date_key)
    blocks=[np.flatnonzero(date_key==d) for d in unique_dates]
    rng=np.random.default_rng(seed)
    samples=np.empty(repeats,dtype=float)
    for i in range(repeats):
        indices=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))])
        samples[i]=float(diff[indices].mean())
    low,high=np.quantile(samples,[.025,.975])
    return float(diff.mean()),float(low),float(high),len(paired)


def evaluate(frame, alpha, base):
    frame=frame.copy()
    frame['independent_v5_home_prob']=frame['my_probability_home'].to_numpy(float)
    frame['independent_v6_home_prob']=shrink(frame.independent_v5_home_prob,alpha,base)
    frame['independent_v6_away_prob']=1.0-frame['independent_v6_home_prob']
    frame['calibration_alpha_2024_only']=alpha
    frame['calibration_home_baseline_past_only']=base
    frame['odds_used_in_calibration_or_model']=False
    metrics=[]
    models=[('independent_v6','independent_v6_home_prob'),
            ('independent_v5','independent_v5_home_prob'),
            ('vig_free_market','vig_free_home_prob')]
    for season,g in frame.groupby('data_period',sort=True):
        for label,col in models:
            valid=g.loc[g[[col,'target']].notna().all(axis=1)]
            if valid.empty:
                continue
            metrics.append({'season':season,'model':label,'games':len(valid),
                            'brier':float(brier(valid.target,valid[col])),
                            'log_loss':float(log_loss(valid.target,valid[col])),
                            'calibration_selected_on':'2024_2025_only',
                            'research_market_timestamp_verified_independently':False})
        for comparator,field in [('v5','independent_v5_home_prob'),('market','vig_free_home_prob')]:
            d,low,high,n=bootstrap_by_day(g,'independent_v6_home_prob',field)
            metrics.append({'season':season,'model':'v6_minus_'+comparator,
                            'games':n,'brier':d,'log_loss':float('nan'),
                            'paired_date_95ci_low':low,'paired_date_95ci_high':high,
                            'calibration_selected_on':'2024_2025_only',
                            'research_market_timestamp_verified_independently':False})
    return frame,pd.DataFrame(metrics)


def self_test():
    test=np.array([.05,.20,.50,.75,.95])
    assert np.allclose(shrink(test,1.,.54),test,atol=1e-12)
    assert np.all(np.diff(shrink(test,.5,.54))>0)
    assert np.all(np.abs(logit(shrink(test,.5,.54))-logit(.54))<=np.abs(logit(test)-logit(.54))+1e-12)
    assert np.allclose(shrink(test,.5,.54)+1.-shrink(test,.5,.54),1.)
    demo=pd.DataFrame({'game_date':['2025_11_01','2025_11_01','2025_11_02'],
                       'target':[1.,0.,1.],'left':[.7,.3,.6],'right':[.5,.5,.5]})
    d,l,h,n=bootstrap_by_day(demo,'left','right',repeats=100)
    assert n==3 and np.isfinite([d,l,h]).all()
    print('SELF-TEST PASS: alpha=1 identity, monotonic shrinkage, odds-free center, paired-date CI')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true',help='Run deterministic tests without reading local data')
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    name,penalty,expected_brier=baseline_model_selection()
    trials,validation,alpha,base=build_2024_calibration(name,penalty,expected_brier)
    # Calibration is NOW LOCKED; only after this point read 2025-27 outcomes.
    lock={'model':MODEL_ID,'base_model':name,'base_penalty':penalty,'alpha':alpha,
          'home_baseline_2022_2024':base,'2024_validation_brier_raw':float(trials.loc[trials.alpha.eq(1.),'brier'].iloc[0]),
          '2024_validation_brier_calibrated':float(trials.iloc[0].brier),
          'calibration_selection_season':'2024_2025',
          'base_model_was_also_selected_on_2024':True,
          '2025_2026_and_2026_2027_used_for_selection':False,
          'model_uses_market_odds':False,'model_uses_provider_predictions':False,
          'market_timestamp_independent_authentication':False,
          'date_asof_rule':'historical results strictly before target date',
          'source':'existing V5 official history and chronological predictions',
          'created_utc':datetime.now(timezone.utc).isoformat()}
    REPORTS.mkdir(parents=True,exist_ok=True)
    with (REPORTS/'calibration_lock.json').open('w',encoding='utf-8') as f:
        json.dump(lock,f,indent=2,sort_keys=True)
    print(f'LOCKED 2024 CALIBRATION alpha={alpha:.2f} baseline={base:.4f} base={name} penalty={penalty}',flush=True)
    holdout=load_holdout()
    scored,summary=evaluate(holdout,alpha,base)
    save(trials,REPORTS/'2024_calibration_candidates.csv')
    save(validation,REPORTS/'2024_validation_predictions.csv')
    save(pd.DataFrame([lock]),REPORTS/'calibration_selection.csv')
    save(scored,REPORTS/'2025_2027_calibrated_holdout_predictions.csv')
    save(summary,REPORTS/'2025_2027_calibrated_summary.csv')
    print(summary[['season','model','games','brier','paired_date_95ci_low','paired_date_95ci_high']].to_string(index=False))
    print('Calibration used 2024-25 only. 2025-27 reports are evaluation, not tuning. No market fallback or betting integration.')


if __name__=='__main__':
    try:
        main()
    except (OSError,ValueError,ImportError,KeyError,TypeError) as exc:
        sys.exit(f'ERROR: {exc}')
