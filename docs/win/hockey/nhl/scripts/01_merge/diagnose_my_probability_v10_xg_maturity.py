#!/usr/bin/env python3
"""V10: identify whether V9's xG compression is a season-maturity artifact.

Uses only V8's locked strict-prior xG cache, V5's existing official results,
and V9's overlap-safe historical loader. No network, training selection,
production write, betting action, or look-ahead features.

Compares 2026-27's partial season with the *same early progress* in previous
seasons, both by number of dates and number of games; also standardizes the
2022-24 reference to 2026-27's distribution of prior team-game counts.

Usage: python diagnose_my_probability_v10_xg_maturity.py [--self-test]
Writes only NHL/my_probability/independent_v10_xg_maturity/reports/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import diagnose_my_probability_v9_xg_drift as v9
from predict_my_probability_independent import RidgeLogit

ROOT = v9.v8.ROOT
OUT = ROOT / 'my_probability' / 'independent_v10_xg_maturity' / 'reports'
XG = v9.XG
YEARS = v9.YEARS
REF_SEASONS = v9.REF_PERIODS
LAST = '2026_2027'
BINS = (-np.inf, 4.5, 9.5, 14.5, 19.5, 29.5, 39.5, np.inf)
LABELS = ('0-4', '5-9', '10-14', '15-19', '20-29', '30-39', '40+')
VERSION = 'xg_maturity_v10_20261120_1'


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    tmp = path.with_suffix('.pending')
    df.to_csv(tmp, index=False, float_format='%.10g')
    tmp.replace(path)
    print(f'WROTE {path} | rows={len(df)}', flush=True)


def summarize(g, tag, xg_name=XG):
    x = pd.to_numeric(g[xg_name], errors='coerce')
    y = x.dropna().to_numpy(float)
    age = pd.to_numeric(g['prior_games_min_asof'], errors='coerce')
    return dict(window=tag, games=len(g), xg_available=len(y),
                xg_missing=len(g)-len(y), xg_coverage=len(y)/len(g) if len(g) else np.nan,
                xg_sd=float(y.std()) if len(y) else np.nan,
                xg_mean=float(y.mean()) if len(y) else np.nan,
                xg_median=float(np.median(y)) if len(y) else np.nan,
                xg_mean_absolute=float(np.abs(y).mean()) if len(y) else np.nan,
                mean_prior_team_games=float(age.mean()),
                median_prior_team_games=float(age.median()),
                min_target_date=g.game_date_dt.min().strftime('%Y-%m-%d') if len(g) else '',
                max_target_date=g.game_date_dt.max().strftime('%Y-%m-%d') if len(g) else '')


def age_bins(values):
    age = pd.to_numeric(values, errors='coerce')
    require((age.dropna()>=0).all(), 'Negative prior team games in as-of features')
    return pd.cut(age, bins=BINS, labels=LABELS, include_lowest=True)


def add_progress(df):
    d = df.copy()
    require({'data_period','game_id','game_date_dt','prior_games_min_asof',XG,
             'xg_max_source_game_date'}.issubset(d), 'Required V8/V9 fields absent')
    d['game_id'] = d.game_id.astype('string')
    require(not d.game_id.isna().any() and not d.game_id.duplicated().any(),
            'xG maturity data contains missing/duplicate game IDs')
    d['game_date_dt'] = pd.to_datetime(d.game_date_dt, errors='coerce')
    require(d.game_date_dt.notna().all(), 'Target date absent')
    require(d.data_period.isin(v9.PERIODS).all(), 'Unexpected season present')
    d[XG] = pd.to_numeric(d[XG], errors='coerce')
    require(np.isfinite(d[XG].dropna().to_numpy(float)).all(), 'Nonfinite prior xG')
    require(pd.to_numeric(d.prior_games_min_asof, errors='coerce').notna().all(),
            'Prior team games field missing; cannot maturity match')
    v9.v8.validate_asof(d)
    d = d.sort_values(['data_period','game_date_dt','game_id']).reset_index(drop=True)
    d['season_day_num'] = (d.groupby('data_period').game_date_dt
                             .transform(lambda s: s.rank(method='dense')).astype(int))
    d['season_game_num'] = d.groupby('data_period').cumcount()+1
    d['age_bin'] = age_bins(d.prior_games_min_asof)
    require(d.age_bin.notna().all(), 'Prior team-games bin unavailable')
    source = pd.to_datetime(d.xg_max_source_game_date, errors='coerce')
    d['source_lag_days'] = (d.game_date_dt-source).dt.days
    require((d.loc[d[XG].notna(),'source_lag_days'] >= 1).all(),
            'Invalid xG source age: not strictly prior to target date')
    return d


def compare_windows(data):
    active = data[data.data_period==LAST]
    n_days = int(active.season_day_num.max())
    n_games = len(active)
    require(n_days>=10 and n_games>=50, 'Insufficient current-season progression')
    rows=[]
    for season, g in data.groupby('data_period', sort=True):
        windows = {
            'all_available_season':g,
            f'first_{n_days}_game_dates':g[g.season_day_num<=n_days],
            f'first_{n_games}_games':g[g.season_game_num<=n_games],
        }
        for tag, segment in windows.items():
            require(len(segment)>=25, f'Too few rows in {season}, {tag}')
            rows.append({'season':season,**summarize(segment,tag)})
    return pd.DataFrame(rows),n_days,n_games


def age_strata(data):
    rows=[]
    for (season,cat), g in data.groupby(['data_period','age_bin'], observed=True,sort=True):
        rows.append({'season':season,'prior_team_games_bin':str(cat),
                     **summarize(g,'all_available_season')})
    return pd.DataFrame(rows)


def mixture_stats(reference, target):
    """Descriptively reweight *reference* age bins to target's age distribution.

    This is not causal attribution. Fixed reference strata and target frequencies
    only; no model refit or selection on outcomes.
    """
    ref=reference.loc[reference[XG].notna()].copy()
    tgt=target.loc[target[XG].notna()].copy()
    require(len(ref)>=500 and len(tgt)>=100,'Too little nonmissing xG for age-match')
    counts=tgt.age_bin.value_counts().reindex(LABELS, fill_value=0)
    rows=[]
    first=second=0.
    for category in LABELS:
        n=int(counts[category]);weight=n/len(tgt)
        base=ref.loc[ref.age_bin.astype(str)==category,XG].to_numpy(float)
        current=tgt.loc[tgt.age_bin.astype(str)==category,XG].to_numpy(float)
        if n and len(base)<10:
            raise ValueError(f'Insufficient training-age reference for {category}: {len(base)} observations, current={n}')
        if len(base):
            a=float(base.mean()); b=float(np.mean(base**2)); sd=float(base.std())
        else:
            a=b=sd=np.nan
        if n:
            first+=weight*a
            second+=weight*b
        rows.append({'prior_team_games_bin':category,
                     'current_games_with_xg':n,
                     'current_bin_fraction':weight,
                     'reference_games_with_xg':len(base),
                     'reference_xg_sd':sd,
                     'reference_xg_mean':a,
                     'current_xg_sd':float(current.std()) if n else np.nan,
                     'current_xg_mean':float(current.mean()) if n else np.nan})
    expected=float(np.sqrt(max(0.,second-first**2)))
    require(expected>1e-4, 'Age-weighted reference distribution unexpectedly constant')
    return pd.DataFrame(rows),expected


def source_quality(data):
    rows=[]
    for season,g in data.groupby('data_period',sort=True):
        for label,section in [('all_available_season',g),('first_10_game_dates',g[g.season_day_num<=10])]:
            aged=section.loc[section[XG].notna(),'source_lag_days']
            missing=section[XG].isna()
            rows.append({'season':season,'window':label,'games':len(section),
                         'missing_xg':int(missing.sum()),
                         'missing_rate':float(missing.mean()),
                         'newest_league_wide_source_lag_median_days':float(aged.median()) if len(aged) else np.nan,
                         'newest_league_wide_source_lag_p90_days':float(aged.quantile(.9)) if len(aged) else np.nan,
                         'league_wide_newest_lag_over_7_days_fraction':float((aged>7).mean()) if len(aged) else np.nan,
                         'league_wide_newest_lag_over_14_days_fraction':float((aged>14).mean()) if len(aged) else np.nan})
    return pd.DataFrame(rows)


def normalization_audit(data, provenance):
    ref=data[data.data_period.isin(REF_SEASONS)]
    require(len(ref)>=2500 and ref.target.notna().all(),
            'Locked 2022-24 training outcomes insufficient')
    model=RidgeLogit(v9.FEATURES, float(provenance['chosen_xg_penalty_2024'])).fit(ref)
    ix=v9.FEATURES.index(XG)
    train_sd=float(model.scales[ix]); train_mean=float(model.means[ix]);
    train_median=float(model.median[ix]); beta=float(model.coefficients[1+ix]);
    require(train_sd>0 and np.isfinite(train_sd),'Fitted training-only xG scale invalid')
    rows=[]
    for season,g in data.groupby('data_period',sort=True):
        clean=g[XG].dropna().to_numpy(float)
        z=(clean-train_mean)/train_sd
        rows.append({'season':season,'xg_available':len(clean),
                     'raw_sd':float(clean.std()),
                     'sd_after_fixed_training_scaling':float(z.std()),
                     'locked_2022_24_training_xg_sd':train_sd,
                     'locked_2022_24_training_xg_mean':train_mean,
                     'locked_2022_24_training_xg_median_for_missing':train_median,
                     'locked_2022_24_model_xg_ridge_coefficient':beta,
                     'fixed_reference_transform_diagnostic_only':True,
                     'raw_sd_ratio_equals_standardized_sd_ratio':True})
    return pd.DataFrame(rows)


def produce(data, provenance):
    d=add_progress(data)
    windows,n_days,n_games=compare_windows(d)
    strata=age_strata(d)
    active=d.loc[d.data_period.eq(LAST)]
    reference=d.loc[d.data_period.isin(REF_SEASONS)]
    mixture,exp_sd=mixture_stats(reference,active)
    quality=source_quality(d)
    norm=normalization_audit(d,provenance)
    ref_full_sd=float(reference[XG].dropna().std(ddof=0))
    early_ref=reference[reference.season_day_num<=n_days]
    ref_early_sd=float(early_ref[XG].dropna().std(ddof=0))
    current_sd=float(active[XG].dropna().std(ddof=0))
    require(ref_full_sd>0 and ref_early_sd>0, 'Reference xG too constant')
    def ratio(a,b):return a/b if b else np.nan
    summary=pd.DataFrame([{'current_season':LAST,'current_games':len(active),
                           'current_game_dates':n_days,
                           '2022_24_full_xg_sd':ref_full_sd,
                           '2022_24_first_N_dates_xg_sd':ref_early_sd,
                           '2022_24_age_reweighted_xg_sd':exp_sd,
                           '2026_xg_sd':current_sd,
                           'current_to_full_training_sd_ratio':ratio(current_sd,ref_full_sd),
                           'current_to_first_N_dates_sd_ratio':ratio(current_sd,ref_early_sd),
                           'current_to_age_reweighted_sd_ratio':ratio(current_sd,exp_sd),
                           'training_full_to_early_sd_ratio':ratio(ref_early_sd,ref_full_sd),
                           'analysis':'descriptive_no_holdout_selection'}])
    return windows,strata,mixture,quality,norm,summary


def interpret(summary):
    r=summary.iloc[0]
    early=float(r['current_to_first_N_dates_sd_ratio'])
    age=float(r['current_to_age_reweighted_sd_ratio'])
    full=float(r['current_to_full_training_sd_ratio'])
    # Do not falsely assert a cause solely from an observational comparison.
    if .8<=early<=1.25 and .8<=age<=1.25:
        verdict='Early-season maturity accounts for most apparent compression descriptively; not proven causally.'
    elif early<.8 and age<.8:
        verdict='Compression persists after both stage and team-experience matching; inspect PBP xG scale/source construction.'
    elif early>1.25 and age>1.25:
        verdict='Current xG is more dispersed than maturity-matched references; full-season comparison is misleading.'
    else:
        verdict='Stage and team-experience comparisons disagree; inspect source/age strata before attributing cause.'
    return ('VERSION '+VERSION+'\n'
            'READ ONLY: no production change; V5/V8/V9 inputs preserved.\n'
            'Odds and providers are not model features; no new model trained for selection.\n'
            'Feature sources use strictly earlier calendar dates, not independently authenticated capture timestamps.\n'
            'Source lag is the newest LEAGUE-WIDE historical event date, NOT a per-team observation age.\n'
            f'CURRENT {LAST}: games={int(r.current_games)}, date_count={int(r.current_game_dates)}\n'
            f'Unmatched current/full historical SD ratio: {full:.3f}\n'
            f'Same first-N-game-dates SD ratio: {early:.3f}\n'
            f'Age-bin-reweighted reference SD ratio: {age:.3f}\n'
            'Raw xG SD and fixed-reference-standardized xG SD have the SAME relative ratio. V8 retrains its scaling on prior dates, but that cannot cause raw xG compression.\n'
            f'INTERPRETATION: {verdict}\n'
            'Age matches are descriptive and may not control schedule quality, opponent effects, or source-model changes.\n'
            'All later-season comparisons are exploratory. Do not refit/select on the 2025-27 holdout.\n')


def self_test():
    # Ensure the same early-maturity xG variability is not falsely called
    # a different season's data bug when full seasons naturally broaden.
    rows=[]
    rng=np.random.default_rng(81)
    for year in YEARS:
        days=50 if year!=2026 else 20
        for day in range(1,days+1):
            stamp=pd.Timestamp(year=year,month=10,day=1)+pd.Timedelta(days=day-1)
            for g in range(6):
                prog=(day-1)*6+g+1
                age=min(50,int((day-1)//3))
                sd=.2 if day<=12 else .55
                x=float(rng.normal(0,sd)) if day>1 else np.nan
                rows.append({'data_period':f'{year}_{year+1}',
                             'game_id':f'{year}02{prog:04d}',
                             'game_date_dt':stamp,'prior_games_min_asof':age,
                             XG:x,'xg_max_source_game_date':(stamp-pd.Timedelta(days=1)).strftime('%Y-%m-%d') if day>1 else None,
                             'target':float(rng.random()>.5)})
    data=add_progress(pd.DataFrame(rows))
    w,days,games=compare_windows(data)
    require(days==20 and games==120,'Incorrect current-season progression')
    ref=data[data.data_period.isin(REF_SEASONS)]
    active=data[data.data_period==LAST]
    m,e=mixture_stats(ref,active)
    ratio=float(active[XG].dropna().std(ddof=0)/ref[ref.season_day_num<=days][XG].dropna().std(ddof=0))
    require(.5<ratio<1.5,f'Unexpected synthetic equal-stage variability {ratio}')
    require(e>0 and len(m)==len(LABELS),'Age-reweighted reference xG failed')
    observed=ref[XG].dropna().std(ddof=0)
    require(observed>ref[ref.season_day_num<=days][XG].dropna().std(ddof=0),
            'Synthetic mature reference should be more variable than early reference')
    wrong=data.copy()
    ix=wrong.index[wrong[XG].notna()][0]
    wrong.loc[ix,'xg_max_source_game_date']=wrong.loc[ix,'game_date_dt'].strftime('%Y-%m-%d')
    try:add_progress(wrong)
    except ValueError:pass
    else:raise AssertionError('Same-day xG source leakage not rejected')
    print('SELF-TEST PASS: maturity matching, age mixture, strict prior source dates, controlled compression')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true',help='Run deterministic fixture, no local V8 caches')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    provenance,_=v9.load_locked_provenance()
    xg=v9.load_xg_cache()
    full=v9.join_locked_inputs(v9.load_history_features(),xg)
    windows,strata,mixture,quality,norm,summary=produce(full,provenance)
    save(windows,'season_progress_matched_xg.csv')
    save(strata,'prior_games_maturity_bins.csv')
    save(mixture,'age_reweighted_reference.csv')
    save(quality,'source_lag_and_missingness.csv')
    save(norm,'fixed_training_normalization.csv')
    save(summary,'compression_decomposition.csv')
    result=interpret(summary)
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'findings.txt';tmp=target.with_suffix('.pending')
    tmp.write_text(result,encoding='utf-8');tmp.replace(target)
    print(result,flush=True)
    print(f'WROTE {target}',flush=True)


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,ImportError,KeyError,TypeError) as exc:
        sys.exit('ERROR: '+str(exc))
