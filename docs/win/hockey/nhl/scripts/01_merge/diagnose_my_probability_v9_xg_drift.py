#!/usr/bin/env python3
"""V9 read-only investigation of V8 xG seasonal drift and coefficient instability.

Research only. Reuses V8 strict-prior xG caches; never reconstructs xG, fetches
network data, changes V5/V8, selects a new model, or modifies betting stages.

Source order:
  2022-23 + 2023-24: training (locked V8); 2024-25: locked selection;
  2025-26 and 2026-27: previously examined historical replay, analysis only.

Run beside predict_my_probability_v5_extended.py and
ablate_my_probability_v8_xg.py, which remain unchanged.

Outputs solely under NHL/my_probability/independent_v9_xg_diagnostics/reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ablate_my_probability_v8_xg as v8
from predict_my_probability_independent import RidgeLogit, brier

OUT = v8.ROOT / 'my_probability' / 'independent_v9_xg_diagnostics' / 'reports'
XG = v8.XG_FIELD
FEATURES = v8.XG_FEATURES
YEARS = v8.YEARS
PERIODS = tuple(f'{y}_{y+1}' for y in YEARS)
REF_PERIODS = ('2022_2023', '2023_2024')
MISSING = 'missing'
VERSION = 'xg_drift_v9_20261120_2_overlap_fix'


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def save(frame, filename):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / filename
    temporary = target.with_suffix('.pending')
    frame.to_csv(temporary, index=False)
    temporary.replace(target)
    print(f'WROTE {target} | rows={len(frame)}', flush=True)


def save_text(text, filename='findings.txt'):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / filename
    temporary = target.with_suffix('.pending')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(target)
    print(f'WROTE {target}', flush=True)


def load_xg_cache():
    parts = []
    for year in YEARS:
        season = f'{year}_{year+1}'
        path = v8.CACHE / f'season_{year}_prior_xg.csv'
        ensure(path.is_file(), f'V8 prior-xG cache absent: {path}; run V8 first')
        data = pd.read_csv(path, dtype={'game_id': 'string'}, low_memory=False)
        expected = {'game_id', 'game_date_dt', XG, 'xg_max_source_game_date'}
        ensure(expected.issubset(data.columns), f'V8 cache {season} missing {expected - set(data)}')
        ensure(data.game_id.notna().all() and data.game_id.str.fullmatch(str(year)+r'02\d{4}').all(),
               f'V8 cache {season} contains invalid regular-season official game IDs')
        ensure(not data.game_id.duplicated().any(), f'V8 cache {season}: repeated game_id')
        data['game_date_dt'] = pd.to_datetime(data['game_date_dt'], errors='coerce')
        ensure(data.game_date_dt.notna().all(), f'V8 cache {season}: missing date')
        data[XG] = pd.to_numeric(data[XG], errors='coerce')
        ensure(not np.isinf(data[XG]).any(), f'V8 cache {season}: infinite xG')
        v8.validate_asof(data)
        data['data_period'] = season
        parts.append(data[['game_id','game_date_dt','data_period',XG,'xg_max_source_game_date']].copy())
    joined = pd.concat(parts, ignore_index=True)
    ensure(not joined.game_id.duplicated().any(), 'xG cache repeats game_id across years')
    return joined


def load_locked_provenance():
    path = v8.OUT / 'reports' / 'selection_provenance.json'
    ensure(path.is_file(), f'V8 locked selection provenance absent: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    ensure(data.get('parameter_selection') == '2024_2025_only',
           'Refusing non-2024 V8 model selection')
    ensure(data.get('training_seasons') == list(REF_PERIODS),
           'Refusing unexpected V8 training seasons')
    ensure(data.get('odds_as_model_features') is False,
           'Refusing V8 model with market probabilities as features')
    penalty = float(data['chosen_xg_penalty_2024'])
    ensure(penalty in v8.PENALTIES, f'Unexpected locked xG penalty {penalty}')
    return data, penalty


def combine_official_history_and_slate(history, current):
    """Keep completed history when the same game is also in the current slate.

    The repository's V5 source deliberately carries both historical scores and
    a current-slate view. As a slate becomes final, IDs legitimately overlap.
    Only exact same-game overlaps can be deduplicated; discrepancies are fatal.
    """
    history = history.copy()
    current = current.copy()
    for frame in (history, current):
        frame['game_id'] = frame['game_id'].astype('string')
        ensure(frame.game_id.notna().all() and frame.game_id.str.fullmatch(r'\d{10}').all(),
               'Official history/slate contains missing or invalid game IDs')
        ensure(not frame.game_id.duplicated().any(), 'Duplicate game IDs WITHIN history or slate')
        frame['game_date_dt'] = pd.to_datetime(frame['game_date_dt'], errors='coerce')
        ensure(frame.game_date_dt.notna().all(), 'Official history/slate missing game dates')
    overlap = history.loc[history.game_id.isin(current.game_id)]
    if not overlap.empty:
        matched = overlap.merge(current, on='game_id', how='inner',
                                suffixes=('_history','_slate'), validate='one_to_one')
        for field in ('game_date_dt','data_period','home_team_full','away_team_full'):
            left, right = matched[field+'_history'], matched[field+'_slate']
            ensure(left.notna().all() and right.notna().all() and left.eq(right).all(),
                   f'Conflicting overlapping history/slate {field}: '
                   f'{matched.loc[~left.eq(right),"game_id"].head(5).tolist()}')
        for field in ('target','home_score','away_score'):
            if field+'_history' not in matched or field+'_slate' not in matched:
                continue
            left = pd.to_numeric(matched[field+'_history'], errors='coerce')
            right = pd.to_numeric(matched[field+'_slate'], errors='coerce')
            both = left.notna() & right.notna()
            ensure(left[both].eq(right[both]).all(),
                   f'Conflicting overlapping history/slate final {field}: '
                   f'{matched.loc[both & ~left.eq(right),"game_id"].head(5).tolist()}')
        print(f'VALIDATED {len(overlap)} completed-history/current-slate overlapping game IDs; '
              'kept completed history, without duplicate training rows', flush=True)
    full = pd.concat([history, current.loc[~current.game_id.isin(history.game_id)]],
                     ignore_index=True)
    ensure(full.game_id.notna().all() and not full.game_id.duplicated().any(),
           'Official history + slate includes duplicate/missing game IDs after reconciliation')
    return full


def load_history_features():
    older = [v8.v5.load_old_season(None, y, offline=True) for y in (2022,2023,2024)]
    history, current, _ = v8.v5.historical_and_current(pd.concat(older, ignore_index=True))
    full = combine_official_history_and_slate(history, current)
    ensure(set(FEATURES).issubset(set(full.columns) | {XG}), 'Missing reconstructed V5 inputs')
    return full


def join_locked_inputs(history, xg):
    h = history.copy()
    h['game_date_dt'] = pd.to_datetime(h.game_date_dt)
    for col in (XG,'xg_max_source_game_date'):
        ensure(col not in h, f'Source history unexpectedly already has {col}')
    m = xg.merge(h, on='game_id', how='left', suffixes=('','_official'), validate='one_to_one', indicator=True)
    ensure((m['_merge']=='both').all(),
           f'No local official history/slate match for {int((m._merge!="both").sum())} cached xG IDs')
    ensure((m.game_date_dt == pd.to_datetime(m.game_date_dt_official)).all(),
           'Cached xG target date differs from local official history')
    ensure((m.data_period == m.data_period_official).all(),
           'Cached xG season differs from local official history')
    m = m.drop(columns=['_merge','game_date_dt_official','data_period_official'])
    ensure(m[XG].notna().groupby(m.data_period).mean().min() >= .75,
           'Season-specific prior-xG coverage below 75%; abort drift analysis')
    return m.sort_values(['game_date_dt','game_id']).reset_index(drop=True)


def ks_stat(a, b):
    """Descriptive two-sample empirical CDF distance, not an independence p-value."""
    x = np.sort(np.asarray(a, float)); y = np.sort(np.asarray(b, float))
    ensure(len(x) and len(y), 'Cannot compute KS distance with empty season')
    at = np.unique(np.concatenate([x, y]))
    return float(np.max(np.abs(np.searchsorted(x, at, side='right')/len(x) -
                               np.searchsorted(y, at, side='right')/len(y))))


def distribution_summary(data):
    reference = data.loc[data.data_period.isin(REF_PERIODS), XG].dropna().to_numpy(float)
    ensure(len(reference)>2000, f'Not enough 2022-24 reference xG: {len(reference)}')
    mu, sd = float(reference.mean()), float(reference.std())
    ensure(sd > .01, 'Training reference xG variation unexpectedly tiny')
    rows=[]
    for season in PERIODS:
        sub = data.loc[data.data_period.eq(season)]
        values = sub[XG].dropna().to_numpy(float)
        ensure(len(values),f'No prior-xG values in {season}')
        rows.append({'season':season,'games':len(sub),'prior_xg_available':len(values),
                     'prior_xg_missing':len(sub)-len(values),'coverage':len(values)/len(sub),
                     'xg_mean':float(values.mean()),'xg_sd':float(values.std()),
                     'mean_abs_xg':float(np.abs(values).mean()),
                     'p05':float(np.quantile(values,.05)), 'p25':float(np.quantile(values,.25)),
                     'p50':float(np.quantile(values,.50)), 'p75':float(np.quantile(values,.75)),
                     'p95':float(np.quantile(values,.95)),
                     'fraction_abs_above_0_5':float((np.abs(values)>.5).mean()),
                     'fraction_abs_above_1_0':float((np.abs(values)>1.).mean()),
                     'ks_distance_from_2022_24_training':ks_stat(reference,values),
                     'mean_shift_in_training_sd':float((values.mean()-mu)/sd),
                     'sd_ratio_to_2022_24_training':float(values.std()/sd),
                     'reference':'pooled_2022_2023_and_2023_2024'})
    return pd.DataFrame(rows), reference


def progression_models(data, penalty, reference):
    """Hold fixed the V8 feature set and its 2024-selected penalty."""
    reference_sd = float(np.std(reference))
    ref_data = data.loc[data.data_period.isin(REF_PERIODS)]
    ensure(len(ref_data)>=2500, 'Unexpectedly short initial training seasons')
    rows=[]
    windows = [
        ('2022_only', ('2022_2023',)),
        ('through_2023', REF_PERIODS),
        ('through_2024', REF_PERIODS+('2024_2025',)),
        ('through_2025', REF_PERIODS+('2024_2025','2025_2026')),
    ]
    # All historical fits here are descriptive; there is no new model selection.
    for label, periods in windows:
        train = data.loc[data.data_period.isin(periods) & data.target.notna()].copy()
        ensure(len(train)>1000 and train.target.nunique()==2,
               f'Insufficient historical outcomes for {label}')
        last = pd.to_datetime(train.game_date_dt.max())
        for period in PERIODS:
            if period in periods:
                continue
            first = data.loc[data.data_period.eq(period),'game_date_dt'].min()
            if pd.notna(first) and set(periods).issubset(set(PERIODS[:PERIODS.index(period)])):
                ensure(last < first, f'{label} includes target/future results for {period}')
        model = RidgeLogit(FEATURES,penalty).fit(train)
        size = len(FEATURES)
        for j,feature in enumerate(FEATURES):
            std = float(model.coefficients[1+j]); missing = float(model.coefficients[1+size+j])
            scale = float(model.scales[j]); median = float(model.median[j]); mean = float(model.means[j])
            rows.append({'training_window':label,'seasons_in_training':';'.join(periods),
                         'training_games':len(train),'last_training_date':last.strftime('%Y-%m-%d'),
                         'feature':feature,'ridge_penalty_locked_2024':penalty,
                         'coefficient_per_training_standard_deviation':std,
                         'coefficient_per_raw_unit':std/scale,
                         'coefficient_per_fixed_training_xg_sd':(std/scale)*reference_sd if feature==XG else np.nan,
                         'missingness_coefficient':missing,
                         'training_feature_mean_after_imputation':mean,
                         'training_feature_sd_after_imputation':scale,
                         'training_feature_median_for_imputation':median,
                         'training_feature_missing_rate':float(train[feature].isna().mean()),
                         'model_diagnostics_only':True})
    return pd.DataFrame(rows)


def season_only_coefs(data, penalty, reference):
    ref_sd = float(np.std(reference))
    rows=[]
    for season in PERIODS[:-1]:
        train = data.loc[data.data_period.eq(season)&data.target.notna()]
        ensure(len(train)>=250, f'Not enough completed games for {season}')
        m = RidgeLogit(FEATURES,penalty).fit(train)
        j = list(FEATURES).index(XG)
        coef = float(m.coefficients[j+1]); per_unit=coef/float(m.scales[j])
        rows.append({'season':season,'games':len(train),
                     'xg_coefficient_per_raw_unit':per_unit,
                     'xg_coefficient_per_fixed_reference_sd':per_unit*ref_sd,
                     'xg_coefficient_per_training_standard_deviation':coef,
                     'xg_missingness_coefficient':float(m.coefficients[1+len(FEATURES)+j]),
                     'xg_missing_rate':float(train[XG].isna().mean()),
                     'training_only_association_not_forecast':True})
    return pd.DataFrame(rows)


def quantile_bins(reference):
    edges = np.quantile(reference,[0,.2,.4,.6,.8,1.])
    edges=np.unique(edges)
    ensure(len(edges)>=4, 'Too few distinct xG quantiles to build stable reference bins')
    edges[0] = -np.inf;edges[-1] = np.inf
    return edges


def signal_bins(data, reference):
    edges=quantile_bins(reference)
    d=data.copy()
    d['reference_xg_bin'] = pd.cut(d[XG], bins=edges, include_lowest=True, duplicates='raise').astype(str)
    d.loc[d[XG].isna(),'reference_xg_bin']=MISSING
    rows=[]
    for (season,interval),g in d.groupby(['data_period','reference_xg_bin'],dropna=False,sort=True):
        complete=g.loc[g.target.notna()]
        rows.append({'season':season,'bin_2022_24_reference':str(interval),
                     'games':len(g),'completed_games':len(complete),
                     'mean_prior_xg':float(g[XG].mean()) if g[XG].notna().any() else np.nan,
                     'empirical_home_win_rate':float(complete.target.mean()) if len(complete) else np.nan,
                     'mean_elo_diff_asof':float(g.elo_diff_asof.mean()),
                     'fixed_bins_no_holdout_selection':True})
    return pd.DataFrame(rows)


def month_coverage(data):
    x=data.copy()
    x['season_month']=x.game_date_dt.dt.strftime('%Y-%m')
    rows=[]
    for (season,month),g in x.groupby(['data_period','season_month'],sort=True):
        a=g[XG].dropna()
        rows.append({'season':season,'month':month,'games':len(g),
                     'prior_xg_coverage':len(a)/len(g),
                     'prior_xg_mean':float(a.mean()) if len(a) else np.nan,
                     'prior_xg_sd':float(a.std(ddof=0)) if len(a) else np.nan,
                     'mean_abs_prior_xg':float(a.abs().mean()) if len(a) else np.nan,
                     'mean_prior_team_games':float(pd.to_numeric(g.prior_games_min_asof,errors='coerce').mean())})
    return pd.DataFrame(rows)


def replay_segments(replay_path, xg_data):
    ensure(replay_path.is_file(),f'Missing V8 replay: {replay_path}')
    r = pd.read_csv(replay_path,dtype={'game_id':'string'},low_memory=False)
    required={'game_id','data_period','target','v5_base_home','v8_xg_home'}
    ensure(required.issubset(r.columns),f'V8 replay missing {required-set(r.columns)}')
    ensure(not r.game_id.duplicated().any(),'V8 replay duplicate game ID')
    r = r.merge(xg_data[['game_id',XG,'game_date_dt']],on='game_id',how='left',validate='one_to_one',suffixes=('','_cache'))
    if f'{XG}_cache' in r:
        original = pd.to_numeric(r[XG], errors='coerce')
        cached = pd.to_numeric(r[f'{XG}_cache'], errors='coerce')
        ensure((original.isna() == cached.isna()).all(), 'Replay/cache prior-xG missingness disagrees')
        overlap = original.notna()
        ensure(np.allclose(original[overlap], cached[overlap], atol=1e-7, rtol=0),
               'Replay and immutable V8 xG-cache values disagree')
    ensure(r[XG].notna().mean()>.9, 'V8 replay xG missing for too many games')
    if 'game_date_dt_cache' in r:
        # Replay dates may be CSV strings, cache dates are datetime.
        ensure((pd.to_datetime(r.game_date_dt,errors='coerce')==pd.to_datetime(r.game_date_dt_cache)).all(),
               'V8 replay games differ from V8 xG cache target dates')
    r['loss_v8_minus_v5']=(r.v8_xg_home-r.target)**2-(r.v5_base_home-r.target)**2
    r['xg_magnitude_bin']=pd.cut(r[XG].abs(),
         bins=[-np.inf,.25,.5,1.,2.,np.inf],labels=['0_to_0.25','0.25_to_0.5','0.5_to_1','1_to_2','over_2'])
    results=[]
    for (season,bin_name),g in r.groupby(['data_period','xg_magnitude_bin'],sort=True,observed=True):
        results.append({'season':season,'xg_abs_bin':str(bin_name),'games':len(g),
                        'average_abs_xg':float(g[XG].abs().mean()),
                        'mean_probability_shift_xg_minus_v5':float((g.v8_xg_home-g.v5_base_home).mean()),
                        'mean_abs_probability_shift':float((g.v8_xg_home-g.v5_base_home).abs().mean()),
                        'v5_brier':float(brier(g.target,g.v5_base_home)),
                        'xg_brier':float(brier(g.target,g.v8_xg_home)),
                        'xg_minus_v5_brier':float(g.loss_v8_minus_v5.mean()),
                        'interpretation':'exploratory_2025_2027_replay_not_new_model_selection'})
    return pd.DataFrame(results)


def selftest():
    # A game becomes historical once completed, but may remain in the current
    # slate. This valid overlap must not be mistaken for corrupted game IDs.
    h = pd.DataFrame([{'game_id':'2026020001','game_date_dt':'2026-11-19',
                       'data_period':'2026_2027','home_team_full':'BOS',
                       'away_team_full':'NYR','target':1.,'home_score':3.,'away_score':2.}])
    slate = h.copy()
    slate.loc[0,['target','home_score','away_score']] = np.nan
    cleaned = combine_official_history_and_slate(h, slate)
    ensure(len(cleaned)==1 and float(cleaned.target.iloc[0])==1.,
           'Completed overlapping game was not kept uniquely')
    malformed = slate.copy()
    malformed.loc[0,'away_team_full']='BUF'
    try:
        combine_official_history_and_slate(h, malformed)
    except ValueError:
        pass
    else:
        raise AssertionError('Conflicting history/slate team identity was not rejected')
    malformed = h.copy()
    malformed.loc[0,'home_score']=4.
    try:
        combine_official_history_and_slate(h, malformed)
    except ValueError:
        pass
    else:
        raise AssertionError('Conflicting overlapping final scores were not rejected')
    good = pd.DataFrame([{'game_id':'2024020001','game_date_dt':pd.Timestamp('2024-10-10'),
                          XG:1.5,'xg_max_source_game_date':'2024-10-09'},
                         {'game_id':'2024020002','game_date_dt':pd.Timestamp('2024-10-10'),
                          XG:np.nan,'xg_max_source_game_date':None}])
    v8.validate_asof(good)
    bad=good.copy();bad.loc[0,'xg_max_source_game_date']='2024-10-10'
    try:v8.validate_asof(bad)
    except ValueError:pass
    else:raise AssertionError('Same-day xG leakage was not rejected')
    a = np.array([-.5,0.,.5,1.]); b = np.array([-.5,0.,.5,1.])
    ensure(abs(ks_stat(a,b))<1e-12,'KS equality regression failure')
    ensure(ks_stat(a,np.array([3.,4.,5.,6.]))>.99,'KS separation regression failure')
    rng = np.random.default_rng(20)
    train = pd.DataFrame({x:rng.normal(size=480) for x in FEATURES})
    train['target']=(rng.random(480)<.5).astype(int)
    model=RidgeLogit(FEATURES,60.).fit(train)
    j=list(FEATURES).index(XG)
    ensure(np.isfinite(model.coefficients[1+j]/model.scales[j]),'xG coefficient regression failed')
    print('SELF-TEST PASS: strict prior-xG timestamps, drift distance, locked ridge coefficient diagnostics',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true',help='Run deterministic checks without local history/cache')
    args=p.parse_args()
    if args.self_test:
        selftest();return
    provenance,penalty=load_locked_provenance()
    xg=load_xg_cache()
    full=join_locked_inputs(load_history_features(),xg)
    ref=full.loc[full.data_period.isin(REF_PERIODS)]
    ensure(len(ref)>=2500,'Missing locked 2022-24 training data')
    ensure((ref.target.notna()).all(),'Missing training outcomes from V5 official games')
    descriptive,reference=distribution_summary(full)
    windows=progression_models(full,penalty,reference)
    solo=season_only_coefs(full,penalty,reference)
    bins=signal_bins(full,reference)
    monthly=month_coverage(full)
    replay=replay_segments(v8.OUT/'reports'/'2025_2027_chronological_replay.csv',full)
    save(descriptive,'season_xg_distribution.csv')
    save(monthly,'monthly_xg_distribution.csv')
    save(windows,'expanding_coefficient_stability.csv')
    save(solo,'single_season_coefficient_diagnostics.csv')
    save(bins,'fixed_training_quantile_outcome_bins.csv')
    save(replay,'holdout_xg_magnitude_errors_exploratory.csv')
    xg_rows=windows.loc[windows.feature.eq(XG)]
    c=xg_rows.set_index('training_window').coefficient_per_raw_unit.to_dict()
    pre=c['through_2023'];post=c['through_2025']
    first=descriptive.set_index('season')
    rows=[f'VERSION {VERSION}',
          'READ ONLY: V5 and V8 untouched. No refits are used for forecasts or selection.',
          f'LOCKED V8 xG RIDGE PENALTY = {penalty:g}, selected on 2024-25 only.',
          'HOLDOUTS 2025-26 / 2026-27: exploratory only, previously examined.',
          'xG features are prior-date only. Historical source capture timestamps are not independently authenticated.',
          f'xG reference 2022-24 SD = {float(np.std(reference)):.6f}',
          f'XG COEFFICIENT: 2022-24 {pre:+.6f} log-odds/unit; through 2025-26 {post:+.6f} log-odds/unit',
          f'2025-26 xG SD ratio to training = {first.loc["2025_2026","sd_ratio_to_2022_24_training"]:.3f}',
          f'2026-27 xG SD ratio to training = {first.loc["2026_2027","sd_ratio_to_2022_24_training"]:.3f}',
          'No significance claims for correlated daily games or exploratory segments.',
          'Do not choose parameters based on these holdouts. Do not integrate with live betting.']
    text='\n'.join(rows)+'\n'
    save_text(text)
    print(text,flush=True)


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,ImportError,KeyError) as exc:
        sys.exit(f'ERROR: {exc}')
