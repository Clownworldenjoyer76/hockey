#!/usr/bin/env python3
"""Optimize only the independent NHL V8 model's score-derived Elo settings.

Uses existing official game results and V8's prior-date xG feature caches.
No network, odds, D-Ratings, or SDV predictions enter model fitting.
2022-24 training; 2024-25 model selection; 2025-27 chronological comparison.
Writes only to my_probability/independent_research/elo_tuning.

Install beside predict_my_probability_v5_extended.py and
ablate_my_probability_v8_xg.py; run without arguments or with --self-test.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import predict_my_probability_v5_extended as v5
import ablate_my_probability_v8_xg as v8
from predict_my_probability_independent import RidgeLogit, brier, log_loss

OUT = v5.NHL / 'my_probability' / 'independent_research' / 'elo_tuning'
K_VALUES = (10., 20., 30., 40.)
CARRY_VALUES = (.25, .45, .65, .85)
ORIGINAL = (20., .45)
EPS = 1e-9


def ensure(condition, msg):
    if not condition:
        raise ValueError(msg)


def write_csv(frame, filename):
    dest = OUT / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.pending')
    frame.to_csv(temp, index=False)
    temp.replace(dest)
    print(f'WROTE {dest} | rows={len(frame)}', flush=True)


def write_json(obj, filename):
    dest = OUT / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.pending')
    temp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    temp.replace(dest)
    print(f'WROTE {dest}', flush=True)


def load_inputs():
    # The V8 routine confirms source fingerprints and date < predicted date.
    old = pd.concat([v5.load_old_season(None, y, offline=True)
                     for y in (2022, 2023, 2024)], ignore_index=True)
    history, slate, local = v5.historical_and_current(old)
    all_targets = pd.concat([history, slate], ignore_index=True).drop_duplicates('game_id')
    challenger = v8.get_challenger()
    chunks = []
    for year in v8.YEARS:
        f, info = v8.build_one(year, all_targets, challenger, rebuild=False)
        chunks.append(f[['game_id', v8.XG_FIELD]])
        print(f'PRIOR xG {year}: {info["filled"]}/{info["games"]}', flush=True)
    xg = pd.concat(chunks, ignore_index=True)
    ensure(not xg.game_id.duplicated().any(), 'Duplicate prior xG game IDs')
    history = history.merge(xg, how='left', on='game_id', validate='one_to_one')
    slate = slate.merge(xg, how='left', on='game_id', validate='one_to_one')
    ensure(history[v8.XG_FIELD].notna().mean() >= .75,
           'Historical xG coverage unexpectedly low')
    ensure(history.game_date_dt.notna().all(), 'Missing historical game dates')
    return history, slate, local


def variants(history, slate, k, carry):
    """Compute Elo from prior finals only, preserving all existing V8 features."""
    elo = v5.prior_elo(history, slate, k=k, offseason_retention=carry,
                       home_advantage=65.)
    ensure(not elo.game_id.duplicated().any(), 'Duplicate Elo game IDs')
    table = elo.set_index('game_id').elo_diff_asof
    def replace(frame):
        alternate = frame.game_id.map(table)
        ensure(alternate.notna().all(), 'Missing Elo target game ID')
        result = frame.copy()
        result['elo_diff_asof'] = alternate.to_numpy(float)
        return result
    result_h, result_s = replace(history), replace(slate)
    if (k, carry) == ORIGINAL:
        ensure(np.allclose(result_h.elo_diff_asof, history.elo_diff_asof,
                           rtol=0., atol=1e-10),
               'Original V8 Elo features do not reproduce: stop rather than compare')
        ensure(np.allclose(result_s.elo_diff_asof, slate.elo_diff_asof,
                           rtol=0., atol=1e-10),
               'Original current Elo features do not reproduce')
    return result_h, result_s


def scoring(df, col, label):
    sub = df.loc[df[col].notna() & df.target.notna()]
    if sub.empty:
        return None
    y = sub.target.to_numpy(float)
    p = sub[col].to_numpy(float)
    ensure(np.isfinite(p).all() and np.all((p >= 0.) & (p <= 1.)),
           f'Invalid probabilities in {label}')
    return {'model':label, 'games':len(sub), 'brier':float(brier(y,p)),
            'log_loss':float(log_loss(y,p))}


def evaluate_selection(history, slate, penalty):
    results = []
    for k in K_VALUES:
        for carry in CARRY_VALUES:
            h, _ = variants(history, slate, k, carry)
            train = h.loc[h.data_period.isin(('2022_2023','2023_2024'))]
            valid = h.loc[h.data_period.eq('2024_2025')]
            ensure(len(train)>=2200 and len(valid)>=1100,
                   'Incomplete historical train/validation periods')
            ensure(train.game_date_dt.max() < valid.game_date_dt.min(),
                   'Temporal validation boundary failed')
            model = RidgeLogit(v8.XG_FEATURES, penalty).fit(train)
            pred = model.predict(valid)
            results.append({'elo_k':k, 'offseason_carry':carry,
                            'xg_penalty_unchanged':penalty,
                            'train_games':len(train), 'validation_games':len(valid),
                            'validation_brier':float(brier(valid.target,pred)),
                            'validation_log_loss':float(log_loss(valid.target,pred)),
                            'selection_period':'2024_2025',
                            'provider_forecasts_used':False,
                            'sportsbook_odds_used':False})
            print(f'2024 VALIDATION Elo K={k:g}, carry={carry:.2f}: '
                  f'Brier={results[-1]["validation_brier"]:.6f}', flush=True)
    report = pd.DataFrame(results).sort_values(
        ['validation_brier','validation_log_loss','elo_k','offseason_carry'])
    base = report.loc[(report.elo_k == ORIGINAL[0]) &
                      (report.offseason_carry == ORIGINAL[1])].iloc[0]
    selected = report.iloc[0]
    original_candidates = (v8.OUT/'reports'/'2024_ablation_candidates.csv')
    ensure(original_candidates.is_file(),
           f'Missing existing V8 validation report: {original_candidates}')
    original = pd.read_csv(original_candidates)
    match = original.loc[(original.model == f'v8_xg_penalty_{penalty:g}')]
    ensure(len(match)==1, 'Existing V8 penalty selection cannot be verified')
    delta = abs(float(base.validation_brier) - float(match.iloc[0].brier))
    ensure(delta < 0.00002,
           f'Original V8 validation failed to reproduce (Brier delta={delta:.7f})')
    return report, selected, base


def compare_walkforward(history, source_replay, penalty, k, carry):
    h = history
    source = pd.read_csv(source_replay, dtype={'game_id':str}, low_memory=False)
    ensure({'game_id','v8_xg_home','target','data_period'}.issubset(source),
           'V8 historical replay missing required fields')
    ensure(not source.game_id.duplicated().any(), 'Duplicate game IDs in V8 replay')
    scores=[]
    targets = h.loc[h.data_period.isin(('2025_2026','2026_2027'))]
    for i, (day, block) in enumerate(targets.groupby('game_date_dt', sort=True), 1):
        prior = h.loc[h.game_date_dt < day]
        ensure(len(prior)>2500 and prior.game_date_dt.max()<day,
               'Walkforward chronology violation')
        pred = RidgeLogit(v8.XG_FEATURES, penalty).fit(prior).predict(block)
        piece = block[['game_id','game_date','data_period','target']].copy()
        piece['candidate_home'] = pred
        piece['train_games_before_target'] = len(prior)
        scores.append(piece)
        if i%60==0:
            print(f'CHRONOLOGICAL EVALUATION: {i} game dates',flush=True)
    candidate=pd.concat(scores,ignore_index=True)
    joined=candidate.merge(source.drop(columns=['game_date','data_period','target'],errors='ignore'),
                           how='left',on='game_id',validate='one_to_one')
    matched=joined.loc[joined.v8_xg_home.notna()].copy()
    ensure(len(matched)>=1600,
           f'Existing V8 replay matches only {len(matched)} games')
    if (k,carry)==ORIGINAL:
        gap=float(np.abs(matched.candidate_home-matched.v8_xg_home).max())
        ensure(gap < .00005, f'Existing V8 holdout predictions do not reproduce: {gap:.8f}')
    comparisons=[]
    fields=[('new_Elo_candidate','candidate_home'),('existing_V8_xG','v8_xg_home'),
            ('D_Ratings_benchmark','drat_home_win_prob'),
            ('SDV_benchmark','sdv_home_win_prob'),
            ('market_benchmark','vig_free_home_prob')]
    for period, frame in matched.groupby('data_period'):
        for label, col in fields:
            if col not in frame:
                continue
            rec=scoring(frame,col,label)
            if rec:
                comparisons.append({'season':period,**rec})
    return joined, pd.DataFrame(comparisons)


def make_forecast(history, slate, penalty, k, carry, requested_day=None):
    day = pd.Timestamp(requested_day.replace('_','-')) if requested_day else slate.game_date_dt.max()
    block=slate.loc[slate.game_date_dt.eq(day)]
    ensure(not block.empty, f'No current slate for {day.date()}')
    prior=history.loc[history.game_date_dt<day]
    ensure(len(prior)>2500 and prior.game_date_dt.max()<day,
           'Current slate forecast chronology violation')
    prob=RidgeLogit(v8.XG_FEATURES, penalty).fit(prior).predict(block)
    result=block[['game_id','game_date','home_team','away_team',v8.XG_FIELD]].copy()
    result['my_probability_home']=prob
    result['my_probability_away']=1.-prob
    result['elo_k']=k
    result['elo_offseason_carry']=carry
    result['xg_penalty']=penalty
    result['model_inputs']='NHL_results_Elo_rest_form_prior_SDV_PBP_xG'
    result['uses_dratings_as_input']=False
    result['uses_sdv_predictions_as_input']=False
    result['uses_sportsbook_odds_as_input']=False
    result['production_changed']=False
    result['generated_at_utc']=datetime.now(timezone.utc).isoformat()
    return result


def self_test():
    dates=pd.to_datetime(['2023-10-10','2023-10-10','2023-10-11','2023-10-12'])
    rows=pd.DataFrame({'game_id':['2023020001','2023020002','2023020003','2023020004'],
                       'game_date_dt':dates, 'data_period':['2023_2024']*4,
                       'home_team_full':['BOS','BUF','BOS','NYR'],
                       'away_team_full':['NYR','MTL','BUF','BOS'],
                       'home_score':[3,2,4,1], 'away_score':[1,3,3,2],
                       'target':[1.,0.,1.,0.]})
    empty=rows.iloc[:0].copy()
    low=v5.prior_elo(rows,empty,k=10.,offseason_retention=.45)
    high=v5.prior_elo(rows,empty,k=40.,offseason_retention=.45)
    ensure(np.allclose(low.elo_diff_asof.to_numpy()[:2],0),
           'Same-day feature includes same-day results')
    ensure(np.any(np.abs(low.elo_diff_asof.to_numpy()[2:]-
                         high.elo_diff_asof.to_numpy()[2:])>1e-6),
           'Changing Elo K had no effect on later games')
    rng=np.random.default_rng(7)
    cols={x:rng.normal(size=350) for x in v8.XG_FEATURES}
    cols['target']=(rng.random(350)<.55).astype(float)
    df=pd.DataFrame(cols)
    m=RidgeLogit(v8.XG_FEATURES,60.).fit(df)
    ensure(np.isfinite(m.predict(df)).all(), 'Ridge model invalid')
    print('SELF-TEST PASS: same-day isolation, Elo parameter effect, independent xG fit')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--date',help='YYYY_MM_DD, defaults latest merged slate')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    history,slate,_=load_inputs()
    manifest=v8.OUT/'reports'/'selection_provenance.json'
    ensure(manifest.is_file(), f'Missing V8 selection provenance: {manifest}')
    selection=json.loads(manifest.read_text(encoding='utf-8'))
    penalty=float(selection['chosen_xg_penalty_2024'])
    report, winner, base=evaluate_selection(history,slate,penalty)
    k,carry=float(winner.elo_k),float(winner.offseason_carry)
    gain=float(base.validation_brier-winner.validation_brier)
    improved=(k,carry)!=ORIGINAL and gain>EPS
    if not improved:
        k,carry=ORIGINAL
    new_hist,new_slate=variants(history,slate,k,carry)
    replay_path=v8.OUT/'reports'/'2025_2027_chronological_replay.csv'
    ensure(replay_path.is_file(),f'Missing V8 replay: {replay_path}')
    replay, comparison=compare_walkforward(new_hist,replay_path,penalty,k,carry)
    forecast=make_forecast(new_hist,new_slate,penalty,k,carry,args.date)
    report=report.copy()
    report['brier_improvement_vs_original']=float(base.validation_brier)-report.validation_brier
    write_csv(report,'2024_2025_elo_selection.csv')
    write_csv(replay,'2025_2027_elo_chronological_comparison.csv')
    write_csv(comparison,'2025_2027_model_benchmarks.csv')
    write_csv(forecast,f'predictions/{forecast.game_date.iloc[0]}_independent_elo_candidate.csv')
    write_json({'selected_elo_k':k,'selected_offseason_carry':carry,
                'original_elo_k':ORIGINAL[0],'original_offseason_carry':ORIGINAL[1],
                'xg_penalty_unchanged':penalty,
                'validation_original_brier':float(base.validation_brier),
                'validation_selected_brier':float(winner.validation_brier),
                'validation_improvement':gain,
                'elo_changed_from_existing_v8':improved,
                'selection_data':'2024_2025_only',
                'evaluation_periods':['2025_2026','2026_2027'],
                'uses_odds_as_feature':False,
                'uses_dratings_or_sdv_predictions_as_feature':False,
                'production_changed':False}, 'run_manifest.json')
    print(f'SELECTED Elo K={k:g}, offseason carry={carry:.2f}; '
          f'2024 validation Brier improvement={gain:+.6f}',flush=True)
    print('COMPARISON:')
    print(comparison[['season','model','games','brier','log_loss']].to_string(index=False))
    print('Research output only; existing V8 and bet selection unchanged.',flush=True)


if __name__=='__main__':
    try:
        main()
    except (ValueError,KeyError,FileNotFoundError,RuntimeError) as exc:
        raise SystemExit(f'ERROR: {exc}')
