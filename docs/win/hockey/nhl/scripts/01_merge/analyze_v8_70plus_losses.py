#!/usr/bin/env python3
"""Analyze V8 >=70% NHL favorite losses using only EXISTING local model data.

The 2024-25 and 2025-26 seasons are historical comparison groups. Results in
2026-27 are analyzed without allowing their outcomes into pregame features.
No provider probabilities, odds, network requests, model fitting, or bet changes.

Install beside improve_probability_win_rate_60plus.py and run with Python.
Use --self-test to exercise grouping, orientation, and source matching.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import improve_probability_win_rate_60plus as prior

OUT = prior.ROOT / 'my_probability' / 'independent_research' / 'probability_win_rate_60plus' / 'loss_diagnosis'
BANDS = ((.60, .65, '60-<65%'), (.65, .70, '65-<70%'), (.70, 1.0000001, '70%+'))
FEATURES = (
    ('favorite_elo_diff', 'Elo strength'),
    ('favorite_xg_diff', 'Prior expected goals'),
    ('favorite_last10_win_diff', 'Recent win fraction'),
    ('favorite_last10_goal_diff', 'Recent goal difference'),
    ('favorite_rest_diff', 'Rest days'),
    ('favorite_back_to_back_diff', 'Back-to-back indicator'),
    ('prior_games_min_asof', 'Prior games per team (minimum)'),
)
FLAGS = {
    'xg_not_in_favor': lambda f: f.favorite_xg_diff <= 0.,
    'elo_not_in_favor': lambda f: f.favorite_elo_diff <= 0.,
    'recent_wins_not_in_favor': lambda f: f.favorite_last10_win_diff <= 0.,
    'recent_goals_not_in_favor': lambda f: f.favorite_last10_goal_diff <= 0.,
    'rest_disadvantage': lambda f: f.favorite_rest_diff < 0.,
    'back_to_back_disadvantage': lambda f: f.favorite_back_to_back_diff > 0.,
}


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def save_csv(frame: pd.DataFrame, file: str) -> None:
    dest = OUT / file
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.pending')
    frame.to_csv(tmp, index=False)
    tmp.replace(dest)
    print(f'WROTE {dest} | rows={len(frame)}', flush=True)


def prepare(pred: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    cols = ['game_id', 'home_team_full', 'away_team_full', 'home_score', 'away_score', 'game_date_dt', 'data_period', 'target']
    require(set(cols).issubset(history.columns), 'Existing NHL game history lacks required matchup/score fields')
    meta = history[cols].copy()
    meta['game_id'] = meta.game_id.astype(str)
    require(not meta.game_id.duplicated().any(), 'Duplicate historical game IDs')
    f = prior.make_features(pred, history)
    f = f.merge(meta.drop(columns=['game_date_dt', 'data_period', 'target']), on='game_id', how='left', validate='one_to_one', indicator=True)
    require(f._merge.eq('both').all(), f'Missing official matchup for {int(f._merge.ne("both").sum())} games')
    f = f.drop(columns='_merge')
    f['fav_conf'] = np.maximum(f.v8_xg_home.to_numpy(float), 1.-f.v8_xg_home.to_numpy(float))
    home = f.v8_xg_home.to_numpy(float) >= .5
    f['predicted_winner'] = np.where(home, f.home_team_full, f.away_team_full)
    f['opponent'] = np.where(home, f.away_team_full, f.home_team_full)
    f['predicted_winner_score'] = np.where(home, f.home_score, f.away_score)
    f['opponent_score'] = np.where(home, f.away_score, f.home_score)
    require(np.array_equal((f.predicted_winner_score > f.opponent_score).to_numpy(int), f.favorite_won.to_numpy(int)),
            'Favorite win/result orientation differs from actual final scores')
    f['result'] = np.where(f.favorite_won.eq(1), 'WIN', 'LOSS')
    f['band'] = 'below_60%'
    for low, high, label in BANDS:
        f.loc[(f.fav_conf >= low) & (f.fav_conf < high), 'band'] = label
    for name, rule in FLAGS.items():
        source = {'xg_not_in_favor':'favorite_xg_diff', 'elo_not_in_favor':'favorite_elo_diff',
                  'recent_wins_not_in_favor':'favorite_last10_win_diff',
                  'recent_goals_not_in_favor':'favorite_last10_goal_diff',
                  'rest_disadvantage':'favorite_rest_diff',
                  'back_to_back_disadvantage':'favorite_back_to_back_diff'}[name]
        f[name] = rule(f).astype('boolean').mask(f[source].isna(), pd.NA)
    return f


def band_summary(f: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for season, group in f.groupby('data_period', sort=True):
        for low, high, band in BANDS:
            g=group.loc[(group.fav_conf >= low)&(group.fav_conf < high)]
            n=len(g); wins=int(g.favorite_won.sum())
            rows.append({'season':season,'band':band,'games':n,'wins':wins,'losses':n-wins,
                         'win_pct':round(100*wins/n,2) if n else np.nan,
                         'avg_forecast_pct':round(100*g.fav_conf.mean(),2) if n else np.nan})
    return pd.DataFrame(rows)


def feature_comparison(f: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    segments=(('65-<70%','all'),('70%+','wins'),('70%+','losses'))
    for season, group in f.groupby('data_period', sort=True):
        for band, subset in segments:
            g=group.loc[group.band.eq(band)]
            if subset != 'all':
                g=g.loc[g.result.eq('WIN' if subset == 'wins' else 'LOSS')]
            for feat, label in FEATURES:
                series=pd.to_numeric(g[feat],errors='coerce').dropna()
                rows.append({'season':season,'band':band,'outcome':subset,'feature':feat,'label':label,
                             'games':len(g),'observed':len(series),
                             'mean':round(float(series.mean()),5) if len(series) else np.nan,
                             'median':round(float(series.median()),5) if len(series) else np.nan})
    return pd.DataFrame(rows)


def historical_flag_comparison(f: pd.DataFrame) -> pd.DataFrame:
    """Compare simple as-of disagreement flags with wins in earlier seasons.

    This is descriptive, not a trained/validated forecast adjustment.
    """
    records=[]
    for season, base in [('2024_2025',f.loc[f.data_period.eq('2024_2025')]),
                         ('2025_2026',f.loc[f.data_period.eq('2025_2026')]),
                         ('2026_2027',f.loc[f.data_period.eq('2026_2027')])]:
        confident = base.loc[base.band.eq('70%+')]
        for name in FLAGS:
            known=confident.loc[confident[name].notna()]
            for flagged in (True,False):
                subset=known.loc[known[name].eq(flagged)]
                n=len(subset); losses=int(subset.result.eq('LOSS').sum())
                records.append({'season':season,'risk_indicator':name,'flag_present':flagged,
                                'games':n,'losses':losses,'wins':n-losses,
                                'loss_pct':round(100*losses/n,2) if n else np.nan})
    return pd.DataFrame(records)


def self_test():
    history=pd.DataFrame({
        'game_id':['2026020001','2026020002','2026020003'],
        'game_date_dt':pd.to_datetime(['2026-10-10','2026-10-11','2026-10-12']),
        'data_period':['2026_2027']*3,'target':[0.,1.,1.],
        'home_team_full':['BOS','NYR','DAL'],'away_team_full':['NYI','BUF','VGK'],
        'home_score':[1,4,3], 'away_score':[3,2,2],
        'elo_diff_asof':[.2,-.25,.1], 'xg_net_diff_prior_asof':[.1,.5,-.3],
        'win_rate_last10_diff_asof':[.1,-.2,.1],
        'goals_diff_last10_diff_asof':[.2,-.1,.2],
        'rest_diff_asof':[0.,1.,-1.], 'back_to_back_diff_asof':[0.,-1.,1.],
        'prior_games_min_asof':[5,7,10]})
    pred=history[['game_id','game_date_dt','data_period','target']].copy()
    pred['v8_xg_home']=[.72,.32,.675]
    result=prepare(pred,history)
    assert result.predicted_winner.tolist()==['BOS','BUF','DAL']
    assert result.result.tolist()==['LOSS','LOSS','WIN']
    assert result.band.tolist()==['70%+','65-<70%','65-<70%']
    assert result.xg_not_in_favor.tolist()==[False,True,True]
    one=band_summary(result)
    assert one.loc[one.band.eq('70%+'),'losses'].iloc[0]==1
    check=feature_comparison(result)
    assert not check.empty
    flags=historical_flag_comparison(result)
    assert not flags.empty
    # Absent source features must not become false signals.
    history.loc[0,'xg_net_diff_prior_asof']=np.nan
    missing=prepare(pred,history)
    assert pd.isna(missing.loc[0,'xg_not_in_favor'])
    print('SELF-TEST PASS: true favorite scores, away favorite, game matching, probability boundaries, missing features',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    print('READING EXISTING V8 PREDICTIONS, NHL FINALS, AND PRIOR-DATE TEAM FEATURES (OFFLINE)',flush=True)
    val=prior.load_predictions(prior.V8_REPORTS/'2024_ablation_predictions.csv','2024_2025')
    replay=prior.load_predictions(prior.V8_REPORTS/'2025_2027_chronological_replay.csv')
    require(set(replay.data_period)=={'2025_2026','2026_2027'},'Missing one 2025-27 comparison season')
    all_pred=pd.concat([val,replay],ignore_index=True)
    require(not all_pred.game_id.duplicated().any(), 'Duplicate predictions across seasons')
    history, _=prior.load_cached_features()
    full=prepare(all_pred,history)
    summary=band_summary(full)
    cases=full.loc[(full.data_period=='2026_2027') & (full.band=='70%+')].copy()
    compare=full.loc[(full.data_period=='2026_2027') & (full.band=='65-<70%')].copy()
    print('\nOBSERVED 60%+ WIN RATES (V8 UNCHANGED):',flush=True)
    print(summary.to_string(index=False),flush=True)
    print('\n2026-27 70%+ GAMES:',len(cases),'| LOSSES:',int(cases.result.eq('LOSS').sum()),
          '| 65-70% COMPARISON GAMES:',len(compare),flush=True)
    cols=['game_date_dt','game_id','predicted_winner','opponent','fav_conf','predicted_winner_score',
          'opponent_score','result']+[f for f,_ in FEATURES]
    losses=cases.loc[cases.result.eq('LOSS'),cols].copy().sort_values(['fav_conf','game_date_dt'],ascending=[False,True])
    losses['fav_conf']=np.round(losses.fav_conf*100,2)
    print('\nEXACT 2026-27 >=70% LOSSES; PROBABILITY DISPLAYED AS PERCENT:',flush=True)
    print(losses.to_string(index=False),flush=True)
    features=feature_comparison(full)
    flags=historical_flag_comparison(full)
    print('\n2026-27 FEATURE MEDIANS: >=70% LOSSES VS >=70% WINS VS 65-70% ALL:',flush=True)
    display=features.loc[(features.season=='2026_2027')].pivot(index='label',columns=['band','outcome'],values='median')
    print(display.round(3).to_string(),flush=True)
    print('\n>=70% RISK INDICATORS (COUNTS, NOT CAUSAL CLAIMS):',flush=True)
    print(flags.to_string(index=False),flush=True)
    save_csv(summary,'2024_2027_fixed_band_win_rates.csv')
    save_csv(losses,'2026_2027_four_high_confidence_losses.csv')
    save_csv(cases[cols].sort_values(['fav_conf','game_id'],ascending=[False,True]),'2026_2027_all_70plus_games.csv')
    save_csv(compare[cols].sort_values(['fav_conf','game_id'],ascending=[False,True]),'2026_2027_all_65_to_70_games.csv')
    save_csv(features,'2024_2027_strength_signal_comparison.csv')
    save_csv(flags,'2024_2027_risk_indicator_win_loss_counts.csv')
    save_csv(full[['game_id','game_date_dt','data_period','predicted_winner','opponent','fav_conf',
                   'band','result']+[f for f,_ in FEATURES]+list(FLAGS.keys())],
             'all_games_prior_feature_audit.csv')
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'goal':'Diagnose observed 2026-27 70%+ win-rate reversal using existing independent V8 forecasts',
              '2026_2027_70plus_games':int(len(cases)),'2026_2027_70plus_losses':int(cases.result.eq('LOSS').sum()),
              '2026_2027_65_70_games':int(len(compare)),
              'provider_forecasts_used':False,'sportsbook_used':False,
              'model_parameters_changed':False,'bet_pipeline_changed':False,
              'disclaimer':'Descriptive retrospective comparisons; flags are not established causal explanations'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'WROTE {OUT / "manifest.json"}',flush=True)
    print('MODEL UNCHANGED. This is an error diagnosis, not a demonstrated correction.',flush=True)


if __name__=='__main__':
    try:
        main()
    except (OSError,ValueError,RuntimeError,KeyError,ImportError,pd.errors.ParserError) as exc:
        raise SystemExit(f'ERROR: {exc}')
