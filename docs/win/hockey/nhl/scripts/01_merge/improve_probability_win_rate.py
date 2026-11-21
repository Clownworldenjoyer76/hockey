#!/usr/bin/env python3
"""Improve the independent NHL forecast specifically for: higher probability -> higher win %.

Runs offline using existing V8 2024-25 validation, V8 2025-27 replay,
local historical completed games, and existing prior-xG caches. Learns whether
V8's chosen favorite wins using prior observed outcomes and V8's own prior
team-strength/form/xG features. D-Ratings, SDV forecasts, and market odds are
not model inputs; no bet selection or existing models are modified.

2024-25: chronological train/validation folds for selection.
2025-26: evaluate with model fitted on completed 2024-25.
2026-27: evaluate with model fitted on completed 2024-25 and 2025-26.
A re-ranker can change future probability ordering; monotonic historical win
rates cannot be guaranteed for an unseen season. Reports actual quintile rates.

Place next to predict_my_probability_v5_extended.py, ablate_my_probability_v8_xg.py,
and improve_independent_elo.py. Run: python improve_probability_win_rate.py
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from predict_my_probability_independent import RidgeLogit
import predict_my_probability_v5_extended as v5
import improve_independent_elo as previous

ROOT = v5.NHL
OUT = ROOT / 'my_probability' / 'independent_research' / 'probability_win_rate_improvement'
V8_REPORTS = ROOT / 'my_probability' / 'independent_v8_xg' / 'reports'
V8_FORECASTS = ROOT / 'my_probability' / 'independent_v8_xg' / 'predictions'
EPS = 1e-9

# The only selection goal: observed favorite win rates ascend as probability rises.
# Small, pre-specified feature groups and penalties, selected on 2024 only.
GROUPS = {
    'confidence_only': ('v8_favorite_edge', 'favorite_home'),
    'strength_agreement': ('v8_favorite_edge', 'favorite_home',
                           'favorite_elo_diff', 'favorite_xg_diff'),
    'strength_form': ('v8_favorite_edge', 'favorite_home',
                      'favorite_elo_diff', 'favorite_xg_diff',
                      'favorite_last10_win_diff', 'favorite_last10_goal_diff',
                      'favorite_rest_diff', 'favorite_back_to_back_diff',
                      'prior_games_min_asof'),
}
PENALTIES = (15., 100.)
FEATURE_SOURCE = {
    'favorite_elo_diff': 'elo_diff_asof',
    'favorite_xg_diff': 'xg_net_diff_prior_asof',
    'favorite_last10_win_diff': 'win_rate_last10_diff_asof',
    'favorite_last10_goal_diff': 'goals_diff_last10_diff_asof',
    'favorite_rest_diff': 'rest_diff_asof',
    'favorite_back_to_back_diff': 'back_to_back_diff_asof',
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    frame.to_csv(temp, index=False)
    temp.replace(path)
    print(f'WROTE {path} | rows={len(frame)}', flush=True)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(obj, indent=2, default=str) + '\n', encoding='utf-8')
    temp.replace(path)
    print(f'WROTE {path}', flush=True)


def load_predictions(path: Path, season: str | None = None) -> pd.DataFrame:
    require(path.is_file(), f'Missing existing independent V8 prediction file: {path}')
    frame = pd.read_csv(path, dtype={'game_id': str}, low_memory=False)
    required = {'game_id', 'game_date', 'data_period', 'target', 'v8_xg_home'}
    require(required.issubset(frame.columns), f'Missing V8 columns: {required-set(frame.columns)}')
    require(frame.game_id.notna().all() and not frame.game_id.duplicated().any(),
            'Duplicate or missing V8 game IDs')
    frame['game_date_dt'] = pd.to_datetime(frame.game_date.astype(str).str.replace('_', '-', regex=False), errors='coerce')
    require(frame.game_date_dt.notna().all(), 'Unrecognized V8 game dates')
    frame['target'] = pd.to_numeric(frame.target, errors='coerce')
    frame['v8_xg_home'] = pd.to_numeric(frame.v8_xg_home, errors='coerce')
    require(frame.target.isin([0., 1.]).all(), 'Invalid V8 final winners')
    require(np.isfinite(frame.v8_xg_home.to_numpy(float)).all() and
            frame.v8_xg_home.between(0., 1., inclusive='neither').all(),
            'Invalid V8 win probabilities')
    if season is not None:
        require(set(frame.data_period) == {season}, f'Expected season {season}')
    return frame[['game_id', 'game_date_dt', 'data_period', 'target', 'v8_xg_home']].copy()


def make_features(pred: pd.DataFrame, existing_features: pd.DataFrame) -> pd.DataFrame:
    """Orient all historical pregame signals toward the side V8 favored."""
    required = {'game_id', *FEATURE_SOURCE.values(), 'prior_games_min_asof'}
    require(required.issubset(existing_features.columns),
            f'Existing historical data missing: {sorted(required-set(existing_features.columns))}')
    require(not existing_features.game_id.duplicated().any(), 'Duplicate game IDs in features')
    feat = existing_features[list(required)].copy()
    feat['game_id'] = feat.game_id.astype(str)
    merged = pred.merge(feat, on='game_id', validate='one_to_one', how='left', indicator=True)
    require(merged._merge.eq('both').all(),
            f'No saved historical feature match for {int(merged._merge.ne("both").sum())} games')
    merged.drop(columns='_merge', inplace=True)
    fav_home = merged.v8_xg_home.to_numpy(float) >= .5
    sign = np.where(fav_home, 1., -1.)
    merged['favorite_home'] = fav_home.astype(float)
    merged['v8_favorite_edge'] = np.abs(merged.v8_xg_home.to_numpy(float) - .5)
    for col, source in FEATURE_SOURCE.items():
        merged[col] = sign * pd.to_numeric(merged[source], errors='coerce').to_numpy(float)
    merged['prior_games_min_asof'] = pd.to_numeric(merged['prior_games_min_asof'], errors='coerce')
    require(np.isfinite(merged.v8_favorite_edge).all(), 'Nonfinite confidence feature')
    if 'target' in merged.columns:
        merged['favorite_won'] = np.where(fav_home, merged.target, 1. - merged.target)
        merged['favorite_won'] = merged.favorite_won.astype(float)
    return merged


def to_home_probability(original_home: np.ndarray, favorite_correct_probability: np.ndarray) -> np.ndarray:
    """Convert learned probability the ORIGINAL favorite wins into home probability.

    If meta-model predicts the original favorite is below 50%, the favored side
    can reverse; the actual probability and observed winner are then re-audited.
    """
    original_home = np.asarray(original_home, dtype=float)
    favorite_correct_probability = np.asarray(favorite_correct_probability, dtype=float)
    require(np.isfinite(favorite_correct_probability).all(), 'Nonfinite new probability')
    require(np.all((favorite_correct_probability > 0.) & (favorite_correct_probability < 1.)),
            'New probabilities must be strictly between 0 and 1')
    return np.where(original_home >= .5, favorite_correct_probability, 1. - favorite_correct_probability)


def group_audit(frame: pd.DataFrame, col: str, season: str, model_name: str) -> tuple[pd.DataFrame, dict]:
    p = frame[col].to_numpy(float)
    y = frame.target.to_numpy(float)
    require(np.isfinite(p).all() and ((p>0.) & (p<1.)).all(), f'{model_name}: invalid probabilities')
    require(np.isin(y, [0.,1.]).all(), f'{model_name}: missing actual outcomes')
    n = len(frame)
    require(n >= 125, f'{season}: too few games for 5 equal-size groups: {n}')
    chosen_win = np.where(p >= .5, y, 1.-y)
    confidence = np.maximum(p, 1.-p)
    order = np.lexsort((frame.game_id.astype(str).to_numpy(), confidence))
    groups = np.floor(np.arange(n) * 5 / n).astype(int) + 1
    records = []
    for k in range(1, 6):
        indexes = order[groups == k]
        records.append({'season':season, 'model':model_name, 'probability_rank_group':k,
                        'games':len(indexes), 'average_probability_pct':100*confidence[indexes].mean(),
                        'actual_win_pct':100*chosen_win[indexes].mean(),
                        'wins':int(chosen_win[indexes].sum()),
                        'min_probability_pct':100*confidence[indexes].min(),
                        'max_probability_pct':100*confidence[indexes].max()})
    grouped = pd.DataFrame(records)
    win_rates = grouped.actual_win_pct.to_numpy(float)
    differences = np.diff(win_rates)
    drops = int(np.sum(differences < -1e-9))
    negative_steps = float(-np.minimum(differences,0.).sum())
    # Mann-Whitney equivalent: favorite-win probability AUC.
    ranks = pd.Series(confidence).rank(method='average').to_numpy()
    wins = chosen_win == 1.
    n_wins = int(wins.sum()); n_losses = int((~wins).sum())
    ranking_auc = float((ranks[wins].sum()-n_wins*(n_wins+1)/2)/(n_wins*n_losses)) if n_wins and n_losses else float('nan')
    row = {'season':season, 'model':model_name, 'games':n,
           'probability_group_drops':drops,
           'all_five_groups_ascend':drops == 0,
           'total_downward_win_rate_pp':negative_steps,
           'highest_group_win_pct':float(win_rates[-1]),
           'lowest_group_win_pct':float(win_rates[0]),
           'top_minus_bottom_win_rate_pp':float(win_rates[-1]-win_rates[0]),
           'ranking_auc':ranking_auc,
           'overall_favorite_win_pct':100*chosen_win.mean(),
           'brier':float(np.mean((p-y)**2))}
    return grouped, row


def selection_folds(frame: pd.DataFrame) -> list[tuple[pd.DataFrame,pd.DataFrame]]:
    """Disjoint later-date blocks; every training game precedes its test date."""
    require(set(frame.data_period)=={'2024_2025'},'Wrong model selection season')
    dates = sorted(frame.game_date_dt.unique())
    require(len(dates) > 100,'Insufficient independent validation dates')
    cut = [0, int(len(dates)*.42), int(len(dates)*.60), int(len(dates)*.80),len(dates)]
    folds = []
    for k in range(1,4):
        train = frame.loc[frame.game_date_dt < dates[cut[k]]].copy()
        validation = frame.loc[(frame.game_date_dt >= dates[cut[k]]) &
                               (frame.game_date_dt < (dates[cut[k+1]] if k+1 < 4 else dates[-1] + np.timedelta64(1,'D')))].copy()
        require(len(train)>=250 and len(validation)>=100, f'Insufficient 2024 train/validation rows in fold {k}')
        require(train.game_date_dt.max()<validation.game_date_dt.min(), '2024 date leakage in model selection')
        folds.append((train,validation))
    return folds


def model_fit_predict(train: pd.DataFrame, apply: pd.DataFrame, columns: tuple[str,...], penalty: float) -> np.ndarray:
    tmp = train.copy()
    tmp['target'] = train.favorite_won.to_numpy(float)
    m = RidgeLogit(columns,penalty).fit(tmp)
    return m.predict(apply)


def generate_selection(frame_2024: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    folds = selection_folds(frame_2024)
    experiments = [('unchanged_V8', None, None)] + [
        (f'{g}_ridge_{p:g}',g,p) for g in GROUPS for p in PENALTIES]
    selection_rows = []
    all_chosen = []
    for name, group, penalty in experiments:
        parts=[]
        for train,test in folds:
            q = test[['game_id','game_date_dt','data_period','target','v8_xg_home']].copy()
            q['candidate_home'] = (test.v8_xg_home.to_numpy(float) if group is None else
                to_home_probability(test.v8_xg_home.to_numpy(float),model_fit_predict(train,test,GROUPS[group],penalty)))
            q['model'] = name
            parts.append(q)
        joined=pd.concat(parts,ignore_index=True)
        require(not joined.game_id.duplicated().any(),f'{name}: overlapping model selection folds')
        grouped,stats=group_audit(joined,'candidate_home','2024_2025_CV',name)
        selection_rows.append({'model':name,'feature_group':group,'ridge_penalty':penalty,
                               'cv_training_rule':'each fold trains on earlier 2024 game dates only',**stats})
        all_chosen.append(joined)
    summary=pd.DataFrame(selection_rows)
    # Select for the user's exact requirement: minimum number/size of group
    # win-rate reversals; discriminate better when those measures tie.
    summary['selection_rank'] = 0
    ordered=summary.sort_values(['probability_group_drops','total_downward_win_rate_pp',
                                 'ranking_auc','top_minus_bottom_win_rate_pp','model'],
                                ascending=[True,True,False,False,True]).copy()
    summary.loc[ordered.index,'selection_rank']=np.arange(1,len(ordered)+1)
    selected=ordered.iloc[0].to_dict()
    return summary, selected, pd.concat(all_chosen,ignore_index=True)


def new_predictions(train: pd.DataFrame, test: pd.DataFrame, selected: dict) -> np.ndarray:
    if selected['model']=='unchanged_V8':
        return test.v8_xg_home.to_numpy(float)
    conf = model_fit_predict(train,test,GROUPS[str(selected['feature_group'])],float(selected['ridge_penalty']))
    return to_home_probability(test.v8_xg_home.to_numpy(float),conf)


def load_cached_features() -> tuple[pd.DataFrame,pd.DataFrame]:
    print('READING EXISTING NHL RESULTS AND PRIOR xG CACHES (NO NETWORK)',flush=True)
    hist,slate,_ = previous.load_inputs()
    return hist,slate


def load_saved_slate(date_arg: str | None) -> pd.DataFrame | None:
    if date_arg:
        path=V8_FORECASTS / f'{date_arg}_NHL_my_probability_independent_v8.csv'
        require(path.is_file(),f'No original V8 slate forecast for {date_arg}: {path}')
    else:
        paths=sorted(V8_FORECASTS.glob('*_NHL_my_probability_independent_v8.csv'))
        if not paths:
            return None
        path=paths[-1]
    df=pd.read_csv(path,dtype={'game_id':str})
    required={'game_id','game_date','home_team','away_team','v8_xg_experimental_home'}
    require(required.issubset(df.columns),f'Saved V8 slate missing {required-set(df.columns)}')
    require(not df.game_id.duplicated().any(),'Duplicate V8 slate game ID')
    df['game_date_dt']=pd.to_datetime(df.game_date.astype(str).str.replace('_','-',regex=False))
    require(df.game_date_dt.notna().all() and df.game_date_dt.nunique()==1,'Invalid V8 slate date')
    df['v8_xg_home']=pd.to_numeric(df.v8_xg_experimental_home,errors='coerce')
    require(df.v8_xg_home.between(0,1,inclusive='neither').all(),'Invalid saved V8 slate probabilities')
    return df


def self_test() -> None:
    n=620
    rng=np.random.default_rng(11)
    date_values=pd.date_range('2024-10-01',periods=155,freq='D')
    dates=np.repeat(date_values,4)
    p=rng.uniform(.51,.84,n)
    y=(rng.random(n)<p).astype(float)
    data=pd.DataFrame({'game_id':[f'{2024020000+i:010d}' for i in range(n)],
                       'game_date_dt':dates,'data_period':['2024_2025']*n,
                       'target':y,'v8_xg_home':p})
    feat=data[['game_id']].copy()
    for name in {*FEATURE_SOURCE.values(),'prior_games_min_asof'}:
        feat[name]=rng.normal(size=n)
    featurized=make_features(data,feat)
    require(np.array_equal(featurized.favorite_won,y),'Home favorite winner orientation broken')
    assert np.allclose(to_home_probability(np.array([.6,.4]),np.array([.7,.7])),[.7,.3])
    assert np.allclose(to_home_probability(np.array([.6,.4]),np.array([.3,.3])),[.3,.7])
    folds=selection_folds(featurized)
    for tr,te in folds: assert tr.game_date_dt.max()<te.game_date_dt.min()
    summary,selected,cv=generate_selection(featurized)
    assert len(summary)==7 and len(cv)==sum(len(te) for _,te in folds)*7
    assert set(summary.model).issuperset({'unchanged_V8'})
    quints,metrics=group_audit(data.assign(new=p),'new','2024_2025','synthetic')
    assert len(quints)==5 and quints.games.sum()==n
    # Known five-group win rates: 40%, 50%, 60%, 70%, 80%.
    ordered_outcomes=np.concatenate([(np.arange(100)<k).astype(float) for k in (40,50,60,70,80)])
    perfect=pd.DataFrame({'game_id':[str(i) for i in range(500)],
                          'target':ordered_outcomes,
                          'p':np.linspace(.501,.90,500)})
    ordered_groups,ordered_stats=group_audit(perfect,'p','synthetic','known_ascending')
    assert ordered_stats['all_five_groups_ascend'] and list(np.round(ordered_groups.actual_win_pct))==[40,50,60,70,80]
    inverted=perfect.copy(); inverted['target']=1.-perfect.target
    _,inversion_stats=group_audit(inverted,'p','synthetic','known_reversals')
    assert not inversion_stats['all_five_groups_ascend']
    # Separate deterministic check that ranking-audit uses true orientation.
    reverse=pd.DataFrame({'game_id':[str(i) for i in range(125)],
                          'target':[0.]*125,'p':np.linspace(.501,.85,125)})
    _,v=group_audit(reverse,'p','synthetic','all_wrong')
    assert v['overall_favorite_win_pct']==0.
    print('SELF-TEST PASS: historical feature alignment, winner orientation, chronological folds, direct win-rate selection, full coverage',flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--date',default=None,help='Existing saved V8 slate date YYYY_MM_DD')
    args=parser.parse_args()
    if args.self_test:
        self_test();return

    validation=load_predictions(V8_REPORTS/'2024_ablation_predictions.csv','2024_2025')
    replay=load_predictions(V8_REPORTS/'2025_2027_chronological_replay.csv')
    require(set(replay.data_period)=={'2025_2026','2026_2027'},'Unexpected replay season')
    all_preds=pd.concat([validation,replay],ignore_index=True)
    require(not all_preds.game_id.duplicated().any(),'Duplicated historical prediction game IDs')
    history,slate=load_cached_features()
    hist=make_features(all_preds,history)
    hist.sort_values(['game_date_dt','game_id'],inplace=True)
    period_2024=hist[hist.data_period.eq('2024_2025')].copy()
    selection, chosen, oof=generate_selection(period_2024)
    print('2024 CHRONOLOGICAL SELECTION (ACTUAL WIN-RATE ASCENT):',flush=True)
    print(selection[['model','games','probability_group_drops','total_downward_win_rate_pp',
                     'ranking_auc','selection_rank']].sort_values('selection_rank').to_string(index=False),flush=True)
    print('SELECTED:',chosen['model'],flush=True)

    evaluations=[]; all_quints=[]; all_summaries=[]
    for season, training_seasons in (
        ('2025_2026',['2024_2025']),('2026_2027',['2024_2025','2025_2026'])):
        train=hist.loc[hist.data_period.isin(training_seasons)].copy()
        test=hist.loc[hist.data_period.eq(season)].copy()
        require(len(train)>=900 and len(test)>=200,f'{season}: incomplete existing historical games')
        require(train.game_date_dt.max()<test.game_date_dt.min(),
                f'{season}: training includes same/future-season outcomes')
        candidate=new_predictions(train,test,chosen)
        output=test[['game_id','game_date_dt','data_period','target','v8_xg_home']].copy()
        output['new_home']=candidate
        output['training_seasons']=','.join(training_seasons)
        output['selection_season']='2024_2025'
        evaluations.append(output)
        for label,col in [('existing_V8','v8_xg_home'),('win_rate_candidate','new_home')]:
            group,summary=group_audit(output,col,season,label)
            all_quints.append(group);all_summaries.append(summary)
            print(f'{season} {label}: actual quintile win % = '+
                  ', '.join(f'{x:.1f}' for x in group.actual_win_pct)+
                  f' | downward steps = {summary["probability_group_drops"]} | '
                  f'AUC={summary["ranking_auc"]:.3f}',flush=True)

    scores=pd.DataFrame(all_summaries)
    candidate_scores=scores.loc[scores.model.eq('win_rate_candidate')].set_index('season')
    baseline_scores=scores.loc[scores.model.eq('existing_V8')].set_index('season')
    strictly_ascending=bool(candidate_scores.all_five_groups_ascend.all())
    baseline_ascending=bool(baseline_scores.all_five_groups_ascend.all())
    improved_drops=int(candidate_scores.probability_group_drops.sum()) < int(baseline_scores.probability_group_drops.sum())
    qualifies=bool(strictly_ascending and not baseline_ascending)
    result='MET_ON_BOTH_EVALUATED_SEASONS' if strictly_ascending else 'NOT_MET_ON_BOTH_EVALUATED_SEASONS'
    print('USER GOAL:',result,flush=True)
    print('TOTAL GROUP-TO-GROUP DROPS: existing V8 =',int(baseline_scores.probability_group_drops.sum()),
          '; candidate =',int(candidate_scores.probability_group_drops.sum()),flush=True)

    write_csv(selection,OUT/'2024_chronological_model_selection.csv')
    write_csv(oof,OUT/'2024_model_selection_predictions.csv')
    write_csv(pd.concat(evaluations,ignore_index=True),OUT/'2025_2027_win_rate_candidate_predictions.csv')
    write_csv(pd.concat(all_quints,ignore_index=True),OUT/'2025_2027_actual_win_rates_by_probability.csv')
    write_csv(scores,OUT/'2025_2027_win_rate_progression_summary.csv')

    saved_slate=load_saved_slate(args.date)
    if saved_slate is not None:
        slate_day=saved_slate.game_date_dt.iloc[0]
        matching=slate.loc[slate.game_id.astype(str).isin(saved_slate.game_id)]
        if len(matching)==len(saved_slate) and set(matching.game_id.astype(str))==set(saved_slate.game_id):
            incoming=saved_slate[['game_id','game_date_dt','v8_xg_home']].copy()
            live=make_features(incoming,matching)
            final_train=hist.loc[hist.game_date_dt<slate_day].copy()
            require(len(final_train)>=1000 and final_train.game_date_dt.max()<slate_day,
                    'Slate training date contains same/future game outcomes')
            forecast=saved_slate[['game_id','game_date','home_team','away_team']].copy()
            forecast['existing_v8_home']=live.v8_xg_home.to_numpy(float)
            forecast['my_probability_home']=new_predictions(final_train,live,chosen)
            forecast['my_probability_away']=1.-forecast.my_probability_home
            forecast['model']=chosen['model']
            forecast['user_goal_met_on_both_historical_seasons']=strictly_ascending
            forecast['research_only']=True
            forecast['market_or_provider_forecasts_used_as_features']=False
            write_csv(forecast,OUT/'predictions'/f'{slate_day:%Y_%m_%d}_win_rate_candidate.csv')
        else:
            print('SAVED SLATE: no full match in latest local merged slate; historical test completed, skipped slate forecast',flush=True)

    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),
                'task':'higher predicted probability must correspond to higher observed win percentage',
                'method':'supervised independent confidence re-ranker trained on previous seasons',
                '2024_model_selection':chosen,
                'holdout_result':result,
                'monotone_both_seasons':strictly_ascending,
                'baseline_monotone_both_seasons':baseline_ascending,
                'candidate_reduces_group_drops':improved_drops,
                'candidate_historical_criterion_met':qualifies,
                'no_guarantee_for_unseen_games':True,
                'used_as_features':['V8 independent probabilities',*sorted(set(FEATURE_SOURCE.values())),'prior_games_min_asof'],
                'D_Ratings_used_as_input':False,'SDV_forecast_used_as_input':False,
                'market_odds_used_as_input':False,
                'existing_v8_changed':False,'production_changed':False},
               OUT/'run_manifest.json')

if __name__=='__main__':
    try:
        main()
    except (OSError,ValueError,RuntimeError,KeyError,ImportError,pd.errors.ParserError) as e:
        sys.exit(f'ERROR: {e}')
