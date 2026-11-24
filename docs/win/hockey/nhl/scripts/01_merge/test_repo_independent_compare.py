#!/usr/bin/env python3
"""Daily independent NHL moneyline/puck/total side-by-side.

Authorized for local test-repository execution and explicitly enabled
GitHub Actions execution in Clownworldenjoyer76/hockey.

Reads: local NHL historical results, prior P5 xG, Stage 01 merged, Stage 03
secondary-signal/EV files. Writes ONLY my_probability/test_repo_side_by_side.
Never replaces a pipeline probability, EV, Kelly, bet selection, or source file.
Fixed research candidates: V8 xG (2024-selected xG penalty) + form gate;
ridge-only puck-line (fit_puck_ridge); totals nearest as-of 4-feature,
k=100, inverse-distance weights, prior=2. Totals candidate was picked after
inspecting both retrospective seasons. No promise of future monotonicity.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skellam
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import predict_my_probability_v5_extended as v5
import ablate_my_probability_v8_xg as v8
from predict_my_probability_independent import RidgeLogit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'my_probability' / 'test_repo_side_by_side'
MERGE = ROOT / '01_merge'
STAGE3 = ROOT / '03_edges' / 'secondary_signals'
PERIOD_YEARS = (2022, 2023, 2024, 2025, 2026)
PUCK_FEATURES = ('elo_diff_asof','xg_net_diff_prior_asof','win_rate_last10_diff_asof',
    'goals_diff_last10_diff_asof','rest_diff_asof','prior_games_min_asof',
    'home_puck_line','expected_home_margin_asof','expected_total_goals_asof')
GOAL_FEATURES = ('elo_diff_asof','xg_net_diff_prior_asof','win_rate_last10_diff_asof',
    'goals_diff_last10_diff_asof','rest_diff_asof','prior_games_min_asof',
    'expected_home_margin_asof','expected_total_goals_asof',
    'estimated_home_goals','estimated_away_goals')
TOTAL_FEATURES = ('expected_total_goals_asof','goals_diff_last10_diff_asof',
    'xg_net_diff_prior_asof','prior_games_min_asof')
COLUMNS = {
 'moneyline':('home','away','home_model_prob_moneyline','away_model_prob_moneyline',
               'my_model_home_prob_moneyline','my_model_away_prob_moneyline'),
 'puck_line':('home','away','home_model_prob_puck_line','away_model_prob_puck_line',
               'my_model_home_prob_puck_line','my_model_away_prob_puck_line'),
 'total':('over','under','over_model_prob_total','under_model_prob_total',
               'my_model_over_prob_total','my_model_under_prob_total'),
}


def check(ok, msg):
    if not bool(ok):
        raise ValueError(msg)


def authorize_environment():
    check(
        tuple(part.lower() for part in ROOT.parts[-4:])
        == ('docs', 'win', 'hockey', 'nhl'),
        f'Unexpected NHL repo layout: {ROOT}'
    )

    running_in_github = (
        os.environ.get('GITHUB_ACTIONS', '').lower() == 'true'
    )

    local_test_repo = (
        not running_in_github
        and 'test_repo' in {part.lower() for part in ROOT.parts}
    )

    github_workspace = os.environ.get('GITHUB_WORKSPACE', '')

    authorized_github = (
        running_in_github
        and os.environ.get('GITHUB_REPOSITORY', '').lower()
        == 'clownworldenjoyer76/hockey'
        and os.environ.get('NHL_ALLOW_INDEPENDENT_COMPARE') == '1'
        and bool(github_workspace)
        and ROOT == (
            Path(github_workspace)
            / 'docs'
            / 'win'
            / 'hockey'
            / 'nhl'
        ).resolve()
    )

    check(
        local_test_repo or authorized_github,
        f'SAFETY: unauthorized independent comparison environment: {ROOT}'
    )

    check(
        OUT.resolve() == ROOT / 'my_probability' / 'test_repo_side_by_side',
        f'SAFETY: unexpected comparison output directory: {OUT.resolve()}'
    )

    return 'GITHUB ACTIONS' if authorized_github else 'LOCAL TEST REPO'


def id_normalize(s):
    x=s.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
    check(x.str.fullmatch(r'\d{10}').fillna(False).all(), 'Missing/invalid 10-digit game_id')
    return x.astype(str)


def atomic_csv(f, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.writing')
    f.to_csv(tmp,index=False)
    tmp.replace(path)
    print(f'WROTE {path} | rows={len(f)}',flush=True)


def find_day(explicit=None):
    if explicit:
        check(bool(re.fullmatch(r'\d{4}_\d{2}_\d{2}',explicit)), 'Date must be YYYY_MM_DD')
        return explicit
    options=sorted(STAGE3.glob('????_??_??_NHL_moneyline.csv'))
    check(bool(options), f'Missing Stage 03 moneyline files: {STAGE3}')
    return options[-1].name[:10]


def read_stage(day):
    out={}
    for market in COLUMNS:
        path=STAGE3/f'{day}_NHL_{market}.csv'
        check(path.is_file(), f'Missing Stage 03 {market} output: {path}')
        f=pd.read_csv(path,dtype={'game_id':str},low_memory=False)
        check('game_id' in f, f'{path} has no game_id')
        f['game_id']=id_normalize(f.game_id)
        check(not f.game_id.duplicated().any(),f'Duplicate game_ids in {path}')
        check(len(f)>0, f'Empty Stage 03 {market}: {path}')
        out[market]=f
    ids=set(out['moneyline'].game_id)
    check(all(set(f.game_id)==ids for f in out.values()),
          'Stage 03 moneyline, puck line and totals game IDs differ')
    merged=MERGE/f'{day}_NHL_merged.csv'
    check(merged.is_file(),f'Missing Stage 01 merged day: {merged}')
    f=pd.read_csv(merged,dtype={'game_id':str},low_memory=False)
    check({'game_id','game_date','home_puck_line','total'}.issubset(f),
          f'Incomplete Stage 01 merged schema: {merged}')
    f['game_id']=id_normalize(f.game_id)
    check(not f.game_id.duplicated().any(),'Duplicate Stage 01 game IDs')
    check(set(f.game_id)==ids,'Stage 01 and Stage 03 game IDs do not match')
    check(f.game_date.astype(str).str.replace('-','_',regex=False).eq(day).all(),
          'Stage 01 slate dates do not match requested date')
    return out,f


def requirements(day):
    files=[
        ROOT/'my_probability'/'independent_v8_xg'/'reports'/'selection_provenance.json',
        ROOT/'research'/'sdv_challenger'/'season_2025'/'standalone_comparison.csv',
        ROOT/'season_master'/'2025_2026'/'sportsbook.csv',
        *[ROOT/'my_probability'/'independent_v5'/'official_schedule_cache'/
          f'season_{y}_regular_games.csv' for y in (2022,2023,2024)],
        *[ROOT/'research'/'sdv_history'/'schedules'/f'season_{y}.parquet'
          for y in PERIOD_YEARS],
        *[ROOT/'research'/'sdv_history'/'play_by_play'/f'season_{y}.parquet'
          for y in PERIOD_YEARS],
        ROOT/'scripts'/'research'/'build_sdv_challenger.py',
        MERGE/f'{day}_NHL_merged.csv',
        *[STAGE3/f'{day}_NHL_{m}.csv' for m in COLUMNS]
    ]
    missing=[str(f) for f in files if not f.is_file()]
    check(not missing, 'Independent comparison lacks the following inputs:\n  '+ '\n  '.join(missing))
    return files


def add_goal_rates(history, slate):
    """Prior finalized team goals only, snapshot before each game's date."""
    cols=['game_id','game_date_dt','data_period','home_team_full','away_team_full',
          'home_score','away_score']
    for f in (history,slate):
        check(set(cols).issubset(f),f'Prior goals source missing: {set(cols)-set(f)}')
    games=pd.concat([history[cols],slate[cols]],ignore_index=True)
    games=games.drop_duplicates('game_id',keep='first').sort_values(['game_date_dt','game_id'])
    scores=defaultdict(lambda:deque(maxlen=12))
    league_goals=[]
    season=None
    per_game={}
    for day, group in games.groupby('game_date_dt',sort=True):
        periods=set(group.data_period.astype(str))
        check(len(periods)==1,f'Mixed seasons on {day}')
        new_season=next(iter(periods))
        if season is not None and season!=new_season:
            scores.clear()
        season=new_season
        league_mean=float(np.mean(league_goals)) if league_goals else 3.
        for game in group.itertuples(index=False):
            h=list(scores[game.home_team_full])[-10:]
            a=list(scores[game.away_team_full])[-10:]
            def prior(recent,i):
                return float((sum(t[i] for t in recent)+5*league_mean)/(len(recent)+5))
            hp=0.5*(prior(h,0)+prior(a,1))
            ap=0.5*(prior(a,0)+prior(h,1))
            per_game[str(game.game_id)]=(hp-ap,hp+ap)
        # Add results only after all same-date snapshots have been taken.
        for game in group.itertuples(index=False):
            h,a=pd.to_numeric(game.home_score,errors='coerce'),pd.to_numeric(game.away_score,errors='coerce')
            if pd.notna(h) and pd.notna(a):
                check(h>=0 and a>=0,'Negative NHL final score')
                scores[game.home_team_full].append((float(h),float(a)))
                scores[game.away_team_full].append((float(a),float(h)))
                league_goals.extend((float(h),float(a)))
    def attach(f):
        result=f.copy()
        check(result.game_id.astype(str).isin(per_game).all(),'Missing prior goal calculation')
        v=np.array([per_game[str(g)] for g in result.game_id],dtype=float)
        result['expected_home_margin_asof']=v[:,0]
        result['expected_total_goals_asof']=v[:,1]
        return result
    return attach(history),attach(slate)


def with_estimated_goals(f):
    result=f.copy()
    total=result.expected_total_goals_asof.astype(float)
    margin=result.expected_home_margin_asof.astype(float)
    result['estimated_home_goals']=np.maximum(.1,(total+margin)/2)
    result['estimated_away_goals']=np.maximum(.1,(total-margin)/2)
    return result


def fit_goals(history):
    check(len(history)>2500,'Not enough prior finalized games for goal model')
    models=[]
    x=history[list(GOAL_FEATURES)].to_numpy(float)
    for target in ('home_score','away_score'):
        y=history[target].to_numpy(float)
        check(np.isfinite(y).all() and (y>=0).all(),'Missing/invalid goal targets')
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
            PoissonRegressor(alpha=5.,max_iter=250,tol=1e-6))
        model.fit(x,y)
        models.append(model)
    return models


def goal_puck(models, slate):
    x=slate[list(GOAL_FEATURES)].to_numpy(float)
    h=np.clip(models[0].predict(x),.25,8.)
    a=np.clip(models[1].predict(x),.25,8.)
    line=slate.home_puck_line.to_numpy(float)
    check(np.isin(line,[-1.5,1.5]).all(),'Puck-line model supports only ±1.5')
    return np.clip(np.where(line==-1.5,skellam.sf(1,h,a),skellam.sf(-2,h,a)),1e-6,1-1e-6)


def fit_puck_ridge(history):
    chunks=[]
    for line in (-1.5,1.5):
        f=history.copy()
        f['home_puck_line']=line
        delta=f.home_score.astype(float)-f.away_score.astype(float)+line
        f['target']=(delta>0).astype(float)
        chunks.append(f)
    data=pd.concat(chunks,ignore_index=True)
    check(data.target.nunique()==2,'Puck-line training has only one class')
    return RidgeLogit(PUCK_FEATURES,15.).fit(data)


def nearest_total(history, slate):
    x=history[list(TOTAL_FEATURES)].to_numpy(float)
    y=slate[list(TOTAL_FEATURES)].to_numpy(float)
    imp=SimpleImputer(strategy='median')
    scale=StandardScaler()
    x=scale.fit_transform(imp.fit_transform(x))
    y=scale.transform(imp.transform(y))
    k=100
    check(len(x)>=k,f'Not enough historical games: {len(x)} < {k}')
    near=NearestNeighbors(n_neighbors=k).fit(x)
    distance,neighbors=near.kneighbors(y)
    totals=(history.home_score.to_numpy(float)+history.away_score.to_numpy(float))
    observed=totals[neighbors]
    line=slate.total.to_numpy(float)[:,None]
    weights=1./np.maximum(distance,.1)
    sw=weights.sum(axis=1)
    weighted_over=(weights*(observed>line)).sum(axis=1)/sw*k
    weighted_under=(weights*(observed<line)).sum(axis=1)/sw*k
    global_over=np.array([(totals>t).sum() for t in slate.total.to_numpy(float)],float)
    global_under=np.array([(totals<t).sum() for t in slate.total.to_numpy(float)],float)
    prior=(global_over+1.)/(global_over+global_under+2.)
    prob=(weighted_over+2.*prior)/(weighted_over+weighted_under+2.)
    return np.clip(prob,1e-6,1-1e-6)


def build_independent(day, merged):
    """Generate independent probabilities for all game IDs of one Stage 01 slate."""
    selected=json.loads((ROOT/'my_probability'/'independent_v8_xg'/'reports'/
                         'selection_provenance.json').read_text(encoding='utf-8'))
    check('chosen_xg_penalty_2024' in selected,'Missing locked 2024 V8 xG penalty')
    penalty=float(selected['chosen_xg_penalty_2024'])
    check(np.isfinite(penalty) and penalty>0,'Invalid historical xG penalty')
    historical=pd.concat([v5.load_old_season(None,y,offline=True)
                           for y in (2022,2023,2024)],ignore_index=True)
    history, slate, _=v5.historical_and_current(historical)
    targets=pd.concat([history,slate],ignore_index=True).drop_duplicates('game_id',keep='first')
    challenger=v8.get_challenger()
    xg=[]
    for year in PERIOD_YEARS:
        feature,info=v8.build_one(year,targets,challenger,rebuild=False)
        xg.append(feature[['game_id',v8.XG_FIELD]])
        print(f'xG {year}: {info["filled"]}/{info["games"]}',flush=True)
    xg=pd.concat(xg,ignore_index=True)
    check(not xg.game_id.duplicated().any(),'Duplicate xG prediction IDs')
    history=history.merge(xg,how='left',on='game_id',validate='one_to_one')
    slate=slate.merge(xg,how='left',on='game_id',validate='one_to_one')
    history,slate=add_goal_rates(history,slate)
    history=with_estimated_goals(history)
    slate=with_estimated_goals(slate)
    target=pd.Timestamp(day.replace('_','-'))
    current=slate.loc[slate.game_date_dt.eq(target)].copy()
    current['game_id']=id_normalize(current.game_id)
    merged_ids=set(id_normalize(merged.game_id))
    check(len(current)==len(merged_ids) and set(current.game_id)==merged_ids,
          'Model target slate game IDs differ from Stage 01; no comparison written')
    past=history.loc[history.game_date_dt<target].copy()
    check(len(past)>2500 and past.game_date_dt.max()<target,
          'Forecast training overlaps predicted date')
    # Apply exact recorded contract lines by game ID, never by order or team text.
    contracts=merged[['game_id','home_puck_line','total']].copy()
    contracts['game_id']=id_normalize(contracts.game_id)
    contract_map=contracts.set_index('game_id')
    for name in ('home_puck_line','total'):
        expected=pd.to_numeric(current.game_id.map(contract_map[name]),errors='coerce')
        check(expected.notna().all(),f'Missing {name} line in current Stage 01')
        current[name]=expected.to_numpy(float)
    check(current.home_puck_line.isin([-1.5,1.5]).all(),
          'Unsupported puck-line contract in Stage 01; no guessed probabilities')
    check(current.total.between(3,10).all(),
          'Unsupported totals line in Stage 01; no guessed probabilities')
    current=current.sort_values('game_id').reset_index(drop=True)
    original=RidgeLogit(v8.XG_FEATURES,penalty).fit(past).predict(current)
    check(np.all(np.isfinite(original)) and np.all((original>0)&(original<1)),
          'Invalid xG moneyline output')
    favorite_home=original>=.5
    confidence=np.maximum(original,1-original)
    form_wins=np.where(favorite_home,current.win_rate_last10_diff_asof.to_numpy(float),
                       -current.win_rate_last10_diff_asof.to_numpy(float))
    form_goals=np.where(favorite_home,current.goals_diff_last10_diff_asof.to_numpy(float),
                        -current.goals_diff_last10_diff_asof.to_numpy(float))
    reduce=(confidence>=.70)&np.isfinite(form_wins)&np.isfinite(form_goals)&(form_wins<.15)&(form_goals<1.)
    new_conf=np.where(reduce,.625,confidence)
    ml=np.where(favorite_home,new_conf,1-new_conf)
    original_puck=fit_puck_ridge(past).predict(current)
    puck_goals=goal_puck(fit_goals(past),current)
    puck=original_puck
    total=nearest_total(past,current)
    independent=current[['game_id','game_date','home_team','away_team','home_puck_line','total']].copy()
    independent['independent_home_prob_moneyline']=ml
    independent['independent_away_prob_moneyline']=1-ml
    independent['independent_home_prob_puck_line']=puck
    independent['independent_away_prob_puck_line']=1-puck
    independent['independent_over_prob_total']=total
    independent['independent_under_prob_total']=1-total
    independent['moneyline_form_adjusted']=reduce
    independent['moneyline_v8_unadjusted_home']=original
    independent['model_moneyline']='V8_xG_form_gate_70pct'
    independent['model_puck_line']='original_ridge_only'
    independent['model_total']='nearest_4f_k100_inverse_distance_prior2'
    for first,second in [('independent_home_prob_moneyline','independent_away_prob_moneyline'),
                         ('independent_home_prob_puck_line','independent_away_prob_puck_line'),
                         ('independent_over_prob_total','independent_under_prob_total')]:
        x=independent[[first,second]].to_numpy(float)
        check(np.isfinite(x).all() and (x>0).all() and (x<1).all() and
              np.allclose(x.sum(axis=1),1,atol=1e-9),
              f'Independent probability pair invalid: {first}')
    return independent


def side_by_side(independent, stage):
    """Only reads existing Stage 03. Never modifies its source frames."""
    out=[]
    for market,(left,right,primary_left,primary_right,secondary_left,secondary_right) in COLUMNS.items():
        original=stage[market].copy()
        required={'game_id',primary_left,primary_right,'home_puck_line' if market=='puck_line' else
                  'total' if market=='total' else 'game_date'}
        check(required.issubset(original),
              f'Stage 03 {market} lacks columns: {sorted(required-set(original))}')
        own1=f'independent_{left}_prob_{market}' if market!='total' else 'independent_over_prob_total'
        own2=f'independent_{right}_prob_{market}' if market!='total' else 'independent_under_prob_total'
        column='home_puck_line' if market=='puck_line' else 'total' if market=='total' else None
        selected=independent[['game_id','game_date','home_team','away_team',own1,own2]+([column] if column else [])].copy()
        f=original.merge(selected,on='game_id',how='inner',validate='one_to_one',suffixes=('_stage',''))
        check(len(f)==len(independent),'Comparison game IDs do not match independent forecast')
        if column:
            from_stage=pd.to_numeric(f[column+'_stage'],errors='coerce') if column+'_stage' in f else pd.to_numeric(f[column],errors='coerce')
            from_model=pd.to_numeric(f[column],errors='coerce')
            check(np.isclose(from_stage,from_model,atol=1e-9,rtol=0).all(),
                  f'{market} Stage 03 contract differs from independent contract')
        p1=pd.to_numeric(f[primary_left],errors='coerce')
        p2=pd.to_numeric(f[primary_right],errors='coerce')
        s1=pd.to_numeric(f[secondary_left],errors='coerce') if secondary_left in f else pd.Series(np.nan,index=f.index)
        s2=pd.to_numeric(f[secondary_right],errors='coerce') if secondary_right in f else pd.Series(np.nan,index=f.index)
        result=f[['game_id','game_date','home_team','away_team']].copy()
        result['market']=market
        result['side_1']=left
        result['side_2']=right
        result['contract_line']=f[column].to_numpy(float) if column else np.nan
        result['independent_side_1']=f[own1].to_numpy(float)
        result['independent_side_2']=f[own2].to_numpy(float)
        result['pipeline_primary_side_1']=p1.to_numpy(float)
        result['pipeline_primary_side_2']=p2.to_numpy(float)
        result['pipeline_secondary_side_1']=s1.to_numpy(float)
        result['pipeline_secondary_side_2']=s2.to_numpy(float)
        result['independent_minus_primary_pp']=100*(result.independent_side_1-result.pipeline_primary_side_1)
        result['independent_minus_secondary_pp']=100*(result.independent_side_1-result.pipeline_secondary_side_1)
        result['independent_60plus']=np.maximum(result.independent_side_1,result.independent_side_2)>=.6
        out.append(result)
    final=pd.concat(out,ignore_index=True).sort_values(['game_date','game_id','market'])
    check(len(final)==3*len(independent),'Expected exactly three markets per game')
    check(not final.duplicated(['game_id','market']).any(),'Duplicate game-market comparisons')
    return final


def self_test():
    train=pd.DataFrame({'game_id':[f'{i:010d}' for i in range(201)],
                        'home_score':np.resize([2.,4.,5.,1.],201),
                        'away_score':np.resize([1.,3.,2.,5.],201),
                        'expected_total_goals_asof':np.resize([5.,6.,7.],201),
                        'goals_diff_last10_diff_asof':np.resize([-.2,.1,.4],201),
                        'xg_net_diff_prior_asof':np.resize([-.4,.3,.5],201),
                        'prior_games_min_asof':np.resize([10.,12.,14.],201)})
    slate=train.iloc[:3].copy()
    slate['total']=[5.5,6.5,5.5]
    p=nearest_total(train,slate)
    check(len(p)==3 and np.isfinite(p).all() and ((p>0)&(p<1)).all(),
          'Totals-neighbor probability self-test failed')
    independent=pd.DataFrame({'game_id':['2026020001','2026020002'],
        'game_date':['2026_11_19']*2,'home_team':['BOS','MTL'],'away_team':['NYR','BUF'],
        'home_puck_line':[-1.5,1.5],'total':[5.5,6.5],
        'independent_home_prob_moneyline':[.7,.4],'independent_away_prob_moneyline':[.3,.6],
        'independent_home_prob_puck_line':[.3,.7],'independent_away_prob_puck_line':[.7,.3],
        'independent_over_prob_total':[.45,.65],'independent_under_prob_total':[.55,.35]})
    stage={}
    for market,(left,right,pr1,pr2,sc1,sc2) in COLUMNS.items():
        frame=independent[['game_id','game_date']].copy()
        frame[pr1]=.55;frame[pr2]=.45
        frame[sc1]=.58;frame[sc2]=.42
        if market=='puck_line':frame['home_puck_line']=independent.home_puck_line
        if market=='total':frame['total']=independent.total
        stage[market]=frame
    compare=side_by_side(independent,stage)
    check(len(compare)==6 and set(compare.market)==set(COLUMNS),'Side-by-side self-test failed')
    check(np.allclose(compare.independent_minus_primary_pp,
          100*(compare.independent_side_1-.55)),'Comparison difference calculation wrong')
    altered=stage['moneyline'].copy()
    altered.loc[0,'game_id']='2026020999'
    stage['moneyline']=altered
    try:
        side_by_side(independent,stage)
    except ValueError:
        pass
    else:
        raise AssertionError('Incorrect game ID join was not rejected')
    print('SELF-TEST PASS: 3-market joins, line consistency, source preservation, 60+ fields, totals kNN',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',help='YYYY_MM_DD; defaults latest Stage 03 day')
    parser.add_argument('--preflight',action='store_true',help='List missing independent comparison source files')
    parser.add_argument('--self-test',action='store_true')
    a=parser.parse_args()

    if a.self_test:
        self_test()
        return

    environment=authorize_environment()

    print(
        f'INDEPENDENT COMPARISON AUTHORIZED | '
        f'environment={environment} | root={ROOT}',
        flush=True
    )

    day=find_day(a.date)
    requirements(day)

    print(
        f'{environment} preflight OK | date={day} | all required files exist',
        flush=True
    )

    if a.preflight:
        return

    original,merged=read_stage(day)
    independent=build_independent(day,merged)
    comparison=side_by_side(independent,original)

    atomic_csv(independent,OUT/'predictions'/f'{day}_NHL_independent.csv')
    atomic_csv(comparison,OUT/'comparisons'/f'{day}_NHL_three_market_side_by_side.csv')

    manifest={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
      'repo_root':str(ROOT),
      'execution_environment':environment,
      'date':day,'games':len(independent),
      'compared_market_games':len(comparison),
      'output':'sidecar_only','production_probabilities_changed':False,
      'source_stage03_read_only':True,
      'model_moneyline':'V8 independent xG + 70pct form gate',
      'model_puck':'independent ridge only',
      'model_total':'nearest 4f k100 inverse distance prior2, retrospectively nominated',
      'models_not_coupled_to_D_Ratings_or_SDV_predictions':True,
      'historical_60plus_pass_does_not_guarantee_future_pass':True}

    p=OUT/'manifests'/f'{day}.json'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')

    print('COMPARISON ONLY: original Stage 03 and Stage 04 outputs unchanged.',flush=True)


if __name__=='__main__':
    try:
        main()
    except (ValueError,RuntimeError,OSError,ImportError,KeyError) as exc:
        raise SystemExit(f'ERROR: {exc}')
