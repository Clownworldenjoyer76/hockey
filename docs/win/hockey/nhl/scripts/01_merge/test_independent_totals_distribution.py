#!/usr/bin/env python3
"""NHL totals-only follow-up using existing independent historical inputs.

Tests whether independent forecasts from prior game TOTAL-GOAL distributions
produce higher observed win rates in 60-<65%, 65-<70%, and 70%+ bands.
Compare candidates using 2025-26 only, evaluate the chosen one on 2026-27.
Moneyline, puck-line and production files are not modified. D-Ratings,
SportsDataverse forecasts and sportsbook *prices* are never model inputs.
Existing recorded totals lines define over/under outcomes.

Place beside three_market_60plus.py, improve_probability_win_rate_60plus.py
and improve_puck_total_win_rate_60plus.py.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import improve_puck_total_win_rate_60plus as prior

ROOT=prior.ROOT
OUT=ROOT/'my_probability'/'independent_research'/'three_market_60plus'/'totals_distribution_test'
PERIODS=('2025_2026','2026_2027')
FEATURE_SETS={
    'goal_pace':('expected_total_goals_asof',),
    'pace_form':('expected_total_goals_asof','goals_diff_last10_diff_asof','xg_net_diff_prior_asof','prior_games_min_asof'),
    'full':('expected_total_goals_asof','estimated_home_goals','estimated_away_goals',
            'xg_net_diff_prior_asof','elo_diff_asof','win_rate_last10_diff_asof',
            'goals_diff_last10_diff_asof','rest_diff_asof','prior_games_min_asof'),
}
POISSON_FEATURES=FEATURE_SETS['full']
K_VALUES=(25,50,100,200,400)
SMOOTH=(2.,12.)
NB_SHAPE=(4.,12.,1e7) # 1e7 approaches Poisson
GB_PARAMS=((7,80),(15,80),(15,160))


def require(ok,message):
    if not bool(ok):raise ValueError(message)

def write(df,name):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.pending');df.to_csv(tmp,index=False);tmp.replace(p)
    print(f'WROTE {p} | rows={len(df)}',flush=True)

def rate_bands(p,y):
    return prior.win_rate_bands(p,y)

def cdf_total(mean,line,shape):
    mean=np.clip(np.asarray(mean,float),.1,12.)
    line=np.asarray(line,float)
    require(len(mean)==len(line) and np.isfinite(line).all(),'Invalid totals lines')
    z=np.floor(line).astype(int)
    integers=np.isclose(line,np.round(line))
    if shape>=1e6:
        over=poisson.sf(z,mean)
        push=np.where(integers,poisson.pmf(np.round(line).astype(int),mean),0.)
    else:
        pp=shape/(shape+mean)
        over=nbinom.sf(z,shape,pp)
        push=np.where(integers,nbinom.pmf(np.round(line).astype(int),shape,pp),0.)
    require(np.all(1-push>0),'Invalid push normalization')
    return np.clip(over/(1-push),1e-6,1-1e-6)

def nearest_candidates(train,test,features):
    require(set(features).issubset(train.columns) and set(features).issubset(test.columns),
            f'Missing prior-feature columns: {features}')
    imp=SimpleImputer(strategy='median')
    scale=StandardScaler()
    tr=scale.fit_transform(imp.fit_transform(train[list(features)].to_numpy(float)))
    te=scale.transform(imp.transform(test[list(features)].to_numpy(float)))
    kmax=min(max(K_VALUES),len(train))
    nn=NearestNeighbors(n_neighbors=kmax,algorithm='auto').fit(tr)
    dist,ids=nn.kneighbors(te)
    prior_goals=(train.home_score.to_numpy(float)+train.away_score.to_numpy(float))
    neighbors=prior_goals[ids]
    line=test.total.to_numpy(float)[:,None]
    overs=(neighbors>line).astype(float)
    unders=(neighbors<line).astype(float)
    # Game-by-game baseline from training outcomes ONLY, at each target line.
    line1d=test.total.to_numpy(float)
    over_global=np.array([(prior_goals>threshold).sum() for threshold in line1d],float)
    under_global=np.array([(prior_goals<threshold).sum() for threshold in line1d],float)
    global_p=(over_global+1.)/(over_global+under_global+2.)
    out={}
    for k in K_VALUES:
        width=min(k,kmax)
        for weight_kind in ('uniform','distance'):
            if weight_kind=='uniform':weights=np.ones((len(test),width),float)
            else:weights=1./np.maximum(dist[:,:width],.1)
            sw=weights.sum(axis=1)
            so=(weights*overs[:,:width]).sum(axis=1)
            su=(weights*unders[:,:width]).sum(axis=1)
            # Scale weights so the effective neighborhood size remains width.
            so=so/sw*width;su=su/sw*width
            for smoothing in SMOOTH:
                p=(so+smoothing*global_p)/(so+su+smoothing)
                out[f'nearest_{features[0]}_{len(features)}f_k{k}_{weight_kind}_prior{int(smoothing)}']=np.clip(p,1e-6,1-1e-6)
    return out

def boost_candidates(train,test):
    feats=list(POISSON_FEATURES)
    require(set(feats).issubset(train.columns) and set(feats).issubset(test.columns),
            'Expected prior-game feature missing')
    imp=SimpleImputer(strategy='median')
    xtrain=imp.fit_transform(train[feats].to_numpy(float))
    xtest=imp.transform(test[feats].to_numpy(float))
    y=train.home_score.to_numpy(float)+train.away_score.to_numpy(float)
    out={}
    for leaves,minleaf in GB_PARAMS:
        model=HistGradientBoostingRegressor(loss='poisson',learning_rate=.045,max_iter=170,
                     max_leaf_nodes=leaves,min_samples_leaf=minleaf,l2_regularization=8.,random_state=17)
        model.fit(xtrain,y)
        mean=model.predict(xtest)
        for shape in NB_SHAPE:
            slug='poisson' if shape>=1e6 else f'nb{int(shape)}'
            out[f'boost_leaves{leaves}_leaf{minleaf}_{slug}']=cdf_total(mean,test.total,shape)
    return out

def calculate_candidates(train,test,existing_p=None):
    require(not train.empty and not test.empty,'Missing training or scored games')
    require(np.isfinite(train[['home_score','away_score']].to_numpy(float)).all(),
            'Historical training contains unfinished games')
    require(pd.Timestamp(train.game_date_dt.max())<pd.Timestamp(test.game_date_dt.min()),
            'Training must finish before the evaluated season')
    results={}
    if existing_p is not None:results['unchanged_independent_total']=np.asarray(existing_p,float)
    for fs in FEATURE_SETS.values():
        results.update(nearest_candidates(train,test,fs))
    results.update(boost_candidates(train,test))
    return results

def choose(report):
    train=report.query('season == "2025_2026"').copy()
    require(not train.empty,'No 2025 selection records')
    train['_fail']=~train.strict_increase_60plus
    train['_unfilled']=~train.bands_populated
    train['_down']=train.nonincreasing_steps.fillna(3)
    train['_magnitude']=train.total_downward_pp.fillna(1e4)
    train['_coverage']=-train.games_at_60plus
    train=train.sort_values(['_fail','_unfilled','_down','_magnitude','_coverage','candidate'])
    return str(train.iloc[0].candidate)

def run():
    print('TOTALS ONLY: historical NHL goals and recorded contract lines; NO NETWORK',flush=True)
    existing=prior.baseline_rows()
    history,slate=prior.old.base.load_cached_features()
    history,slate=prior.old.add_prior_goal_rates(history,slate)
    history=prior.with_goal_features(history)
    line_data=prior.old.get_lines()
    targets=prior.prepare_targets(history,line_data,existing)
    rows=[];band_rows=[];forecasts={}
    for season,earlier in [('2025_2026',('2022_2023','2023_2024','2024_2025')),
                           ('2026_2027',('2022_2023','2023_2024','2024_2025','2025_2026'))]:
        tr=history.loc[history.data_period.isin(earlier)].copy()
        data=targets[season,'total'].copy()
        models=calculate_candidates(tr,data,data.original_probability)
        y=data.observed_side1.to_numpy(int)
        for name,p in models.items():
            bands,summary=rate_bands(p,y)
            rows.append({'season':season,'market':'total','candidate':name,'settled_games':len(data),**summary})
            band_rows.extend([{'season':season,'candidate':name,**b} for b in bands])
            forecasts[season,name]=(data,p)
        print(f'{season}: evaluated {len(models)} independent total candidates on {len(data)} settled games',flush=True)
    report=pd.DataFrame(rows)
    winner=choose(report)
    selected=report.loc[report.candidate.eq(winner)].copy()
    require(len(selected)==2,'Selection coverage error')
    selected_games=[]
    for season in PERIODS:
        data,p=forecasts[season,winner]
        selected_games.append(pd.DataFrame({'game_id':data.game_id,'season':season,
            'contract_total':data.total,'independent_over_probability':p,
            'independent_under_probability':1-p,
            'predicted_side':np.where(p>=.5,'over','under'),
            'predicted_win_probability':np.maximum(p,1-p),
            'predicted_side_won':np.where(p>=.5,data.observed_side1,1-data.observed_side1)}))
    write(report,'all_totals_models_60plus.csv')
    write(pd.DataFrame(band_rows),'all_totals_models_probability_bands.csv')
    write(selected,'2025_selected_2026_totals_results.csv')
    write(pd.concat(selected_games,ignore_index=True),'selected_game_level_totals.csv')
    print('\nTOTALS — SELECTED USING 2025-26 ONLY:',flush=True)
    print(selected[['season','candidate','band_counts','actual_win_rates_pct',
                    'strict_increase_60plus']].to_string(index=False),flush=True)
    both=int(report.pivot(index='candidate',columns='season',values='strict_increase_60plus')
             .pipe(lambda p:(p['2025_2026']&p['2026_2027']).sum()))
    print(f'All tested totals candidates passing BOTH seasons retrospectively: {both}',flush=True)
    # Forecast saved target slate without changing another market's predictions.
    saved=prior.old.base.load_saved_slate(None)
    if saved is not None:
        ids=set(saved.game_id.astype(str))
        current=slate.loc[slate.game_id.astype(str).isin(ids)].copy()
        require(len(current)==len(ids),'Saved-slate game ID mismatch')
        current=prior.old.attach_saved_slate_lines(current,line_data)
        current=prior.with_goal_features(current)
        current=current.loc[current.total.between(3,10)].copy()
        if not current.empty:
            date=current.game_date_dt.iloc[0]
            require(current.game_date_dt.eq(date).all(),'Saved slate contains multiple dates')
            tr=history.loc[history.game_date_dt<date].copy()
            # Baseline is only needed if selected; other candidate families use prior goals.
            base_p=None
            if winner=='unchanged_independent_total':
                tf=prior.old.make_training(tr,'total')
                base_p=prior.old.RidgeLogit(prior.old.TOTAL_FEATURES,prior.old.PENALTY).fit(tf).predict(current)
            p=calculate_candidates(tr,current,base_p)[winner]
            slate_rows=current[['game_id','game_date','home_team','away_team','total']].copy()
            slate_rows['market']='total';slate_rows['model']=winner
            slate_rows['independent_over_probability']=p
            slate_rows['independent_under_probability']=1-p
            slate_rows['predicted_side']=np.where(p>=.5,'over','under')
            slate_rows['predicted_win_probability']=np.maximum(p,1-p)
            slate_rows['research_only']=True
            write(slate_rows,f'predictions/{pd.Timestamp(date).strftime("%Y_%m_%d")}_independent_totals.csv')
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'selection_period':'2025_2026','evaluation_period':'2026_2027',
      'selected_model':winner,'strictly_ascending_both_seasons':bool(selected.strict_increase_60plus.all()),
      'count_retroactive_both_pass':both,
      'moneyline_or_puck_line_changed':False,'production_changed':False,
      'uses_sdv_or_dratings_predictions':False,'uses_bookmaker_prices':False,
      'recorded_totals_lines_only_define_event':True}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('TOTALS RESULT: '+('PASS' if manifest['strictly_ascending_both_seasons'] else 'FAIL')+
          ' | existing moneyline and puck-line untouched',flush=True)

def self_test():
    mu=np.array([5.8,6.2]);lines=np.array([5.5,6.0])
    p=cdf_total(mu,lines,1e7)
    require(np.isclose(p[0],poisson.sf(5,5.8)),'Half-goal over probability incorrect')
    require(np.isclose(p[1],poisson.sf(6,6.2)/(1-poisson.pmf(6,6.2))),
            'Integer total push treatment incorrect')
    require(np.all((cdf_total(mu,lines,4.)>0)&(cdf_total(mu,lines,4.)<1)),
            'Negative-binomial probability out of range')
    rng=np.random.default_rng(71)
    n=850
    a=pd.DataFrame({'home_score':rng.poisson(3.,n),'away_score':rng.poisson(3.1,n),
                    'game_date_dt':pd.Timestamp('2024-05-01')})
    b=pd.DataFrame({'home_score':rng.poisson(3.,55),'away_score':rng.poisson(3.1,55),
                    'game_date_dt':pd.Timestamp('2025-05-01'),'total':rng.choice([5.5,6.5],55)})
    for f in set(POISSON_FEATURES):
        a[f]=rng.normal(size=n)+(.2 if f=='expected_total_goals_asof' else 0)
        b[f]=rng.normal(size=len(b))
    preds=calculate_candidates(a,b)
    require(len(preds)>=45,'Missing model candidates')
    for name,p in preds.items():
        require(len(p)==len(b) and np.isfinite(p).all() and np.all((p>0)&(p<1)),
                f'Invalid forecast: {name}')
    rr=[]
    for name,p in preds.items():
        _,v=rate_bands(p,rng.integers(0,2,len(p)))
        rr.append({'season':'2025_2026','candidate':name,**v})
    require(choose(pd.DataFrame(rr)) in preds,'Selection failed')
    print(f'SELF-TEST PASS: {len(preds)} totals candidates, push treatment, historical-only training')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    try:self_test() if args.self_test else run()
    except (ValueError,KeyError,FileNotFoundError,RuntimeError,OSError) as exc:
        raise SystemExit(f'ERROR: {exc}')
