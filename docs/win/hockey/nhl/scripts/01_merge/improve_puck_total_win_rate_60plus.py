#!/usr/bin/env python3
"""Test independent goal-count alternatives for NHL puck line and totals.

Reads the existing three-market scored-game CSV, NHL finals/features, and
recorded contract lines. Fits a pair of goal-count models (one per team)
using past completed games only; creates puck-line/total probabilities from
the implied goal distributions. Selects by observed 60%+ band ascent on
2025-26 alone, then reports results on 2026-27. Reports counts and win rates;
never equates an empty band to success. Original probabilities are preserved.

No D-Ratings or SDV provider forecasts, sportsbook odds/prices, network calls,
or new data sources. Recorded puck/total lines only specify the event.
Does not update existing predictions or production bet selection.

Place beside three_market_60plus.py and improve_probability_win_rate_60plus.py.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import three_market_60plus as old

ROOT = old.ROOT
INPUT = ROOT / 'my_probability' / 'independent_research' / 'three_market_60plus' / 'game_level_three_market_probabilities_and_outcomes.csv'
OUT = ROOT / 'my_probability' / 'independent_research' / 'three_market_60plus' / 'goal_count_correction'
FEATURES = ('elo_diff_asof','xg_net_diff_prior_asof','win_rate_last10_diff_asof',
            'goals_diff_last10_diff_asof','rest_diff_asof','prior_games_min_asof',
            'expected_home_margin_asof','expected_total_goals_asof',
            'estimated_home_goals','estimated_away_goals')
BANDS = ((.60,.65,'60-<65%'),(.65,.70,'65-<70%'),(.70,1.0000001,'70%+'))
ALPHAS=(.1,1.,5.)
MIX=(.25,.5,.75)
PERIODS=('2025_2026','2026_2027')


def require(ok, explanation):
    if not bool(ok):
        raise ValueError(explanation)


def write(df, name):
    p=OUT/name
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp')
    df.to_csv(temp,index=False)
    temp.replace(p)
    print(f'WROTE {p} | rows={len(df)}',flush=True)


def with_goal_features(df):
    x=df.copy()
    require(set(FEATURES[:-2]).issubset(x.columns),'Existing pregame features incomplete')
    total=pd.to_numeric(x.expected_total_goals_asof,errors='coerce')
    margin=pd.to_numeric(x.expected_home_margin_asof,errors='coerce')
    x['estimated_home_goals']=np.maximum(.1,(total+margin)/2)
    x['estimated_away_goals']=np.maximum(.1,(total-margin)/2)
    return x


def fit_goal_model(train, alpha):
    require(len(train)>2500,'Insufficient earlier final scores for goal model')
    require(set(FEATURES).issubset(train),'Goal features incomplete')
    x=train[list(FEATURES)].to_numpy(float)
    require(np.isfinite(train[['home_score','away_score']].to_numpy(float)).all(),
            'Goal training includes unfinished games')
    fitted=[]
    for target in ('home_score','away_score'):
        y=train[target].to_numpy(float)
        require(np.all(y>=0),'Negative goal count')
        m=make_pipeline(SimpleImputer(strategy='median',add_indicator=False),
                        StandardScaler(),PoissonRegressor(alpha=alpha,max_iter=250,tol=1e-6))
        m.fit(x,y)
        fitted.append(m)
    return fitted


def goal_prob(models, games, market):
    x=games[list(FEATURES)].to_numpy(float)
    h=np.clip(models[0].predict(x),.25,8.)
    a=np.clip(models[1].predict(x),.25,8.)
    if market=='puck_line':
        line=games.home_puck_line.to_numpy(float)
        require(np.isin(line,[-1.5,1.5]).all(),'Unsupported puck line')
        # Home covers -1.5 when goal margin >=2; covers +1.5 when >=-1.
        p=np.where(line==-1.5, skellam.sf(1,h,a),skellam.sf(-2,h,a))
    else:
        line=games.total.to_numpy(float)
        require(np.isfinite(line).all(),'Missing total line')
        lam=h+a
        p=poisson.sf(np.floor(line).astype(int),lam)
        # Integer totals can push: probabilities conditional on a settled bet.
        integer=np.isclose(line,np.round(line))
        push=np.where(integer,poisson.pmf(np.round(line).astype(int),lam),0.)
        p=p/(1.-push)
    return np.clip(np.asarray(p,float),1e-6,1.-1e-6)


def win_rate_bands(prob,actual):
    prob=np.asarray(prob,float);actual=np.asarray(actual,float)
    require(len(prob)==len(actual) and len(prob)>0,'Empty or misaligned scored games')
    require(np.isfinite(prob).all() and np.all((prob>0)&(prob<1)), 'Bad probabilities')
    require(np.isin(actual,[0,1]).all(),'Missing final outcomes')
    confidence=np.maximum(prob,1.-prob)
    won=np.where(prob>=.5,actual,1.-actual)
    rows=[]
    for low,high,name in BANDS:
        inside=(confidence>=low)&(confidence<high)
        n=int(inside.sum())
        rows.append({'band':name,'games':n,'wins':int(won[inside].sum()),
                     'actual_win_pct':100*won[inside].mean() if n else np.nan,
                     'average_predicted_pct':100*confidence[inside].mean() if n else np.nan})
    rates=np.asarray([x['actual_win_pct'] for x in rows],float)
    complete=bool(np.isfinite(rates).all())
    steps=int(np.sum(np.diff(rates)<=0)) if complete else None
    worst_drop=float(np.maximum(0,-np.diff(rates)).sum()) if complete else None
    return rows, {'bands_populated':complete,
                  'strict_increase_60plus':bool(complete and steps==0),
                  'nonincreasing_steps':steps,'total_downward_pp':worst_drop,
                  'games_at_60plus':int(sum(x['games'] for x in rows)),
                  'band_counts':str([x['games'] for x in rows]),
                  'actual_win_rates_pct':str([float(round(x['actual_win_pct'],2)) if x['games'] else None for x in rows]),
                  'minimum_step_pp':float(np.diff(rates).min()) if complete else np.nan}


def baseline_rows():
    require(INPUT.is_file(),f'Existing scored-game report missing: {INPUT}')
    f=pd.read_csv(INPUT,dtype={'game_id':str},low_memory=False)
    needed={'game_id','data_period','market','model','prob_side1','outcome_side1','line'}
    require(needed.issubset(f),f'Missing scored-game columns: {sorted(needed-set(f))}')
    require(not f.duplicated(['game_id','data_period','market','model']).any(),'Duplicate scored predictions')
    return f


def prepare_targets(history,lines,existing):
    require(set(PERIODS).issubset(set(history.data_period)), 'Missing evaluated seasons')
    require(not history.game_id.astype(str).duplicated().any(),'Duplicate completed games')
    require(not lines.game_id.astype(str).duplicated().any(),'Duplicate game lines')
    merged=old.attach_results(history.loc[history.data_period.isin(PERIODS)],lines)
    outputs={}
    for period in PERIODS:
        for market in ('puck_line','total'):
            col='home_puck_line' if market=='puck_line' else 'total'
            target=merged.loc[merged.data_period.eq(period)&merged[col].notna()].copy()
            if market=='puck_line':
                target=target.loc[target.home_puck_line.isin([-1.5,1.5])].copy()
                diff=target.home_score-target.away_score+target.home_puck_line
            else:
                target=target.loc[target.total.between(3,10)].copy()
                diff=target.home_score+target.away_score-target.total
            target=target.loc[diff.ne(0)].copy()
            target=target.sort_values('game_id').reset_index(drop=True)
            source=existing.loc[(existing.data_period==period)&(existing.market==market)&
                                (existing.model=='independent_'+market)].copy()
            require(not source.game_id.duplicated().any(),f'{period} {market} duplicate baseline')
            source=source.set_index('game_id').reindex(target.game_id.astype(str))
            require(source.prob_side1.notna().all(),f'{period} {market}: baseline game IDs differ')
            observed = ((target.home_score-target.away_score+target.home_puck_line)>0
                        if market=='puck_line' else
                        (target.home_score+target.away_score-target.total)>0)
            require((source.outcome_side1.to_numpy(float)==observed.to_numpy(float)).all(),
                    f'{period} {market}: existing outcomes disagree with reconstructed finals')
            require(np.allclose(source.line.to_numpy(float),target[col].to_numpy(float),atol=1e-9),
                    f'{period} {market}: contract lines differ')
            target['original_probability']=source.prob_side1.to_numpy(float)
            target['observed_side1']=source.outcome_side1.to_numpy(int)
            outputs[period,market]=target
    return outputs


def candidates(baseline, computed):
    out={'unchanged_independent':baseline}
    for alpha,p in computed.items():
        slug=f'goal_count_alpha_{alpha:g}'
        out[slug]=p
        for weight in MIX:
            out[f'blend_{int(round(weight*100))}pct_goals_alpha_{alpha:g}'] = (
                weight*p+(1-weight)*baseline)
    return out


def choose_market(report,market):
    view=report[(report.market==market)&(report.season=='2025_2026')].copy()
    require(not view.empty,f'Missing 2025 candidate selection for {market}')
    # Only user's requirement determines eligibility. No 10-game threshold.
    # Prefer ascending; if none, rank failures and report that no candidate passed.
    view['_invalid']=~view.bands_populated
    view['_bad']=view.nonincreasing_steps.fillna(3).astype(int)
    view['_drop']=view.total_downward_pp.fillna(100.)
    view['_margin']=-view.minimum_step_pp.fillna(-100.)
    view['_coverage']=-view.games_at_60plus
    view['_unchanged']=view.candidate.ne('unchanged_independent')
    view=view.sort_values(['_invalid','_bad','_drop','_margin','_coverage','_unchanged','candidate'],kind='stable')
    return str(view.iloc[0].candidate)


def write_saved_slate(history,slate,lines,winners):
    saved=old.base.load_saved_slate(None)
    if saved is None:
        print('No saved V8 slate available; skipped current-slate output',flush=True)
        return
    ids=set(saved.game_id.astype(str))
    current=slate.loc[slate.game_id.astype(str).isin(ids)].copy()
    require(len(current)==len(ids),'Saved-slate game ID mismatch')
    current=old.attach_saved_slate_lines(current,lines)
    date=current.game_date_dt.iloc[0]
    require(current.game_date_dt.eq(date).all(),'Saved slate contains mixed dates')
    train=with_goal_features(history[history.game_date_dt<date].copy())
    require(len(train)>3000 and train.game_date_dt.max()<date,'Current slate training chronology violation')
    forecasts=[]
    for market in ('puck_line','total'):
        col='home_puck_line' if market=='puck_line' else 'total'
        ready=with_goal_features(current[current[col].notna()].copy())
        if market=='puck_line':ready=ready.loc[ready.home_puck_line.isin([-1.5,1.5])].copy()
        else:ready=ready.loc[ready.total.between(3,10)].copy()
        if ready.empty:continue
        chosen=winners[market]
        base_train=with_goal_features(history[history.data_period.isin(
            ('2022_2023','2023_2024','2024_2025','2025_2026'))].copy())
        orig=old.make_training(base_train,market)
        original=old.RidgeLogit(old.PUCK_FEATURES if market=='puck_line' else old.TOTAL_FEATURES,old.PENALTY).fit(orig).predict(ready)
        if chosen=='unchanged_independent':p=original
        else:
            pieces=chosen.split('_alpha_')
            alpha=float(pieces[-1]);goal=goal_prob(fit_goal_model(train,alpha),ready,market)
            if chosen.startswith('goal_count_'):p=goal
            else:
                import re
                weight=int(re.search(r'blend_(\d+)pct_',chosen).group(1))/100
                p=weight*goal+(1-weight)*original
        f=ready[['game_id','game_date','home_team','away_team']].copy()
        f['market']=market;f['contract_line']=ready[col].to_numpy(float)
        f['model']=chosen;f['prob_side1']=p;f['prob_side2']=1.-p
        f['side1']='home_covers' if market=='puck_line' else 'over'
        f['side2']='away_covers' if market=='puck_line' else 'under'
        f['predicted_side']=np.where(p>=.5,f.side1,f.side2)
        f['predicted_win_pct']=100*np.maximum(p,1-p)
        forecasts.append(f)
    if forecasts:
        result=pd.concat(forecasts,ignore_index=True)
        result['research_only']=True
        write(result,f'predictions/{date.strftime("%Y_%m_%d")}_independent_puck_total_candidate.csv')


def run():
    old.OUT=ROOT/'my_probability'/'independent_research'/'three_market_60plus'
    print('READING EXISTING GAME-LEVEL MARKET RESULTS; NO NETWORK',flush=True)
    existing=baseline_rows()
    history,slate=old.base.load_cached_features()
    history,slate=old.add_prior_goal_rates(history,slate)
    history=with_goal_features(history)
    lines=old.get_lines()
    target=prepare_targets(history,lines,existing)
    scores=[];bands=[];bykey={}
    for season,trained in [('2025_2026',('2022_2023','2023_2024','2024_2025')),
                           ('2026_2027',('2022_2023','2023_2024','2024_2025','2025_2026'))]:
        tr=history[history.data_period.isin(trained)].copy()
        require(tr.game_date_dt.max()<history.loc[history.data_period.eq(season),'game_date_dt'].min(),
                f'{season}: train/evaluation chronology invalid')
        fitted={alpha:fit_goal_model(tr,alpha) for alpha in ALPHAS}
        for market in ('puck_line','total'):
            data=target[season,market]
            orig=data.original_probability.to_numpy(float)
            probs={alpha:goal_prob(m,data,market) for alpha,m in fitted.items()}
            for name,p in candidates(orig,probs).items():
                one,metrics=win_rate_bands(p,data.observed_side1.to_numpy(int))
                metrics.update({'season':season,'market':market,'candidate':name,'scored_games':len(data)})
                scores.append(metrics)
                bands.extend([{'season':season,'market':market,'candidate':name,**v} for v in one])
                bykey[season,market,name]=(data,p)
    report=pd.DataFrame(scores)
    retrospective=(report.pivot(index=['market','candidate'],columns='season',
                                values='strict_increase_60plus').reset_index())
    retrospective['passes_both_evaluated_seasons']=(
        retrospective['2025_2026'] & retrospective['2026_2027'])
    winner={market:choose_market(report,market) for market in ('puck_line','total')}
    selected=report.loc[report.apply(lambda r:r.candidate==winner[r.market],axis=1)].copy()
    selected['selected_using']='2025_2026_only'
    chosen=pd.concat([pd.DataFrame({'game_id':data.game_id,'season':season,'market':market,
                                    'model':winner[market],'line':data['home_puck_line' if market=='puck_line' else 'total'],
                                    'prob_side1':p,'actual_side1':data.observed_side1,
                                    'predicted_win_pct':100*np.maximum(p,1-p)})
                      for (season,market,name),(data,p) in bykey.items() if name==winner[market]],ignore_index=True)
    # Moneyline results from the existing independent experimental form-gate are reported, untouched.
    ml=existing[existing.model.eq('experimental_v8_form_gate')].copy()
    ml_rows=[]
    for season,g in ml.groupby('data_period'):
        if season not in PERIODS: continue
        _,m=win_rate_bands(g.prob_side1,g.outcome_side1)
        ml_rows.append({'season':season,'market':'moneyline','candidate':'experimental_v8_form_gate',**m})
    full_status=pd.concat([selected[['season','market','candidate','bands_populated','strict_increase_60plus',
                                      'band_counts','actual_win_rates_pct','games_at_60plus']],
                           pd.DataFrame(ml_rows)[['season','market','candidate','bands_populated',
                                                  'strict_increase_60plus','band_counts',
                                                  'actual_win_rates_pct','games_at_60plus']]],ignore_index=True)
    full_status['result']=np.where(full_status.strict_increase_60plus,'PASS',
                                   np.where(full_status.bands_populated,'FAIL','NO_GAMES_IN_ONE_OR_MORE_BANDS'))
    write(report,'all_puck_total_candidates.csv')
    write(retrospective,'retrospective_both_seasons_candidates.csv')
    write(pd.DataFrame(bands),'all_puck_total_candidate_bands.csv')
    write(selected,'2025_selected_2026_evaluated.csv')
    write(chosen,'selected_game_level_predictions.csv')
    write(full_status,'all_three_markets_user_goal.csv')
    print('\nUSER REQUIREMENT: 60-<65% < 65-<70% < 70%+ OBSERVED WIN RATE',flush=True)
    print(full_status[['season','market','candidate','band_counts','actual_win_rates_pct','result']].to_string(index=False),flush=True)
    print('2025-selected candidates; 2026 is the separate historical check.',flush=True)
    for market in ('puck_line','total'):
        n=int(retrospective.loc[retrospective.market.eq(market),
                                'passes_both_evaluated_seasons'].sum())
        print(f'{market}: {n} candidates pass BOTH seasons retrospectively (not independent selection)',flush=True)
    write_saved_slate(history,slate,lines,winner)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'success_means':'each 60-<65%, 65-<70%, 70%+ band nonempty and strictly ascending win percentage',
              'user_imposed_minimum_sample_size':False,
              'uses_dratings_or_sdv_predictions':False,'uses_bookmaker_probabilities':False,
              'recorded_lines_define_contracts_only':True,'historical_2025_selection':winner,
              'all_markets_both_seasons_pass':bool(full_status.strict_increase_60plus.all()),
              'production_changed':False,
              'model_limitations':'Experimental: 2025 was already inspected in earlier research; 2026 also previously evaluated.'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('RESULT: '+('PASS' if manifest['all_markets_both_seasons_pass'] else 'NOT YET PASSING'),flush=True)


def self_test():
    # Known Poisson benchmarks: independent GOAL_COUNT contracts, including integer push.
    x=pd.DataFrame({f:[0.] for f in FEATURES})
    x['home_puck_line']=-1.5;x['total']=6.5
    class Constant:
        def __init__(self,x):self.x=x
        def predict(self,f):return np.full(len(f),self.x)
    m=[Constant(3.),Constant(2.)]
    p=goal_prob(m,x,'puck_line')[0]
    require(abs(p-skellam.sf(1,3.,2.))<1e-12,'Puck probability incorrect')
    t=goal_prob(m,x,'total')[0]
    require(abs(t-poisson.sf(6,5.))<1e-12,'Totals probability incorrect')
    x['total']=6.
    push=goal_prob(m,x,'total')[0]
    require(abs(push-poisson.sf(6,5.)/(1-poisson.pmf(6,5.)))<1e-12,
            'Integer total push not excluded')
    a,b=win_rate_bands([.62]*10+[.67]*10+[.73]*10,[0]*3+[1]*7+[0]*2+[1]*8+[0]+[1]*9)
    require(b['strict_increase_60plus'] and b['band_counts']=='[10, 10, 10]',
            'Simple ascending example incorrectly assessed')
    _,b=win_rate_bands([.62,.67],[1,1])
    require(not b['strict_increase_60plus'] and not b['bands_populated'],
            'Missing 70%+ band must not count as success')
    print('SELF-TEST PASS: goals, puck lines, totals, integer pushes, bands, no sample-size threshold')


if __name__=='__main__':
    try:
        ap=argparse.ArgumentParser(description=__doc__)
        ap.add_argument('--self-test',action='store_true')
        arg=ap.parse_args()
        if arg.self_test:self_test()
        else:run()
    except (ValueError,KeyError,FileNotFoundError,RuntimeError,OSError) as e:
        raise SystemExit(f'ERROR: {e}')
