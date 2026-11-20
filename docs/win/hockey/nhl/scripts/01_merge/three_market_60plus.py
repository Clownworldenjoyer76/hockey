#!/usr/bin/env python3
"""Independent NHL 60%+ win-rate analysis for moneyline, puck line and totals.

Reuses local historical results, existing V8 forecasts and pregame-only xG/Elo/form.
Derives NEW, independent puck-line and total probabilities from past final
scores and game strength using binary ridge-logistic models. The sports-book
PUCK LINE and TOTAL numbers define the bet being predicted; bookmaker prices,
bookmaker probabilities, D-Ratings, and SDV forecasts never enter fitting.

2022-24 historical completed games: fit ML-independent market models from
synthetic +/-1.5 puck-line and 4-8.5 total thresholds; 2025-26 is evaluated
using those earlier fits. Fit again with 2025-26 finalized results to evaluate
2026-27. This avoids needing 2024 sportsbook lines to build synthetic labels.

Moneyline results are the actual saved V8 and prior experimental form-gate
results. Puck line and total results are new research models, not V8 outputs.
The form gate was devised after inspecting 2026 outcomes; mark it experimental.

Output is strictly research-only. Existing production/models are not changed.
Requires existing improve_probability_win_rate_60plus.py, V5, V8, and
2025 sportsbook/2026 merged files already in the NHL repository.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from predict_my_probability_independent import RidgeLogit
import improve_probability_win_rate_60plus as base

ROOT = base.ROOT
OUT = ROOT / 'my_probability' / 'independent_research' / 'three_market_60plus'
FORM = ROOT / 'my_probability' / 'independent_research' / 'probability_win_rate_60plus' / 'form_gate' / '2024_2027_game_by_game_form_gate.csv'
BOOK_2025 = ROOT / 'season_master' / '2025_2026' / 'sportsbook.csv'
MERGED = ROOT / '01_merge'
BANDS = [(.60,.65,'60-<65%'),(.65,.70,'65-<70%'),(.70,1.00000001,'70%+')]
PENALTY = 15.
COMMON = ('elo_diff_asof','xg_net_diff_prior_asof','win_rate_last10_diff_asof',
          'goals_diff_last10_diff_asof','rest_diff_asof','prior_games_min_asof')
PUCK_FEATURES = COMMON + ('home_puck_line','expected_home_margin_asof','expected_total_goals_asof')
TOTAL_FEATURES = COMMON + ('total','expected_total_goals_asof','expected_home_margin_asof')
TOTAL_THRESHOLDS = (4.,4.5,5.,5.5,6.,6.5,7.,7.5,8.,8.5)


def require(condition:bool, message:str):
    if not condition: raise ValueError(message)


def save(frame:pd.DataFrame,name:str):
    path=OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending')
    frame.to_csv(temp,index=False)
    temp.replace(path)
    print(f'WROTE {path} | rows={len(frame)}',flush=True)


def get_lines():
    require(BOOK_2025.is_file(),f'Missing existing 2025-26 recorded lines: {BOOK_2025}')
    cols=('game_id','home_puck_line','total')
    prev=pd.read_csv(BOOK_2025,dtype={'game_id':str},low_memory=False)
    require(set(cols).issubset(prev),f'{BOOK_2025} missing {set(cols)-set(prev)}')
    prev=prev[list(cols)].copy();prev['data_period']='2025_2026'
    require(not prev.game_id.duplicated().any(),'Duplicate 2025 game IDs in source lines')
    paths=sorted(MERGED.glob('*_NHL_merged.csv'))
    require(bool(paths),f'No merged 2026 lines in {MERGED}')
    chunks=[]
    for path in paths:
        f=pd.read_csv(path,dtype={'game_id':str},low_memory=False)
        require(set(cols).issubset(f),f'{path} missing required puck/total lines')
        chunks.append(f[list(cols)].copy())
    cur=pd.concat(chunks,ignore_index=True)
    cur=cur.loc[cur.game_id.astype(str).str.match(r'^202602\d{4}$')].copy()
    require(not cur.game_id.duplicated().any(),'Duplicate 2026 game IDs in merged source: stop and inspect')
    cur['data_period']='2026_2027'
    result=pd.concat([prev,cur],ignore_index=True)
    for c in ('home_puck_line','total'):result[c]=pd.to_numeric(result[c],errors='coerce')
    result['game_id']=result.game_id.astype(str)
    require(not result.game_id.duplicated().any(),'Duplicate game IDs across recorded lines')
    return result


def add_prior_goal_rates(history:pd.DataFrame, slate:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    """Same-date forecasts share the same prior team snapshots. No same-date result used."""
    cols=['game_id','game_date_dt','data_period','home_team_full','away_team_full',
          'home_score','away_score']
    require(set(cols).issubset(history),'Historical score/matchup columns unavailable')
    require(set(cols).issubset(slate),'Saved slate columns unavailable')
    all_rows=pd.concat([history[cols],slate[cols]],ignore_index=True).drop_duplicates('game_id')
    require(not all_rows.game_id.duplicated().any(),'Duplicate game IDs in scoring ledger')
    all_rows=all_rows.sort_values(['game_date_dt','game_id'])
    games_by_date={day:g for day,g in all_rows.groupby('game_date_dt',sort=True)}
    rates={}; latest_season=None
    observed_league=[]
    team_scores=defaultdict(lambda:deque(maxlen=12))
    for day, games in games_by_date.items():
        periods=set(games.data_period.astype(str))
        require(len(periods)==1,f'Multiple NHL seasons on {day}')
        period=next(iter(periods))
        if latest_season is not None and period!=latest_season:
            team_scores.clear()
        latest_season=period
        mean_goal=(float(np.mean(observed_league)) if observed_league else 3.)
        for game in games.itertuples(index=False):
            h=list(team_scores[game.home_team_full]); a=list(team_scores[game.away_team_full])
            def rate(recent,index):
                # Shrink small season samples toward scoring rate known before the date.
                recent=recent[-10:]
                return float((sum(r[index] for r in recent)+5*mean_goal)/(len(recent)+5))
            home_for,home_against=rate(h,0),rate(h,1)
            away_for,away_against=rate(a,0),rate(a,1)
            h_goals=.5*(home_for+away_against)
            a_goals=.5*(away_for+home_against)
            rates[str(game.game_id)]=(h_goals-a_goals,h_goals+a_goals)
        for game in games.itertuples(index=False):
            h=pd.to_numeric(game.home_score,errors='coerce')
            a=pd.to_numeric(game.away_score,errors='coerce')
            if pd.notna(h) and pd.notna(a):
                require(float(h)>=0 and float(a)>=0,'Negative final goals')
                team_scores[game.home_team_full].append((float(h),float(a)))
                team_scores[game.away_team_full].append((float(a),float(h)))
                observed_league.extend([float(h),float(a)])
    def attach(f):
        out=f.copy()
        require(out.game_id.astype(str).isin(rates).all(),'Missing prior goal expectation')
        vals=np.array([rates[str(x)] for x in out.game_id],dtype=float)
        out['expected_home_margin_asof']=vals[:,0]
        out['expected_total_goals_asof']=vals[:,1]
        return out
    return attach(history),attach(slate)


def make_training(df,market):
    """Generate labels for defined alternative lines; no historical prices needed."""
    require({'home_score','away_score','game_id',*COMMON,
             'expected_home_margin_asof','expected_total_goals_asof'}.issubset(df),
            f'Missing historical training columns for {market}')
    require(df[['home_score','away_score']].notna().all().all(),
            'Training rows contain unfinished games')
    chunks=[]
    for line in ((-1.5,1.5) if market=='puck_line' else TOTAL_THRESHOLDS):
        block=df.copy()
        if market=='puck_line':
            block['home_puck_line']=line
            diff=block.home_score.astype(float)-block.away_score.astype(float)+line
        else:
            block['total']=line
            diff=block.home_score.astype(float)+block.away_score.astype(float)-line
        block=block.loc[diff.ne(0)].copy() # pushes excluded from training
        block['target']=(diff.loc[block.index]>0).astype(float)
        chunks.append(block)
    out=pd.concat(chunks,ignore_index=True)
    require(out.target.nunique()==2,'Synthetic outcomes did not produce two classes')
    return out


def attach_results(scored:pd.DataFrame,lines:pd.DataFrame):
    df=scored.merge(lines,on=['game_id','data_period'],how='left',validate='one_to_one')
    require(not df.game_id.duplicated().any(),'Duplicate matched games')
    return df


def audit(pred:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    records=[]; summaries=[]
    for (season,market,model),f in pred.groupby(['data_period','market','model'],sort=True):
        p=f.prob_side1.astype(float).to_numpy()
        y=f.outcome_side1.astype(float).to_numpy()
        require(np.isfinite(p).all() and np.all((p>0)&(p<1)),f'Invalid probabilities {season} {market}')
        require(np.isin(y,[0,1]).all(),f'Invalid outcomes {season} {market}')
        conf=np.maximum(p,1-p)
        won=np.where(p>=.5,y,1-y)
        grouped=[]
        for low,high,label in BANDS:
            idx=(conf>=low)&(conf<high)
            n=int(idx.sum()); wins=int(won[idx].sum())
            row={'season':season,'market':market,'model':model,'band':label,'games':n,
                 'wins':wins,'losses':n-wins,'win_pct':100*wins/n if n else np.nan,
                 'avg_assigned_pct':100*np.mean(conf[idx]) if n else np.nan}
            records.append(row);grouped.append(row)
        counts=[r['games'] for r in grouped]
        rates=np.array([r['win_pct'] for r in grouped],float)
        enough=bool(all(n>=10 for n in counts))
        asc=bool(enough and np.all(np.diff(rates)>0))
        summaries.append({'season':season,'market':market,'model':model,'games_with_settled_outcome':len(f),
                          'games_at_60plus':sum(counts),'band_counts':str(counts),
                          'actual_win_rates_pct':str([round(v,2) if np.isfinite(v) else None for v in rates]),
                          'every_band_at_least_10':enough,'strictly_ascending_60plus':asc,
                          'result':'PASS' if asc else ('INSUFFICIENT_10_PER_BAND' if not enough else 'FAIL')})
    return pd.DataFrame(records),pd.DataFrame(summaries)


def format_market(df:pd.DataFrame,market:str,p:np.ndarray):
    require(len(df)==len(p),'Prediction/event row count mismatch')
    data=df[['game_id','game_date_dt','data_period']].copy()
    data['market']=market;data['model']='independent_'+market
    data['prob_side1']=p
    if market=='puck_line':
        data['side1']='home_covers_puck_line'
        data['line']=df.home_puck_line.astype(float).to_numpy()
        diff=df.home_score.astype(float)-df.away_score.astype(float)+df.home_puck_line.astype(float)
    else:
        data['side1']='over'
        data['line']=df.total.astype(float).to_numpy()
        diff=df.home_score.astype(float)+df.away_score.astype(float)-df.total.astype(float)
    data['outcome_side1']=(diff>0).to_numpy(dtype=int)
    require((diff!=0).all(),f'{market} includes push games')
    data['predicted_side']=np.where(p>=.5,data.side1,
                                    'away_covers_puck_line' if market=='puck_line' else 'under')
    data['predicted_win_pct']=100*np.maximum(p,1-p)
    return data


def eval_independent(history,lines):
    completed=history.loc[history.data_period.isin(('2025_2026','2026_2027'))].copy()
    merged=attach_results(completed,lines)
    reports=[]; predicts=[]
    for season, training_periods in [
        ('2025_2026',('2022_2023','2023_2024','2024_2025')),
        ('2026_2027',('2022_2023','2023_2024','2024_2025','2025_2026'))]:
        train=history.loc[history.data_period.isin(training_periods)].copy()
        require(len(train)>3000,f'{season}: insufficient earlier completed NHL games')
        require(train.game_date_dt.max()<completed.loc[completed.data_period.eq(season),'game_date_dt'].min(),
                f'{season}: training data overlaps evaluation season')
        target=merged.loc[merged.data_period.eq(season)].copy()
        for market,line_col,features in [('puck_line','home_puck_line',PUCK_FEATURES),
                                         ('total','total',TOTAL_FEATURES)]:
            fit_df=make_training(train,market)
            print(f'TRAIN {market} -> {season}: earlier games={len(train)} threshold-labeled rows={len(fit_df)}',flush=True)
            model=RidgeLogit(features,PENALTY).fit(fit_df)
            ready=target.loc[target[line_col].notna()].copy()
            if market=='puck_line':
                ready=ready.loc[ready.home_puck_line.isin([-1.5,1.5])]
                diff=ready.home_score-ready.away_score+ready.home_puck_line
            else:
                ready=ready.loc[ready.total.between(3,10)]
                diff=ready.home_score+ready.away_score-ready.total
            before=len(ready)
            ready=ready.loc[diff.ne(0)].copy()
            reports.append({'season':season,'market':market,
                            'scored_games':len(target),'recorded_lines':before,'settled_games':len(ready),
                            'excluded_pushes':before-len(ready),'training_completed_games':len(train),
                            'training_latest_day':str(train.game_date_dt.max().date()),
                            'uses_provider_predictions':False,'uses_bookmaker_probabilities':False})
            if len(ready):
                prob=model.predict(ready)
                predicts.append(format_market(ready,market,prob))
    return pd.concat(predicts,ignore_index=True),pd.DataFrame(reports)


def load_moneyline():
    require(FORM.is_file(),f'Existing moneyline research output missing: {FORM}')
    src=pd.read_csv(FORM,dtype={'game_id':str},low_memory=False)
    cols={'game_id','game_date_dt','data_period','target','v8_xg_home','adjusted_home_probability'}
    require(cols.issubset(src),f'Missing form-gate columns: {cols-set(src)}')
    src=src.loc[src.data_period.isin(['2025_2026','2026_2027'])].copy()
    require(not src.game_id.duplicated().any(),'Duplicate V8 moneyline predictions')
    data=[]
    for col,model in [('v8_xg_home','independent_v8_moneyline'),
                      ('adjusted_home_probability','experimental_v8_form_gate')]:
        a=src[['game_id','game_date_dt','data_period']].copy()
        a['market']='moneyline';a['model']=model;a['line']=np.nan
        a['side1']='home_wins';a['prob_side1']=pd.to_numeric(src[col],errors='coerce').to_numpy(float)
        a['outcome_side1']=pd.to_numeric(src.target,errors='coerce').to_numpy(int)
        a['predicted_side']=np.where(a.prob_side1>=.5,'home_wins','away_wins')
        a['predicted_win_pct']=100*np.maximum(a.prob_side1,1-a.prob_side1)
        data.append(a)
    return pd.concat(data,ignore_index=True)



def attach_saved_slate_lines(current: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Reconcile the existing slate lines with recorded game-ID lines, without suffixes.

    The original slate is already loaded from merged files and can already
    include home_puck_line and total. A plain merge would rename both copies
    to *_x/*_y, causing KeyError in the slate prediction loop.
    """
    required={'game_id','home_puck_line','total'}
    require(required.issubset(lines),
            f'Recorded line data missing {sorted(required-set(lines))}')
    require('game_id' in current,'Current slate missing game_id')
    require(not current.game_id.astype(str).duplicated().any(),
            'Duplicate current slate game IDs')
    require(not lines.game_id.astype(str).duplicated().any(),
            'Duplicate recorded game IDs')
    recorded=lines.set_index(lines.game_id.astype(str))
    out=current.copy()
    for name in ('home_puck_line','total'):
        expected=pd.to_numeric(out.game_id.astype(str).map(recorded[name]), errors='coerce')
        if name in out.columns:
            already=pd.to_numeric(out[name],errors='coerce')
            # Missing/mismatched recorded values must not silently replace
            # available lines on the very same game ID.
            present=already.notna()
            absent_recorded=present & expected.isna()
            require(not absent_recorded.any(),
                    f'Existing slate has {name} but recorded lines do not: '
                    f'{out.loc[absent_recorded,"game_id"].tolist()[:5]}')
            inconsistent=present & expected.notna() & ~np.isclose(
                already.fillna(0).to_numpy(float),
                expected.fillna(0).to_numpy(float),atol=1e-8,rtol=0)
            require(not inconsistent.any(),
                    f'Conflicting {name} for same official game IDs: '
                    f'{out.loc[inconsistent,"game_id"].tolist()[:5]}')
        out[name]=expected.to_numpy(float)
    require(len(out)==len(current),'Recorded line mapping changed slate size')
    return out


def current_saved_slate(history: pd.DataFrame, slate: pd.DataFrame, lines: pd.DataFrame):
    """Issue three-market research predictions for the latest already-saved V8 slate."""
    saved=base.load_saved_slate(None)
    if saved is None:
        print('No saved V8 slate exists; current-slate comparison skipped',flush=True)
        return None
    ids=saved.game_id.astype(str)
    require(not ids.duplicated().any(),'Duplicate V8 saved game IDs')
    current=slate.loc[slate.game_id.astype(str).isin(ids)].copy()
    require(len(current)==len(saved), 'Missing latest V8-slate games in reconstructed team features')
    current=attach_saved_slate_lines(current, lines)
    require(current.game_date_dt.nunique()==1,'Saved slate must be one game date')
    day=current.game_date_dt.iloc[0]
    require(day==saved.game_date_dt.iloc[0],'Saved V8 slate date differs from feature date')
    require(pd.to_datetime(current.game_date.astype(str).str.replace('_','-',regex=False)).eq(day).all(),
            'Saved slate date labels differ from actual feature dates')
    earlier=history.loc[history.data_period.isin(('2022_2023','2023_2024','2024_2025','2025_2026'))].copy()
    require(earlier.game_date_dt.max()<day,'Current slate independent training overlaps target day')
    forecasts=[]
    money=current[['game_id','game_date','home_team','away_team']].copy()
    money['market']='moneyline';money['contract_line']=np.nan
    prior=saved.set_index('game_id').v8_xg_home
    money['prob_side1']=money.game_id.map(prior).astype(float)
    money['side1']='home_wins';money['side2']='away_wins'
    money['prob_side2']=1.-money.prob_side1
    money['model']='existing_v8_independent'
    forecasts.append(money)
    form_gate_file=FORM.parent/'predictions'/'latest_saved_v8_experimental_form_gate.csv'
    if form_gate_file.is_file():
        flag=pd.read_csv(form_gate_file,dtype={'game_id':str})
        if set(ids)==set(flag.game_id.astype(str)) and 'experimental_home_probability' in flag:
            revised=money.copy()
            revised['prob_side1']=revised.game_id.map(flag.set_index('game_id').experimental_home_probability).astype(float)
            require(revised.prob_side1.between(0,1,inclusive='neither').all(),'Invalid saved experimental form-gate probability')
            revised['prob_side2']=1.-revised.prob_side1
            revised['model']='experimental_v8_form_gate'
            forecasts.append(revised)
    for market,line_col,features in [('puck_line','home_puck_line',PUCK_FEATURES),
                                     ('total','total',TOTAL_FEATURES)]:
        eligible=current.loc[current[line_col].notna()].copy()
        if market=='puck_line':
            eligible=eligible.loc[eligible.home_puck_line.isin([-1.5,1.5])]
        else:
            eligible=eligible.loc[eligible.total.between(3,10)]
        if eligible.empty:
            print(f'Saved slate: no {market} games have a supported recorded line',flush=True)
            continue
        training=make_training(earlier,market)
        fit=RidgeLogit(features,PENALTY).fit(training)
        p=fit.predict(eligible)
        f=eligible[['game_id','game_date','home_team','away_team']].copy()
        f['market']=market;f['contract_line']=eligible[line_col].astype(float).to_numpy()
        f['prob_side1']=p;f['prob_side2']=1.-p
        f['side1']='home_covers' if market=='puck_line' else 'over'
        f['side2']='away_covers' if market=='puck_line' else 'under'
        f['model']='independent_'+market
        forecasts.append(f)
    final=pd.concat(forecasts,ignore_index=True)
    require(final.prob_side1.between(0,1,inclusive='neither').all(),
            'Invalid saved slate probability')
    final['predicted_side']=np.where(final.prob_side1>=.5,final.side1,final.side2)
    final['predicted_win_pct']=100*np.maximum(final.prob_side1,final.prob_side2)
    final['research_only']=True
    return final


def self_test():
    rng=np.random.default_rng(7)
    n=450
    f=pd.DataFrame({x:rng.normal(size=n) for x in set(PUCK_FEATURES+TOTAL_FEATURES)})
    f['home_score']=rng.integers(0,7,size=n);f['away_score']=rng.integers(0,7,size=n)
    f['game_id']=[str(i) for i in range(n)]
    f['game_date_dt']=pd.Timestamp('2024-11-01');f['data_period']='2024_2025'
    puck=make_training(f,'puck_line');total=make_training(f,'total')
    require(len(puck)==2*n,'Puck alternate-line expansion incorrect')
    require(len(total)>n*6,'Total alternate-line expansion incomplete')
    p1=RidgeLogit(PUCK_FEATURES,PENALTY).fit(puck).predict(puck.head(4))
    p2=RidgeLogit(TOTAL_FEATURES,PENALTY).fit(total).predict(total.head(4))
    require(np.isfinite(p1).all() and np.isfinite(p2).all(),'Independent market fit invalid')
    ptest=f.head(6).copy();ptest['home_puck_line']=1.5
    require(len(format_market(ptest,'puck_line',np.full(6,.65)))==6,'Puck output wrong')
    ttest=f.head(6).copy();ttest['total']=6.5
    require(len(format_market(ttest,'total',np.full(6,.65)))==6,'Total output wrong')
    rows=[]
    for bandp,wins in [(.62,11),(.67,14),(.73,17)]:
        for i in range(20):rows.append({'game_id':str(len(rows)),'game_date_dt':pd.Timestamp('2026-11-15'),
            'data_period':'2026_2027','market':'total','model':'independent_total',
            'prob_side1':bandp,'outcome_side1':int(i<wins)})
    _,report=audit(pd.DataFrame(rows))
    require(report.strictly_ascending_60plus.iloc[0],'Correct ascending win rates not recognized')
    forbidden=('drat','sdv_','sportsbook','bookmaker','_dk_','market_prob')
    require(not any(any(word in f.lower() for word in forbidden) for f in PUCK_FEATURES+TOTAL_FEATURES),
            'A provider/market-derived predictor slipped into independent features')
    # Regression: a recorded-line lookup must not suffix existing columns.
    source=pd.DataFrame({'game_id':['a','b'],'home_puck_line':[1.5,-1.5],
                         'total':[6.5,5.5]})
    with_existing=source.iloc[:1].copy()
    rec=attach_saved_slate_lines(with_existing,source)
    require(rec['home_puck_line'].iloc[0]==1.5 and rec['total'].iloc[0]==6.5,
            'Existing slate line columns changed unexpectedly')
    require(not any(x.endswith('_x') or x.endswith('_y') for x in rec.columns),
            'Line lookup unexpectedly produced suffix columns')
    without_existing=source[['game_id']].copy()
    rec2=attach_saved_slate_lines(without_existing,source)
    require(rec2[['home_puck_line','total']].equals(source[['home_puck_line','total']]),
            'Line lookup missing values when slate has no line columns')
    wrong=with_existing.copy();wrong['home_puck_line']=0.5
    try:
        attach_saved_slate_lines(wrong,source)
    except ValueError as ex:
        require('Conflicting' in str(ex), 'Unexpected conflicting-line error')
    else:
        raise AssertionError('Conflicting recorded line was silently accepted')
    print('SELF-TEST PASS: line-column collision, exact game-ID matching, conflicting lines rejected, all three markets',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--resume',action='store_true',
                    help='Use the four already-completed historical CSV reports; '
                         'only build the saved-slate forecasts and manifest.')
    args=ap.parse_args()
    if args.self_test:return self_test()
    print('THREE MARKETS: READING LOCAL HISTORY + V8, NO NETWORK',flush=True)
    if args.resume:
        needed=('game_level_three_market_probabilities_and_outcomes.csv',
                'three_market_fixed_60plus_bands.csv',
                'three_market_60plus_progression_summary.csv',
                'puck_total_historical_line_coverage.csv')
        require(all((OUT/f).is_file() for f in needed),
                'Completed historical reports are missing; rerun without --resume')
        all_preds=pd.read_csv(OUT/needed[0],dtype={'game_id':str})
        bands=pd.read_csv(OUT/needed[1])
        summary=pd.read_csv(OUT/needed[2])
        coverage=pd.read_csv(OUT/needed[3])
        require(not all_preds.empty and not bands.empty and not summary.empty,
                'Historical reports are empty')
        require(not all_preds.duplicated(['game_id','market','model']).any(),
                'Duplicate reported game/market/model')
        require(set(summary.market)=={'moneyline','puck_line','total'} and
                set(summary.season)=={'2025_2026','2026_2027'},
                'Historical report does not cover all three markets and both seasons')
        require(set(summary.result).issubset({'PASS','FAIL','INSUFFICIENT_10_PER_BAND'}),
                'Unexpected report status')
        require(set(all_preds.market)=={'moneyline','puck_line','total'},
                'Missing market in historical game-level report')
        print(f'REUSED existing historical results: {len(all_preds)} market-game rows; '
              f'{len(summary)} progression records. No historical refit.',flush=True)
    history,slate=base.load_cached_features()
    history,slate=add_prior_goal_rates(history,slate)
    lines=get_lines()
    if not args.resume:
        models,coverage=eval_independent(history,lines)
        ml=load_moneyline()
        all_preds=pd.concat([ml,models],ignore_index=True)
        require(not all_preds.duplicated(['game_id','market','model']).any(),
                'Duplicate game/market/model in combined predictions')
        bands,summary=audit(all_preds)
        save(all_preds,'game_level_three_market_probabilities_and_outcomes.csv')
        save(bands,'three_market_fixed_60plus_bands.csv')
        save(summary,'three_market_60plus_progression_summary.csv')
        save(coverage,'puck_total_historical_line_coverage.csv')
    latest=current_saved_slate(history,slate,lines)
    if latest is not None:
        dates=latest.game_date.astype(str).unique()
        require(len(dates)==1,'Multiple dates in saved slate output')
        save(latest,f'predictions/{dates[0]}_three_market_independent_research.csv')
    print('\nALL THREE MARKETS — HIGHER PROBABILITY = HIGHER OBSERVED WIN % (60%+):',flush=True)
    print(summary[['season','market','model','band_counts','actual_win_rates_pct','result']].to_string(index=False),flush=True)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'task':'60%+ rising observed win percentage independently for moneyline, puck line, totals',
              'lines_define_contract_only':True,'sportsbook_prices_or_provider_forecasts_used_in_training':False,
              'puck_total_models_new':True,'moneyline_experimental_posthoc_flag':True,
              'training_scheme':'2022-24 completed games to predict 2025; 2022-25 to predict 2026',
              'fixed_penalty':PENALTY,'all_markets_seasons_pass':bool(summary.strictly_ascending_60plus.all()),
              'production_changed':False,'V8_model_changed':False,
              'reused_existing_historical_reports':bool(args.resume),
              'limitations':['Puck/total are new research models, not the existing V8 moneyline model',
                             '2026 evaluated on available completed games only',
                             'Historical 2025 sportsbook lines are existing season_master data',
                             'Moneyline form-gate choice was derived after inspecting 2026 outcomes',
                             '10-game minimum per probability band, or result insufficient']}
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'WROTE {path}',flush=True)
    print('NO MODEL DEPLOYMENT. INDEPENDENT PUCK/TOTAL EXPERIMENTAL; 60+ RESULTS AS SHOWN.',flush=True)


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,FileNotFoundError,RuntimeError,OSError) as exc:
        raise SystemExit(f'ERROR: {exc}')
