#!/usr/bin/env python3
"""Independent V8 historical xG ablation, never changes production or V5.

Requires previously built 2022--2026 P5 schedules and play_by_play files.
Uses repository SportsDataverse team_game_xg_rates and prior_adjusted_ratings;
features for target day D use ONLY PBP from games dated strictly before D.

2022-23 and 2023-24: train. 2024-25: choose xG penalty and gate adoption.
2025-26 and 2026-27: chronological historical replay (diagnostic ONLY;
these periods were repeatedly examined during earlier V1--V7 research).

Odds are never included in features/selection. Missing xG coverage is fatal.
Goaltender/lineup retrospective placeholders are never used.

Usage: python ablate_my_probability_v8_xg.py [--date YYYY_MM_DD] [--self-test]
All files written under NHL/my_probability/independent_v8_xg/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import predict_my_probability_v5_extended as v5
from predict_my_probability_independent import RidgeLogit, brier, log_loss

ROOT = v5.NHL
OUT = ROOT / 'my_probability' / 'independent_v8_xg'
CACHE = OUT / 'xg_feature_cache'
P5 = ROOT / 'research' / 'sdv_history'
YEARS = (2022, 2023, 2024, 2025, 2026)
BASE_MODEL = 'elo_form'
BASE_PENALTY = 15.0
BASE = v5.FEATURE_SETS[BASE_MODEL]
XG_FIELD = 'xg_net_diff_prior_asof'
XG_FEATURES = BASE + (XG_FIELD,)
PENALTIES = (15.0, 60.0, 200.0)
MIN_XG_COVERAGE = .75
MIN_VALIDATION_IMPROVEMENT = .0004
VERSION = 'independent_v8_xg_ablation_20261120_1'


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    frame.to_csv(temp, index=False)
    temp.replace(path)
    print(f'WROTE {path} | rows={len(frame)}', flush=True)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding='utf-8')
    temp.replace(path)


def marker(path):
    stat = path.stat()
    return {'path':str(path), 'bytes':stat.st_size, 'modified_ns':stat.st_mtime_ns}


def key_hash(targets):
    values = targets[['game_id','game_date_dt','home_team_full','away_team_full']].copy()
    values['game_date_dt'] = values.game_date_dt.dt.strftime('%Y-%m-%d')
    values = values.sort_values('game_id')
    return hashlib.sha256(values.to_csv(index=False).encode()).hexdigest()


def get_challenger():
    research = ROOT / 'scripts' / 'research'
    check(research.is_dir(), f'Repository SDV research scripts missing: {research}')
    if str(research) not in sys.path:
        sys.path.insert(0,str(research))
    try:
        return importlib.import_module('build_sdv_challenger')
    except Exception as e:
        raise RuntimeError(f'Cannot import SDV P5 xG builder: {type(e).__name__}: {e}') from e


def validate_asof(rows):
    """Validate exported source date with actual calendar date for each game."""
    check(not rows.game_id.duplicated().any(), 'Duplicate xG feature game_id')
    valid = rows[XG_FIELD].notna()
    src = pd.to_datetime(rows.loc[valid,'xg_max_source_game_date'],errors='coerce')
    check(src.notna().all(), 'Non-null xG value without a valid historical source date')
    check((src.values < rows.loc[valid,'game_date_dt'].values).all(),
          'LEAKAGE: xG source_game_date is not earlier than predicted game date')
    check(np.isfinite(pd.to_numeric(rows.loc[valid,XG_FIELD])).all(),
          'Nonfinite xG differential')
    if (~valid).any():
        check(rows.loc[~valid,'xg_max_source_game_date'].isna().all(),
              'Missing xG cannot have a falsely attributed source date')


def source_consistency(schedule, targets, year):
    """Exact official-ID/date/team matches; never infer identities from names."""
    required = {'game_id','date','home_abbr','away_abbr'}
    check(required.issubset(schedule.columns), f'SDV {year} normalized schedule missing {required-set(schedule.columns)}')
    ref = schedule.select('game_id','date','home_abbr','away_abbr').to_pandas()
    ref['game_id'] = ref.game_id.astype(str)
    check(not ref.game_id.duplicated().any(),f'SDV {year} schedule duplicate game IDs')
    tgt = targets[['game_id','game_date_dt','home_team_full','away_team_full']].copy()
    tgt['game_id'] = tgt.game_id.astype(str)
    joined = tgt.merge(ref, how='left',on='game_id',validate='one_to_one')
    found = joined.date.notna()
    check(found.mean() >= .90, f'SDV {year} schedule only matches {found.mean():.1%} of target IDs')
    if found.any():
        observed = pd.to_datetime(joined.loc[found,'date'])
        canonical = joined.loc[found,'game_date_dt']
        bad_day = observed.ne(canonical)
        check(not bad_day.any(), f'SDV {year} dates disagree with official final games: {joined.loc[found].loc[bad_day,"game_id"].head(3).tolist()}')
        # Some historical source abbreviations may differ from the official API
        # (for example relocation). Reject ambiguity rather than fuzzy joining.
        for side in ('home','away'):
            source = joined.loc[found,side+'_abbr'].astype(str).str.upper()
            actual = joined.loc[found,side+'_team_full'].astype(str).str.upper()
            different = source.ne(actual)
            check(not different.any(), f'SDV {year} {side} team identity mismatch: '+
                  str(joined.loc[found].loc[different,['game_id',side+'_abbr',side+'_team_full']].head(4).to_dict('records')))
    return int(found.sum()),len(joined)


def build_one(year, targets, challenger, rebuild=False):
    season = f'{year}_{year+1}'
    target = targets.loc[targets.data_period.eq(season)].copy()
    check(not target.empty, f'No target games for {season}')
    for col in ('game_id','game_date_dt','home_team_full','away_team_full'):
        check(col in target, f'Missing V5 target field {col}')
    target['game_id'] = target.game_id.astype(str)
    target = target.sort_values(['game_date_dt','game_id']).reset_index(drop=True)
    schedule_path = P5/'schedules'/f'season_{year}.parquet'
    pbp_path = P5/'play_by_play'/f'season_{year}.parquet'
    for f in (schedule_path,pbp_path):
        check(f.is_file(), f'Missing local P5 input (run build_sdv_history.py): {f}')
    fingerprints = {'version':VERSION,'season':season,'targets_sha256':key_hash(target),
                    'sources':[marker(schedule_path),marker(pbp_path)]}
    cache_path = CACHE/f'season_{year}_prior_xg.csv'
    manifest_path = CACHE/f'season_{year}_prior_xg.manifest.json'
    if cache_path.is_file() and manifest_path.is_file() and not rebuild:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest == fingerprints:
            old = pd.read_csv(cache_path, dtype={'game_id':str})
            needed = {'game_id','game_date_dt','xg_max_source_game_date',XG_FIELD}
            check(needed.issubset(old),f'Cached {season} xG is missing columns')
            old['game_date_dt'] = pd.to_datetime(old.game_date_dt)
            check(set(old.game_id)==set(target.game_id) and len(old)==len(target),
                  f'Cached {season} xG does not cover exactly target game IDs')
            validate_asof(old)
            print(f'CACHE HIT xG {season}: {len(old)} games',flush=True)
            return old,{'season':season,'games':len(old),'filled':int(old[XG_FIELD].notna().sum()),
                        'coverage':float(old[XG_FIELD].notna().mean()),'source':'validated_xg_cache'}
    print(f'BUILDING STRICT PRIOR xG {season}: local schedules/PBP; no network',flush=True)
    schedule, game_rates = challenger.load_p5_game_rates(year)
    found, total = source_consistency(schedule,target,year)
    check('date' in game_rates.columns, f'SDV game rates {year} missing date')
    # Precompute max source game dates; no same-day event can enter a feature.
    rates_dates = game_rates.get_column('date').to_list()
    check(rates_dates, f'SDV {year} game rates empty')
    check(all(isinstance(x,date) for x in rates_dates if x is not None),
          f'SDV {year} game rates have unparseable source dates')
    dates = sorted(set(x for x in rates_dates if x is not None))
    prior_day_latest={}
    cursor=0
    newest=None
    for d in sorted(target.game_date_dt.dt.date.unique()):
        while cursor<len(dates) and dates[cursor]<d:
            newest=dates[cursor];cursor+=1
        prior_day_latest[d]=newest
    records=[]
    n_days=target.game_date_dt.nunique()
    for i,(day,group) in enumerate(target.groupby('game_date_dt',sort=True),1):
        historical_day = prior_day_latest[day.date()]
        index={}
        if historical_day is not None:
            adjusted=challenger.ratings_from_prior_games(game_rates,day.date())
            if not adjusted.is_empty():
                rating = adjusted.select(['team','adj_xg_net']).to_pandas()
                check(not rating.team.astype(str).duplicated().any(),
                      f'SDV {season} duplicate team ratings on {day:%Y-%m-%d}')
                for r in rating.itertuples(index=False):
                    x=float(r.adj_xg_net)
                    if np.isfinite(x):index[str(r.team).strip().upper()]=x
        for r in group.itertuples(index=False):
            h,a=r.home_team_full.upper(),r.away_team_full.upper()
            if h in index and a in index:
                xg=float(index[h]-index[a]); max_source=historical_day.isoformat()
            else:
                xg=float('nan');max_source=None
            records.append({'game_id':r.game_id,'game_date_dt':r.game_date_dt,
                            XG_FIELD:xg,'xg_max_source_game_date':max_source,
                            'xg_asof_rule':'source_game_date < target_game_date'})
        if i%45==0:
            print(f'xG {season}: {i}/{n_days} target dates',flush=True)
    out=pd.DataFrame(records)
    check(len(out)==len(target) and set(out.game_id)==set(target.game_id),
          f'{season} xG cardinality mismatch')
    validate_asof(out)
    coverage=float(out[XG_FIELD].notna().mean())
    print(f'xG {season}: matched {found}/{total} schedule IDs; '+
          f'features {out[XG_FIELD].notna().sum()}/{len(out)} ({coverage:.1%})',flush=True)
    # Reject entire-season missingness; do not let missingness flags encode the year.
    check(coverage >= MIN_XG_COVERAGE,
          f'{season}: xG coverage {coverage:.1%} below {MIN_XG_COVERAGE:.0%}; cannot compare xG without a season availability confound')
    save_csv(out,cache_path)
    save_json(fingerprints,manifest_path)
    return out,{'season':season,'games':len(out),'filled':int(out[XG_FIELD].notna().sum()),
                'coverage':coverage,'matched_schedule_games':found,
                'source':'prior_adjusted_ratings_from_local_P5'}


def date_block_interval(group, diff_col, n=2500, seed=20261120):
    check(not group.empty,'Empty data for date-block bootstrap')
    vals=group[["game_date_dt",diff_col]].reset_index(drop=True)
    days=[g[diff_col].to_numpy(float) for _,g in vals.groupby('game_date_dt',sort=True)]
    check(len(days)>=20,'Too few distinct validation dates for bootstrap')
    rng=np.random.default_rng(seed)
    means=np.empty(n,float)
    for i in range(n):
        chosen=rng.integers(0,len(days),len(days))
        combined=np.concatenate([days[k] for k in chosen])
        means[i]=combined.mean()
    return tuple(float(x) for x in np.quantile(means,[.025,.975]))


def scores(group, name, col):
    data=group.loc[group[col].notna()]
    if data.empty:return None
    return {'season':name, 'model':col,'games':len(data),
            'brier':float(brier(data.target,data[col])),
            'log_loss':float(log_loss(data.target,data[col]))}


def compare_historical(pred, market):
    out=pred.merge(market[['game_id','vig_free_home_prob','drat_home_win_prob','sdv_home_win_prob']].drop_duplicates('game_id'),
                   how='left',on='game_id',validate='one_to_one')
    summaries=[]
    for season,g in out.groupby('data_period',sort=True):
        for name,col in [('v5_baseline','v5_base_home'),('xg_experiment','v8_xg_home'),
                         ('vig_free_market','vig_free_home_prob'),('D_Ratings','drat_home_win_prob'),
                         ('SportsDataverse','sdv_home_win_prob')]:
            row=scores(g,season,col)
            if row:
                row['model']=name
                row['selection_period']='2024_2025_only'
                row['interpretation']='historical_replay_exploratory_not_untouched_holdout'
                summaries.append(row)
        diff=g.copy()
        diff['xg_minus_v5']=(diff.v8_xg_home-diff.target)**2-(diff.v5_base_home-diff.target)**2
        lo,hi=date_block_interval(diff,'xg_minus_v5',n=1200)
        summaries.append({'season':season,'model':'xg_minus_v5','games':len(g),
                          'brier':float(diff.xg_minus_v5.mean()),'paired_95ci_low':lo,'paired_95ci_high':hi,
                          'selection_period':'2024_2025_only',
                          'interpretation':'historical_replay_exploratory_not_untouched_holdout'})
    return out,pd.DataFrame(summaries)


def self_test():
    # Strict inequality, mixed missing-feature rows, and invalid source dates.
    good=pd.DataFrame([
        {'game_id':'2024020001','game_date_dt':pd.Timestamp('2024-10-10'),XG_FIELD:1.25,
         'xg_max_source_game_date':'2024-10-09'},
        {'game_id':'2024020002','game_date_dt':pd.Timestamp('2024-10-10'),XG_FIELD:np.nan,
         'xg_max_source_game_date':None},
    ])
    validate_asof(good)
    for stamp in ('2024-10-10','2024-10-11'):
        bad=good.copy();bad.loc[0,'xg_max_source_game_date']=stamp
        try:validate_asof(bad)
        except ValueError:pass
        else:raise AssertionError('Should reject xG dated on/after game')
    check(tuple(XG_FEATURES[:-1])==BASE,'V5 baseline feature list changed')
    # Exercise penalty/training math deterministically, without downloads.
    rng=np.random.default_rng(9)
    n=300
    sample=pd.DataFrame({x:rng.normal(size=n) for x in BASE})
    sample[XG_FIELD]=rng.normal(size=n)
    sample['target']=(rng.random(n)<.55).astype(float)
    model=RidgeLogit(XG_FEATURES,60.).fit(sample)
    preds=model.predict(sample)
    check(len(preds)==n and np.isfinite(preds).all() and ((preds>0)&(preds<1)).all(),
          'Ridge xG prediction self-test failed')
    print('SELF-TEST PASS: strict xG cutoffs, missingness, regression fit, unchanged V5 features')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',help='YYYY_MM_DD; defaults latest merged slate')
    parser.add_argument('--rebuild-xg',action='store_true',help='Recompute pregame xG from local P5 despite a matching cache')
    parser.add_argument('--self-test',action='store_true',help='Validate chronology/model logic without P5')
    a=parser.parse_args()
    if a.self_test:
        self_test();return
    historical=[]
    for year in (2022,2023,2024):
        historical.append(v5.load_old_season(None,year,offline=True))
    history,current,local=v5.historical_and_current(pd.concat(historical,ignore_index=True))
    challenger=get_challenger()
    all_targets=(pd.concat([history,current],ignore_index=True)
                 .drop_duplicates('game_id',keep='first'))
    features=[];source_rows=[]
    for y in YEARS:
        f,s=build_one(y,all_targets,challenger,rebuild=a.rebuild_xg)
        features.append(f[['game_id',XG_FIELD,'xg_max_source_game_date']])
        source_rows.append(s)
    joined=pd.concat(features,ignore_index=True)
    check(not joined.game_id.duplicated().any(),'Duplicate game IDs across xG seasons')
    history=history.merge(joined,on='game_id',how='left',validate='one_to_one')
    current=current.merge(joined,on='game_id',how='left',validate='one_to_one')
    check(history[XG_FIELD].notna().mean()>=MIN_XG_COVERAGE,
          'Not enough xG across historical training games')
    first=history.loc[history.data_period.isin(['2022_2023','2023_2024'])].copy()
    valid=history.loc[history.data_period.eq('2024_2025')].copy()
    check(len(first)>=2200 and len(valid)>=1100,'Insufficient historical training/validation')
    check(first.game_date_dt.max()<valid.game_date_dt.min(),'Temporal leakage in xG selection train/validate')
    prior_selection_path=v5.OUT/'reports'/'model_selection_2024_validation.csv'
    check(prior_selection_path.is_file(),f'Missing locked V5 selection: {prior_selection_path}')
    previous=pd.read_csv(prior_selection_path)
    previous=previous.sort_values(['brier','log_loss','model','penalty']).iloc[0]
    check(str(previous.model)==BASE_MODEL and float(previous.penalty)==BASE_PENALTY,
          'V5 locked baseline differs from expected elo_form penalty=15; refusing comparison')
    raw_baseline=RidgeLogit(BASE,BASE_PENALTY).fit(first).predict(valid)
    base_brier=float(brier(valid.target,raw_baseline))
    check(abs(base_brier-float(previous.brier))<2e-5,
          f'Cannot reproduce locked V5 2024 validation: {base_brier:.7f} vs {previous.brier:.7f}')
    candidate=[]
    hold={}
    candidate.append({'model':'v5_baseline','penalty':BASE_PENALTY,'games':len(valid),
                      'brier':base_brier,'log_loss':float(log_loss(valid.target,raw_baseline)),
                      'brier_minus_v5':0.0,'selection_dataset':'2024_2025_only'})
    for penalty in PENALTIES:
        pred=RidgeLogit(XG_FEATURES,penalty).fit(first).predict(valid)
        label=f'v8_xg_penalty_{penalty:g}'
        hold[label]=pred
        candidate.append({'model':label,'penalty':penalty,'games':len(valid),
                          'brier':float(brier(valid.target,pred)),
                          'log_loss':float(log_loss(valid.target,pred)),
                          'brier_minus_v5':float(brier(valid.target,pred)-base_brier),
                          'selection_dataset':'2024_2025_only'})
    report=pd.DataFrame(candidate).sort_values(['brier','log_loss','model'])
    xg_best=report.loc[report.model.ne('v5_baseline')].iloc[0]
    xg_penalty=float(xg_best.penalty)
    xg_valid=hold[str(xg_best.model)]
    val=valid[['game_id','game_date','game_date_dt','data_period','target']].copy()
    val['v5_base_home']=raw_baseline;val['v8_xg_home']=xg_valid
    val['diff_xg_minus_v5']=(val.v8_xg_home-val.target)**2-(val.v5_base_home-val.target)**2
    low,high=date_block_interval(val,'diff_xg_minus_v5')
    improvement=base_brier-float(xg_best.brier)
    adopted=(improvement>=MIN_VALIDATION_IMPROVEMENT and high<0)
    print(f'LOCKED 2024-25 VALIDATION: V5={base_brier:.6f}; best xG={float(xg_best.brier):.6f}; '+
          f'xG penalty={xg_penalty:g}; gain={improvement:.6f}; paired 95% CI [{low:+.6f},{high:+.6f}]',flush=True)
    print('ADOPTION GATE:', 'xG candidate qualified on 2024 only' if adopted else 'REJECT xG candidate; keep V5 baseline',flush=True)
    rows=[]
    for season in ('2025_2026','2026_2027'):
        targets=history.loc[history.data_period.eq(season)]
        for i,(day,group) in enumerate(targets.groupby('game_date_dt',sort=True),1):
            past=history.loc[history.game_date_dt<day]
            check(len(past)>2500 and past.game_date_dt.max()<day,
                  'Strict chronological training guard failed')
            base=RidgeLogit(BASE,BASE_PENALTY).fit(past).predict(group)
            xg=RidgeLogit(XG_FEATURES,xg_penalty).fit(past).predict(group)
            scored=group[['game_id','game_date','game_date_dt','data_period','target',XG_FIELD,'xg_max_source_game_date']].copy()
            scored['v5_base_home']=base;scored['v8_xg_home']=xg
            scored['selected_home']=xg if adopted else base
            scored['train_games_before_target']=len(past)
            scored['max_training_game_date']=past.game_date_dt.max().strftime('%Y-%m-%d')
            rows.append(scored)
            if i%45==0:print(f'HISTORICAL REPLAY {season}: {i} target dates',flush=True)
    replay=pd.concat(rows,ignore_index=True)
    replay_with_ref,summaries=compare_historical(replay,local)
    v5_eval_file=v5.OUT/'reports'/'chronological_2025_2026_evaluation.csv'
    if v5_eval_file.is_file():
        prior=pd.read_csv(v5_eval_file,dtype={'game_id':str})
        prior=prior[['game_id','my_probability_home']].rename(columns={'my_probability_home':'prior_v5_home'})
        matched=replay.merge(prior,on='game_id',how='inner',validate='one_to_one')
        check(len(matched)>=1200,'Too few existing V5 predictions to cross-validate baseline')
        maxdelta=float(np.abs(matched.v5_base_home-matched.prior_v5_home).max())
        check(maxdelta<.00005,
              f'V8 baseline differs from locked V5 walkforward probabilities (max difference {maxdelta}); refusing invalid ablation')
        print(f'BASELINE REPRODUCED: {len(matched)} locked V5 predictions, max |delta|={maxdelta:.8f}',flush=True)
    target=pd.Timestamp(a.date.replace('_','-')) if a.date else current.game_date_dt.max()
    slate=current.loc[current.game_date_dt.eq(target)]
    check(not slate.empty, f'No merged slate for {target.date()}')
    past=history.loc[history.game_date_dt<target]
    check(len(past)>2500 and past.game_date_dt.max()<target,'Forecast uses same/future outcomes')
    base=RidgeLogit(BASE,BASE_PENALTY).fit(past).predict(slate)
    experimental=RidgeLogit(XG_FEATURES,xg_penalty).fit(past).predict(slate)
    forecast=slate[['game_id','game_date','home_team','away_team','vig_free_home_prob',XG_FIELD,'xg_max_source_game_date']].copy()
    forecast['v5_baseline_home']=base
    forecast['v8_xg_experimental_home']=experimental
    forecast['my_probability_home']=experimental if adopted else base
    forecast['my_probability_away']=1.-forecast.my_probability_home
    forecast['xg_adopted_on_2024_validation']=adopted
    forecast['market_fallback_used']=False
    forecast['model_uses_sportsbook_odds']=False
    forecast['generated_at_utc']=datetime.now(timezone.utc).isoformat()
    provenance={'model_version':VERSION,'source_seasons':list(YEARS),
                'strict_target_source_rule':'source_game_date < target_game_date',
                'xg_source':'local_P5_PBP_SDV_prior_adjusted_ratings',
                'training_seasons':['2022_2023','2023_2024'],
                'parameter_selection':'2024_2025_only',
                'chosen_xg_penalty_2024':xg_penalty,
                'validation_brier_v5':base_brier,'validation_brier_xg':float(xg_best.brier),
                'paired_date_95ci_xg_minus_v5':[low,high],
                'xg_adopted_by_2024_gate':adopted,
                'historical_2025_2027_role':'exploratory_replay_not_pristine_holdout',
                'market_fallback_used':False,'odds_as_model_features':False,
                'postgame_goalies_or_lineups_used':False,
                'production_changed':False}
    report_dir=OUT/'reports'
    save_csv(pd.DataFrame(source_rows),report_dir/'p5_xg_coverage.csv')
    save_csv(report,report_dir/'2024_ablation_candidates.csv')
    save_csv(val.drop(columns='game_date_dt'),report_dir/'2024_ablation_predictions.csv')
    save_json(provenance,report_dir/'selection_provenance.json')
    save_csv(replay_with_ref.drop(columns='game_date_dt'),report_dir/'2025_2027_chronological_replay.csv')
    save_csv(summaries,report_dir/'2025_2027_replay_summary.csv')
    save_csv(forecast,OUT/'predictions'/f'{target:%Y_%m_%d}_NHL_my_probability_independent_v8.csv')
    print('HISTORICAL REPLAY RESULTS (exploratory):',flush=True)
    print(summaries[['season','model','games','brier']].to_string(index=False),flush=True)
    print('Independent xG experiment only; no production changes. Market prices benchmark only.',flush=True)


if __name__=='__main__':
    try: main()
    except (ValueError,RuntimeError,OSError,ImportError,KeyError) as e:
        sys.exit(f'ERROR: {e}')
