#!/usr/bin/env python3
"""Evaluate whether independent V8 xG and historical D-Ratings complement each other.

SELECTION: Only the pre-existing 2024-25 V8 validation forecasts joined to
independently saved 2024-25 pregame D-Ratings may choose a blend weight.
The already-reviewed 2025-26/2026-27 chronological replay is exploratory and
never selects any weight, rule, or model.

If 2024-25 historical D-Ratings cannot be located/verified, fail *closed* for
blend selection; still report fixed 50/50 sensitivity and pairwise comparisons
on the existing replay. No sportsbook prices enter any blended probabilities.

Outputs under NHL/my_probability/independent_v14_dratings_blend/reports only.
No V5/V8 files or production selections are modified.

Optional: --validation-file PATH  Explicit historical 2024-25 D-Ratings CSV.
Optional: --nhl-root PATH         Test override.
Optional: --self-test             Run local fixture regression tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 'v14_independent_v8_plus_dratings_20261120'
WEIGHTS_V8 = (0.0, 0.25, 0.50, 0.75, 1.0)
MIN_2024_COVERAGE = 0.90
MIN_2024_MATCHES = 1000
PREDICTORS = {
    'independent_V8_xG': 'v8_xg_home',
    'D_Ratings': 'drat_home_win_prob',
    'SportsDataverse': 'sdv_home_win_prob',
    'no_vig_market': 'vig_free_home_prob',
}


def demand(ok, message):
    if not bool(ok):
        raise ValueError(message)


def fingerprint(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def atomic_csv(data: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    data.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f'WROTE {path} | rows={len(data)}', flush=True)


def atomic_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)
    print(f'WROTE {path}', flush=True)


def normalize_ids(data, label):
    demand('game_id' in data.columns, f'{label}: missing official game_id')
    out = data.copy()
    ids = out['game_id'].astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
    demand(ids.notna().all() and ids.str.fullmatch(r'\d{10}').all(), f'{label}: invalid game IDs')
    demand(~ids.duplicated().any(), f'{label}: duplicated official game IDs')
    out['game_id'] = ids.astype(str)
    return out


def days(values, label):
    parsed = pd.to_datetime(values.astype(str).str.replace('_', '-', regex=False), errors='coerce')
    demand(parsed.notna().all(), f'{label}: invalid dates')
    return parsed


def probs(data, columns, label, allow_missing=False):
    for column in columns:
        demand(column in data.columns, f'{label}: missing {column}')
        x = pd.to_numeric(data[column], errors='coerce')
        valid = x.notna()
        if not allow_missing:
            demand(valid.all(), f'{label}: missing {column} probability')
        demand(np.isfinite(x[valid]).all() and x[valid].between(0, 1, inclusive='neither').all(),
               f'{label}: {column} probabilities must be strictly between 0 and 1')
        data[column] = x


def load_validation(path: Path):
    demand(path.is_file(), f'V8 validation file missing: {path}')
    df = normalize_ids(pd.read_csv(path, dtype={'game_id': 'string'}), 'V8 2024 validation')
    demand({'game_date', 'target', 'v8_xg_home'}.issubset(df), '2024 validation missing fields')
    demand(df.game_id.str.startswith('202402').all(), '2024 V8 validation contains non-2024 regular-season games')
    df['day'] = days(df.game_date, '2024 V8 validation')
    demand(df.day.between('2024-09-01', '2025-07-01').all(), '2024 validation has wrong dates')
    df['target'] = pd.to_numeric(df.target, errors='coerce')
    demand(df.target.isin((0.,1.)).all(), '2024 validation has invalid outcome')
    probs(df, ['v8_xg_home'], '2024 V8 validation')
    demand(len(df) >= 1100, f'2024 validation unexpectedly small ({len(df)})')
    return df[['game_id','day','target','v8_xg_home']].copy()


def load_replay(path: Path):
    demand(path.is_file(), f'V8 chronological replay missing: {path}')
    df = normalize_ids(pd.read_csv(path, dtype={'game_id':'string'}), 'V8 replay')
    columns=['game_date','data_period','target',*PREDICTORS.values()]
    demand(set(columns).issubset(df.columns), 'V8 replay missing: '+', '.join(sorted(set(columns)-set(df.columns))))
    demand(df.game_id.str.slice(0,4).isin(('2025','2026')).all(), 'V8 replay has wrong season IDs')
    df['day'] = days(df.game_date, 'V8 replay')
    demand(df.data_period.isin(('2025_2026','2026_2027')).all(), 'V8 replay has unexpected seasons')
    demand((df.game_id.str.slice(0,4)+'_'+(df.game_id.str.slice(0,4).astype(int)+1).astype(str)).eq(df.data_period).all(),
           'V8 replay season ID/data_period mismatch')
    df['target'] = pd.to_numeric(df.target,errors='coerce')
    demand(df.target.isin((0.,1.)).all(), 'V8 replay contains nonbinary outcomes')
    probs(df, list(PREDICTORS.values()), 'V8 replay', allow_missing=True)
    demand(df.v8_xg_home.notna().all(), 'V8 replay missing independent predictions')
    demand(df.groupby('data_period').size().ge(300).all() and df.data_period.nunique()==2,
           'V8 replay lacks both seasons of completed games')
    demand(df.loc[df.data_period.eq('2025_2026'),'day'].max() < df.loc[df.data_period.eq('2026_2027'),'day'].min(),
           'V8 replay is not separated chronologically by season')
    return df


def enumerate_2024_files(root: Path):
    # Search only plausible 2024 historical prediction directories. Never use
    # V8/V5 outputs, which would be circular rather than real D-Ratings input.
    patterns = [
        '00_intake/predictions/**/*2024*.csv',
        '00_intake/predictions/**/*2025*.csv',
        '00_intake/drat_raw/**/*2024*.csv',
        '00_intake/drat_raw/**/*2025*.csv',
        '01_merge/*2024*.csv',
        '01_merge/*2025*.csv',
        'archive/2024*/**/*.csv',
        'backtest/2024*/**/*.csv',
        'season_master/2024*/**/*.csv',
        'research/sdv_challenger/season_2024/**/*.csv',
        'research/dratings/**/*2024*.csv',
        'research/dratings/**/*2025*.csv',
    ]
    paths=[]
    seen=set()
    for pat in patterns:
        for item in root.glob(pat):
            if item.is_file() and item not in seen:
                seen.add(item);paths.append(item)
    return sorted(paths)


def inspect_2024_source(path: Path, target_ids: set[str]):
    """Return matching official-ID 2024 D-Ratings rows; never infer by names."""
    try:
        header=pd.read_csv(path,nrows=0,encoding='utf-8-sig').columns
        predcol=next((c for c in ('home_prob_moneyline','drat_home_win_prob','dratings_home_prob')
                      if c in header),None)
        if not predcol or 'game_id' not in header:
            return None,'missing_canonical_id_or_D_Ratings_field'
        use=['game_id',predcol]
        for extra in ('game_date','pulled_at','observed_at_utc','start_time_utc','home_team','away_team'):
            if extra in header:use.append(extra)
        table=pd.read_csv(path,usecols=use,dtype={'game_id':'string'},encoding='utf-8-sig',low_memory=False)
        table['game_id']=table.game_id.astype('string').str.strip().str.replace(r'\.0$', '',regex=True)
        table=table.loc[table.game_id.isin(target_ids)].copy()
        if table.empty:
            return None,'no_matching_2024_official_ids'
        demand(not table.game_id.duplicated().any(), f'{path}: ambiguous duplicate 2024 predictions')
        table[predcol]=pd.to_numeric(table[predcol],errors='coerce')
        demand(table[predcol].notna().all() and table[predcol].between(0,1,inclusive='neither').all(),
               f'{path}: invalid D-Ratings probability')
        if 'game_date' in table:
            table['game_date_source']=days(table.game_date,f'{path} D-Ratings dates')
        table=table.rename(columns={predcol:'drat_home_win_prob'})
        return table,'2024_matching_rows'
    except (OSError,KeyError,pd.errors.ParserError) as exc:
        return None,'invalid_source: '+str(exc)[:150]


def gather_2024_dratings(root:Path, validation:pd.DataFrame, explicit:Path|None):
    candidates=[explicit] if explicit is not None else enumerate_2024_files(root)
    available=[];inventory=[]
    id_set=set(validation.game_id)
    for path in candidates:
        demand(path is not None and path.is_file(),f'Explicit 2024 file missing: {path}')
        table,status=inspect_2024_source(path,id_set)
        inventory.append({'source':str(path),'status':status,'matches':0 if table is None else len(table)})
        if table is None:continue
        if 'game_date_source' in table:
            paired=table[['game_id','game_date_source']].merge(validation[['game_id','day']],on='game_id',validate='one_to_one')
            demand((paired.game_date_source==paired.day).all(),f'{path}: D-Ratings game dates mismatch official game dates')
        available.append(table[['game_id','drat_home_win_prob']].assign(_source=str(path)))
    if not available:
        return None,pd.DataFrame(inventory,columns=['source','status','matches']),'no_matching_2024_D_Ratings_data'
    together=pd.concat(available,ignore_index=True)
    repeated=together.loc[together.game_id.duplicated(keep=False)]
    if not repeated.empty:
        different=repeated.groupby('game_id').drat_home_win_prob.agg(lambda x: x.max()-x.min())
        demand((different<=1e-12).all(), 'Conflicting 2024 D-Ratings probabilities across files; refusing arbitrary selection')
        together=together.drop_duplicates('game_id',keep='first')
    joined=validation.merge(together[['game_id','drat_home_win_prob']],on='game_id',how='left',validate='one_to_one')
    coverage=float(joined.drat_home_win_prob.notna().mean())
    if coverage < MIN_2024_COVERAGE or int(joined.drat_home_win_prob.notna().sum()) < MIN_2024_MATCHES:
        return None,pd.DataFrame(inventory,columns=['source','status','matches']),f'insufficient_2024_D_Ratings_coverage_{coverage:.1%}'
    return joined.dropna(subset=['drat_home_win_prob']).copy(),pd.DataFrame(inventory,columns=['source','status','matches']),f'matched_{len(joined.dropna(subset=["drat_home_win_prob"]))}_of_{len(validation)}'


def brier(data, column):
    return float(np.mean((data[column].to_numpy(float)-data.target.to_numpy(float))**2))


def logloss(data, column):
    p=data[column].to_numpy(float).clip(1e-12,1-1e-12)
    y=data.target.to_numpy(float)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))


def bootstrap_paired(data, first, second, seed=20261120, n=2000):
    copy=data[['day','target',first,second]].dropna().copy()
    if len(copy)<25 or copy.day.nunique()<15:
        return (float('nan'),float('nan'))
    diff=(copy[first]-copy.target)**2-(copy[second]-copy.target)**2
    copy['delta']=diff
    group=copy.groupby('day',sort=True).delta.agg(['sum','count'])
    sums=group['sum'].to_numpy(float);counts=group['count'].to_numpy(int)
    rng=np.random.default_rng(seed)
    selected=rng.integers(0,len(sums),size=(n,len(sums)))
    draws=sums[selected].sum(axis=1)/counts[selected].sum(axis=1)
    return tuple(np.quantile(draws,[0.025,0.975]).tolist())


def validate_and_select(validation):
    """Selection is NEVER based on 2025/26 replay."""
    results=[]
    for w in WEIGHTS_V8:
        col=f'blend_v8_{int(w*100):03d}'
        validation[col]=w*validation.v8_xg_home+(1-w)*validation.drat_home_win_prob
        results.append({'weight_v8':w,'weight_dratings':1-w,'games':len(validation),
                        'brier':brier(validation,col),'log_loss':logloss(validation,col),
                        'training_or_selection_period':'2024_2025_only'})
    results=pd.DataFrame(results).sort_values(['brier','log_loss','weight_v8'],ascending=[True,True,False])
    selected=float(results.iloc[0].weight_v8)
    selected_column=f'blend_v8_{int(selected*100):03d}'
    ci_v8=bootstrap_paired(validation,selected_column,'v8_xg_home')
    ci_drat=bootstrap_paired(validation,selected_column,'drat_home_win_prob')
    passes_both=(ci_v8[1]<0 and ci_drat[1]<0)
    return results,{'weight_v8':selected,'selected_2024_brier':float(results.iloc[0].brier),
                    'v8_brier':brier(validation,'v8_xg_home'),
                    'dratings_brier':brier(validation,'drat_home_win_prob'),
                    'selected_vs_v8_95ci':ci_v8,'selected_vs_dratings_95ci':ci_drat,
                    'beats_both_2024_with_block_bootstrap':passes_both,
                    'recommend_adoption':False,
                    'validation_source_capture_timestamp_independently_authenticated':False}


def run_reports(validation, replay):
    # Replay covers already-examined seasons: fixed sensitivity analyses ONLY.
    # Do NOT inspect these results to choose which weight to publish/deploy.
    summary=[];pairs=[]
    for season,game in replay.groupby('data_period',sort=True):
        data=game.dropna(subset=['target','v8_xg_home','drat_home_win_prob']).copy()
        demand(len(data)>=300, f'{season}: too few paired V8/DRatings rows')
        for w in WEIGHTS_V8:
            label=f'blend_v8_{int(w*100):03d}'
            data[label]=w*data.v8_xg_home+(1-w)*data.drat_home_win_prob
        for label,column in [*PREDICTORS.items(),*[(f'fixed_V8_{int(w*100)}pct',f'blend_v8_{int(w*100):03d}') for w in WEIGHTS_V8]]:
            complete=data.dropna(subset=[column])
            summary.append({'season':season,'model':label,'games':len(complete),
                            'brier':brier(complete,column),'log_loss':logloss(complete,column),
                            'classification':'replayed_2025_2027_exploratory_only'})
        # Compare on exactly the paired complete-case sample, not changing denominators.
        for name,compare_to in [('V8_vs_D_Ratings','drat_home_win_prob'),
                                ('fixed_50_50_vs_V8','v8_xg_home'),
                                ('fixed_50_50_vs_D_Ratings','drat_home_win_prob')]:
            comp='v8_xg_home' if name=='V8_vs_D_Ratings' else 'blend_v8_050'
            lo,hi=bootstrap_paired(data,comp,compare_to)
            difference=brier(data,comp)-brier(data,compare_to)
            pairs.append({'season':season,'test':name,'games':len(data),'dates':data.day.nunique(),
                          'brier_difference':difference,'paired_date_95ci_low':lo,
                          'paired_date_95ci_high':hi,
                          'exploratory_not_new_holdout':True})
    return pd.DataFrame(summary),pd.DataFrame(pairs)


def main(root:Path, explicit:Path|None):
    v8=root/'my_probability'/'independent_v8_xg'/'reports'
    validation_path=v8/'2024_ablation_predictions.csv'
    replay_path=v8/'2025_2027_chronological_replay.csv'
    output=root/'my_probability'/'independent_v14_dratings_blend'/'reports'
    val=load_validation(validation_path)
    replay=load_replay(replay_path)
    historical,inventory,status=gather_2024_dratings(root,val,explicit)
    selected=None
    if historical is not None:
        selection,selected=validate_and_select(historical)
        atomic_csv(selection,output/'2024_blend_candidates.csv')
        print(f'2024 VALIDATION: {status}; selected weight V8={selected["weight_v8"]:.2f}',flush=True)
        print('2024 PAIRED DATE CI blend minus V8:',selected['selected_vs_v8_95ci'],flush=True)
        print('2024 PAIRED DATE CI blend minus D-Ratings:',selected['selected_vs_dratings_95ci'],flush=True)
        print('NOT ADOPTED: provider capture timestamp is not independently authenticated',flush=True)
    else:
        print(f'NO 2024 BLEND SELECTION: {status}',flush=True)
    summary,pairs=run_reports(val,replay)
    atomic_csv(inventory,output/'2024_historical_provider_inventory.csv')
    atomic_csv(summary,output/'2025_2027_fixed_weight_comparisons.csv')
    atomic_csv(pairs,output/'2025_2027_paired_date_intervals.csv')
    record={'version':VERSION,'validation_status':status,'2024_validated_blend_selection':selected,
            'blend_selection_allowed':historical is not None,
            'selection_period':'2024_2025_only_if_verified_historical_provider_source',
            '2025_2027_role':'repeatedly_inspected_exploratory_sensitivity_not_selection',
            'market_probabilities_used_as_features':False,
            'sportsdataverse_probabilities_used_as_features':False,
            '2024_provider_capture_timestamps_authenticated':False,
            'production_modified':False,'model_outputs_modified':False,
            'input_sha256':{str(validation_path):fingerprint(validation_path),
                            str(replay_path):fingerprint(replay_path)}}
    atomic_json(record,output/'manifest.json')
    print('FIXED-WEIGHT COMPARISON (lower Brier is better):',flush=True)
    print(summary.loc[summary.model.isin(['independent_V8_xG','D_Ratings','SportsDataverse','no_vig_market','fixed_V8_50pct']),['season','model','games','brier']].to_string(index=False),flush=True)
    print('No V8 or V5 prediction changed; no bet selection.',flush=True)


def self_test():
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        directory=root/'my_probability'/'independent_v8_xg'/'reports'
        directory.mkdir(parents=True)
        ids=[f'202402{i:04d}' for i in range(1312)]
        rng=np.random.default_rng(100)
        validation=pd.DataFrame({'game_id':ids,
             'game_date':[(pd.Timestamp('2024-10-01')+pd.Timedelta(days=i//8)).strftime('%Y_%m_%d') for i in range(1312)],
             'target':rng.integers(0,2,size=1312),
             'v8_xg_home':rng.uniform(.30,.70,size=1312)})
        validation.to_csv(directory/'2024_ablation_predictions.csv',index=False)
        replay=[]
        for year,period,n in ((2025,'2025_2026',360),(2026,'2026_2027',350)):
            for i in range(n):
                replay.append({'game_id':f'{year}02{i:04d}',
                  'game_date':(pd.Timestamp(f'{year}-10-01')+pd.Timedelta(days=i//8)).strftime('%Y_%m_%d'),
                  'data_period':period,'target':float(rng.integers(0,2)),
                  'v8_xg_home':float(rng.uniform(.3,.7)),
                  'drat_home_win_prob':float(rng.uniform(.3,.7)),
                  'sdv_home_win_prob':float(rng.uniform(.3,.7)),
                  'vig_free_home_prob':float(rng.uniform(.3,.7))})
        pd.DataFrame(replay).to_csv(directory/'2025_2027_chronological_replay.csv',index=False)
        # Absent provider must not select, but exploratory reports are allowed.
        main(root,None)
        manifest=json.loads((root/'my_probability'/'independent_v14_dratings_blend'/'reports'/'manifest.json').read_text())
        demand(not manifest['blend_selection_allowed'],'Test: absent historical D Ratings must not select')
        # When saved historical provider forecasts do exist, select ONLY on 2024.
        source=root/'2024_D_Ratings.csv'
        pd.DataFrame({'game_id':ids,'home_prob_moneyline':rng.uniform(.3,.7,size=len(ids))}).to_csv(source,index=False)
        main(root,source)
        manifest=json.loads((root/'my_probability'/'independent_v14_dratings_blend'/'reports'/'manifest.json').read_text())
        demand(manifest['blend_selection_allowed'] and manifest['2024_validated_blend_selection'] is not None,
               'Test: 2024-only blend selection failed')
        bad=pd.read_csv(source)
        bad.loc[1,'game_id']=bad.loc[0,'game_id']
        bad.to_csv(source,index=False)
        try:
            main(root,source)
        except ValueError:
            pass
        else:
            raise AssertionError('Test: duplicated 2024 provider ID must fail closed')
        print('SELF-TEST PASS: no fake 2024 selection, 2024-only tuning, strict duplicate rejection',flush=True)


if __name__=='__main__':
    try:
        ap=argparse.ArgumentParser(description=__doc__)
        ap.add_argument('--nhl-root',type=Path,default=Path(__file__).resolve().parents[2])
        ap.add_argument('--validation-file',type=Path)
        ap.add_argument('--self-test',action='store_true')
        args=ap.parse_args()
        if args.self_test:self_test()
        else:main(args.nhl_root,args.validation_file)
    except (OSError,ValueError,KeyError,pd.errors.ParserError) as exc:
        sys.exit(f'ERROR: {exc}')
