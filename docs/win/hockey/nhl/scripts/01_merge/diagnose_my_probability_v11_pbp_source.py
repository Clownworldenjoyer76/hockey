#!/usr/bin/env python3
"""V11, read-only audit: locate where the apparent xG scale shrinks.

Compares matched first-N-game-date P5 raw xG events and game differentials
against the locked, strict-prior V8 adjusted xG differential. N equals
2026-27's available target-game dates. Historical 2022-24 records are the
reference; 2024-26 results are *descriptive*, not tuning/selection evidence.

Uses local P5 schedules/PBP and existing V8 CSVs. Never fetches, trains,
changes V5/V8, selects hyperparameters, uses odds, or enters betting stages.

Run beside ablate_my_probability_v8_xg.py and v9 diagnostics script:
    python diagnose_my_probability_v11_pbp_source.py
    python diagnose_my_probability_v11_pbp_source.py --self-test
Writes under NHL/my_probability/independent_v11_xg_source_audit/reports/.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Must precede importing Polars (including imports that bring in SDV).
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')

import numpy as np
import pandas as pd

import diagnose_my_probability_v9_xg_drift as v9

YEARS = tuple(v9.YEARS)
REF_YEARS = (2022, 2023)
CURRENT_YEAR = 2026
XG = v9.XG
ROOT = v9.v8.ROOT
P5 = ROOT / 'research' / 'sdv_history'
OUT = ROOT / 'my_probability' / 'independent_v11_xg_source_audit' / 'reports'
VERSION = 'v11_xg_raw_source_diagnosis_20261120_1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / name
    pending = dest.with_suffix('.pending')
    df.to_csv(pending, index=False, float_format='%.10g')
    pending.replace(dest)
    print(f'WROTE {dest} | rows={len(df)}', flush=True)


def write_text(content: str, name: str = 'findings.txt') -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / name
    pending = dest.with_suffix('.pending')
    pending.write_text(content, encoding='utf-8')
    pending.replace(dest)
    print(content, flush=True)
    print(f'WROTE {dest}', flush=True)


def normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)


def normalized_day(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype('string').str.slice(0, 10).str.replace('_', '-', regex=False),
                          errors='coerce').dt.normalize()


def get_maturity_window(caches: pd.DataFrame):
    df = caches.copy()
    require({'game_id', 'game_date_dt', 'data_period', XG}.issubset(df), 'V8 prior xG cache fields missing')
    df['game_id'] = normalize_ids(df.game_id)
    df['game_date_dt'] = normalized_day(df.game_date_dt)
    require(df.game_date_dt.notna().all() and df.game_id.notna().all(), 'Invalid V8 cache dates/game IDs')
    require(not df.game_id.duplicated().any(), 'Duplicate game IDs across V8 xG cache')
    df = df.sort_values(['data_period', 'game_date_dt', 'game_id']).reset_index(drop=True)
    dates = df.loc[df.data_period.eq('2026_2027'), 'game_date_dt'].drop_duplicates()
    n_dates = int(len(dates))
    require(n_dates >= 20, 'Current-season V8 cache has too few game dates to compare')
    df['date_rank'] = df.groupby('data_period').game_date_dt.rank(method='dense').astype(int)
    require(set(df.data_period) == set(f'{y}_{y+1}' for y in YEARS),
            'V8 cache missing a season; refusing mismatched comparison')
    return df, n_dates


def load_parquet_selected(path: Path, required: tuple[str, ...], optional: tuple[str, ...] = ()): 
    require(path.is_file(), f'P5 file not found: {path}')
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError('Polars missing; run in existing NHL environment where build_sdv_history.py worked') from exc
    try:
        cols = pl.scan_parquet(path).collect_schema().names()
    except Exception as exc:
        raise RuntimeError(f'Cannot inspect P5 schema {path}: {exc}') from exc
    require(set(required).issubset(cols), f'P5 {path} missing {set(required)-set(cols)}')
    selected = list(dict.fromkeys((*required, *(c for c in optional if c in cols))))
    frame = pl.read_parquet(path, columns=selected)
    return frame.to_pandas(), cols


def load_schedule(year: int) -> tuple[pd.DataFrame, dict]:
    path = P5 / 'schedules' / f'season_{year}.parquet'
    # The builder uses home_team_abbr / home_team_abbrev / home_abbr aliases.
    required = ('game_id', 'source_game_date')
    optional = ('home_team_abbr', 'home_team_abbrev', 'home_abbr',
                'away_team_abbr', 'away_team_abbrev', 'away_abbr')
    frame, cols = load_parquet_selected(path, required, optional)
    home = next((c for c in ('home_team_abbr', 'home_team_abbrev', 'home_abbr') if c in frame), None)
    away = next((c for c in ('away_team_abbr', 'away_team_abbrev', 'away_abbr') if c in frame), None)
    require(bool(home) and bool(away), f'P5 schedule {year} has no team abbreviations')
    frame = frame[['game_id', 'source_game_date', home, away]].copy()
    frame.columns = ['game_id', 'source_game_date', 'home_abbr', 'away_abbr']
    frame['game_id'] = normalize_ids(frame.game_id)
    frame['source_game_date'] = normalized_day(frame.source_game_date)
    for side in ('home', 'away'):
        frame[side+'_abbr'] = frame[side+'_abbr'].astype('string').str.strip().str.upper()
    frame = frame.loc[frame.game_id.str.fullmatch(str(year)+r'02\d{4}').fillna(False)].copy()
    require(frame.game_id.notna().all() and frame.source_game_date.notna().all(),
            f'P5 schedule {year} has null IDs/dates')
    require(not frame.game_id.duplicated().any(), f'P5 schedule {year} has duplicate game IDs')
    require(frame.home_abbr.notna().all() and frame.away_abbr.notna().all() and
            frame.home_abbr.ne(frame.away_abbr).all(), f'P5 schedule {year} has invalid teams')
    return frame, {'season': f'{year}_{year+1}', 'dataset': 'schedules',
                   'path': str(path), 'bytes': path.stat().st_size,
                   'columns': len(cols), 'rows': len(frame),
                   'modified_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def load_pbp(year: int) -> tuple[pd.DataFrame, dict]:
    path = P5 / 'play_by_play' / f'season_{year}.parquet'
    required = ('game_id', 'event_type', 'event_team_abbr', 'xg', 'source_game_date')
    optional = ('event_id', 'strength_state', 'shot_distance', 'shot_angle',
                'home_skaters', 'away_skaters')
    frame, cols = load_parquet_selected(path, required, optional)
    frame['game_id'] = normalize_ids(frame.game_id)
    frame = frame.loc[frame.game_id.str.fullmatch(str(year)+r'02\d{4}').fillna(False)].copy()
    frame['source_game_date'] = normalized_day(frame.source_game_date)
    frame['event_team_abbr'] = frame.event_team_abbr.astype('string').str.strip().str.upper()
    frame['event_type'] = frame.event_type.astype('string').str.strip().str.upper()
    frame['xg'] = pd.to_numeric(frame.xg, errors='coerce')
    require(not np.isinf(frame.xg).any(), f'P5 {year}: infinite xG values')
    # PBP can have null source-game-date for some rows; these cannot receive source attestation.
    return frame, {'season': f'{year}_{year+1}', 'dataset': 'play_by_play',
                   'path': str(path), 'bytes': path.stat().st_size,
                   'columns': len(cols), 'rows': len(frame),
                   'modified_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def validate_and_join(pbp: pd.DataFrame, schedule: pd.DataFrame, target: pd.DataFrame):
    t = target[['game_id', 'game_date_dt', 'date_rank', XG]].copy()
    s = t.merge(schedule, on='game_id', how='left', validate='one_to_one', indicator=True)
    require((s._merge == 'both').all(),
            f'Schedule missing V8 IDs: {s.loc[s._merge.ne("both"),"game_id"].head(5).tolist()}')
    require((s.game_date_dt == s.source_game_date).all(),
            f'Schedule/V8 date disagreement: {s.loc[s.game_date_dt.ne(s.source_game_date), "game_id"].head(5).tolist()}')
    s = s.drop(columns=['_merge'])
    p = pbp.loc[pbp.game_id.isin(s.game_id)].copy()
    require(len(p)>0, 'No PBP matched any V8 cache games')
    p = p.merge(s[['game_id', 'game_date_dt', 'home_abbr', 'away_abbr', 'date_rank']],
                on='game_id', how='left', validate='many_to_one')
    # Never silently mix PBP events from a different date into a game's raw profile.
    mismatch = p.source_game_date.notna() & p.source_game_date.ne(p.game_date_dt)
    require(not mismatch.any(),
            f'PBP/V8 event source date disagreement: {p.loc[mismatch,"game_id"].head(5).tolist()}')
    return s, p


def per_game_components(s: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    # All PBP event types retained; only positive finite xG represents an xG event.
    # Unassigned team events remain in coverage counts but never home/away xG sums.
    positives = p.loc[p.xg.notna() & p.xg.gt(0)].copy()
    positives['assigned_home'] = positives.event_team_abbr.eq(positives.home_abbr).fillna(False)
    positives['assigned_away'] = positives.event_team_abbr.eq(positives.away_abbr).fillna(False)
    positives['unassigned'] = ~(positives.assigned_home | positives.assigned_away)
    positives['home_xg'] = positives.xg.where(positives.assigned_home, 0.)
    positives['away_xg'] = positives.xg.where(positives.assigned_away, 0.)
    aggregated = positives.groupby('game_id', sort=False).agg(
        positive_xg_event_count=('xg', 'size'),
        home_raw_xg=('home_xg', 'sum'), away_raw_xg=('away_xg', 'sum'),
        unassigned_xg_event_count=('unassigned', 'sum'),
        positive_event_mean_xg=('xg', 'mean')
    ).reset_index()
    game_events = p.groupby('game_id', sort=False).agg(
        pbp_events=('game_id', 'size'),
        xg_nonnull_events=('xg', 'count'),
        source_date_null_events=('source_game_date', lambda x: int(x.isna().sum()))
    ).reset_index()
    game = s[['game_id','game_date_dt','date_rank','home_abbr','away_abbr',XG]].merge(
        game_events,on='game_id',how='left',validate='one_to_one').merge(
        aggregated,on='game_id',how='left',validate='one_to_one')
    game['pbp_events'] = game.pbp_events.fillna(0).astype(int)
    game['positive_xg_event_count'] = game.positive_xg_event_count.fillna(0).astype(int)
    game['unassigned_xg_event_count'] = game.unassigned_xg_event_count.fillna(0).astype(int)
    game['raw_home_minus_away_xg'] = (game.home_raw_xg-game.away_raw_xg).where(game.positive_xg_event_count.gt(0))
    game['raw_total_xg'] = (game.home_raw_xg+game.away_raw_xg).where(game.positive_xg_event_count.gt(0))
    return game


def numeric_stats(values: pd.Series, prefix: str):
    y = pd.to_numeric(values, errors='coerce').dropna().to_numpy(float)
    return {prefix+'_n':len(y),prefix+'_mean':float(y.mean()) if len(y) else np.nan,
            prefix+'_sd':float(y.std(ddof=0)) if len(y) else np.nan,
            prefix+'_p50':float(np.median(y)) if len(y) else np.nan,
            prefix+'_p95':float(np.percentile(y,95)) if len(y) else np.nan}


def summarize(games: pd.DataFrame, pbp: pd.DataFrame, season: str, label: str):
    ids = set(games.game_id)
    events = pbp.loc[pbp.game_id.isin(ids)]
    xg = events.xg
    positive = events.loc[xg.notna() & xg.gt(0)]
    n_event = len(events)
    n_positive = len(positive)
    n_assigned = (positive.event_team_abbr.eq(positive.home_abbr) |
                  positive.event_team_abbr.eq(positive.away_abbr)).fillna(False).sum()
    row = {'season':season,'window':label,'games_scheduled':len(games),
           'games_with_pbp':int(games.pbp_events.gt(0).sum()),
           'games_with_positive_xg':int(games.positive_xg_event_count.gt(0).sum()),
           'pbp_event_count':n_event,'xg_nonnull_count':int(xg.notna().sum()),
           'xg_positive_count':n_positive,
           'xg_over_1_event_count':int(xg.gt(1).sum()),
           'xg_negative_event_count':int(xg.lt(0).sum()),
           'positive_xg_event_unassigned_count':int(n_positive-n_assigned),
           'null_pbp_source_date_count':int(events.source_game_date.isna().sum())}
    row.update(numeric_stats(positive.xg,'positive_event_xg'))
    row.update(numeric_stats(games.raw_total_xg,'raw_game_xg_total'))
    row.update(numeric_stats(games.raw_home_minus_away_xg,'raw_game_xg_diff'))
    row.update(numeric_stats(games[XG],'prior_adjusted_diff'))
    row['raw_game_xg_diff_mean_absolute'] = float(games.raw_home_minus_away_xg.abs().mean())
    row['prior_adjusted_diff_mean_absolute'] = float(games[XG].abs().mean())
    return row


def event_types(games: pd.DataFrame, pbp: pd.DataFrame, season: str, window: str):
    included = pbp.loc[pbp.game_id.isin(games.game_id)].copy()
    included['positive'] = included.xg.gt(0).fillna(False)
    e = included.groupby('event_type',dropna=False, observed=True).agg(
        events=('game_id','size'),positive_xg_events=('positive','sum'),
        xg_nonnull=('xg','count'),raw_xg_total=('xg',lambda x:float(x.clip(lower=0).sum()))
    ).reset_index()
    e.insert(0,'season',season)
    e.insert(1,'window',window)
    return e.sort_values('positive_xg_events', ascending=False)


def measure_and_compare(rows: pd.DataFrame):
    early = rows.loc[rows.window.str.startswith('first_')].copy()
    ref=early[early.season.isin(f'{y}_{y+1}' for y in REF_YEARS)]
    current=early[early.season.eq('2026_2027')]
    require(len(ref)==2 and len(current)==1, 'Reference/current maturity windows missing')
    # Descriptive summary mean of season scale stats; not a pooled event-level estimate.
    colnames = ('positive_event_xg_mean','positive_event_xg_sd',
                'raw_game_xg_total_mean','raw_game_xg_diff_sd','prior_adjusted_diff_sd')
    records=[]
    for col in colnames:
        base=float(ref[col].mean())
        now=float(current[col].iloc[0])
        records.append({'quantity':col,'reference_2022_24_first_N_dates_mean_of_seasons':base,
                        'current_2026_first_N_dates':now,
                        'current_to_reference_ratio':now/base if np.isfinite(base) and abs(base)>1e-8 else np.nan,
                        'status':'descriptive_only_not_hypothesis_test'})
    return pd.DataFrame(records)


def findings(comparison: pd.DataFrame, summary: pd.DataFrame, n_dates: int):
    values = dict(zip(comparison.quantity, comparison.current_to_reference_ratio))
    shot = values['positive_event_xg_mean']
    raw_diff = values['raw_game_xg_diff_sd']
    adjusted = values['prior_adjusted_diff_sd']
    if np.isfinite(shot) and shot < .80:
        interpretation='Raw per-positive-event xG is smaller in 2026; investigate PBP event xG definition/version before attributing anything to team ratings.'
    elif np.isfinite(shot) and .8 <= shot <= 1.2 and np.isfinite(raw_diff) and raw_diff < .8:
        interpretation='Raw xG per event is broadly comparable, but game-level raw differentials are compressed; inspect event counts, team attribution and schedule/opponent mix.'
    elif np.isfinite(shot) and .8 <= shot <= 1.2 and np.isfinite(raw_diff) and raw_diff >= .8 and np.isfinite(adjusted) and adjusted < .8:
        interpretation='Raw xG is broadly comparable; compression appears downstream in adjusted pregame team ratings or the limited set of prior games.'
    else:
        interpretation='Source-scale and adjusted-rating movements are mixed or incomplete; review the component reports before assigning causation.'
    parts = [
        f'VERSION {VERSION}',
        'READ ONLY: V5/V8/V9/V10 unchanged; no network, fitting, calibration, or betting.',
        f'EARLY WINDOW: first {n_dates} distinct regular-season target dates in each season.',
        'Raw PBP event xG is postgame event data used for source diagnostics, NOT a pregame model input.',
        'V8 adjusted xG is strictly prior-calendar-day; upstream historical capture timestamps remain unauthenticated.',
        'Same-stage reference is the mean of 2022-23 and 2023-24 season statistics.',
        f'2026 positive-event xG mean / early reference = {shot:.3f}',
        f'2026 raw game xG differential SD / early reference = {raw_diff:.3f}',
        f'2026 adjusted prior xG differential SD / early reference = {adjusted:.3f}',
        f'INTERPRETATION: {interpretation}',
        'The ratios are descriptive; lack of equal game samples, source-model provenance, opponent adjustments and correlated games prevents causal/significance claims.',
        'Do not tune on the previously inspected 2025-27 periods; no production integration.'
    ]
    return '\n'.join(parts) + '\n'


def self_test():
    # A synthetic duplicate-ish PBP fixture must preserve raw team assignment,
    # report missing source dates, and reject source-date mismatches.
    dates = pd.to_datetime(['2022-10-01','2022-10-01'])
    s=pd.DataFrame({'game_id':pd.Series(['2022020001','2022020002'],dtype='string'),
                    'game_date_dt':dates,'date_rank':[1,1], XG:[.3,-.3],
                    'source_game_date':dates,'home_abbr':['NYR','BOS'], 'away_abbr':['BOS','NYR']})
    events=pd.DataFrame({'game_id':pd.Series(['2022020001']*4+['2022020002']*2,dtype='string'),
                         'event_type':['SHOT','SHOT','GOAL','FACEOFF','SHOT','GOAL'],
                         'xg':[.1,.2,.3,None,.4,.2],
                         'event_team_abbr':pd.Series(['NYR','BOS','NYR',None,'BOS','NYR'],dtype='string'),
                         'source_game_date':list(pd.to_datetime(['2022-10-01']*6))})
    ss,p=validate_and_join(events,s[['game_id','source_game_date','home_abbr','away_abbr']],s)
    games=per_game_components(ss,p)
    require(len(games)==2,'Fixture per-game cardinality wrong')
    require(abs(float(games.iloc[0].raw_home_minus_away_xg)-.2)<1e-9,'Raw xG team attribution wrong')
    require(abs(float(games.iloc[1].raw_home_minus_away_xg)-.2)<1e-9,'Raw xG differential wrong')
    report=summarize(games,p,'2022_2023','first_1_dates')
    require(report['xg_positive_count']==5 and report['xg_nonnull_count']==5,'xG event counting wrong')
    wrong=events.copy();wrong.loc[0,'source_game_date']=pd.Timestamp('2022-10-02')
    try:validate_and_join(wrong,s[['game_id','source_game_date','home_abbr','away_abbr']],s)
    except ValueError:pass
    else:raise AssertionError('Mismatched PBP source date not rejected')
    print('SELF-TEST PASS: source date contract, event attribution, positive xG rates, nonshot missing xG')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true',help='Fixture tests without P5 Parquet files')
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    cache,n_dates=get_maturity_window(v9.load_xg_cache())
    manifest=[];summary_rows=[];event_frames=[];game_frames=[]
    for year in YEARS:
        season=f'{year}_{year+1}'
        print(f'AUDIT P5 RAW xG {season} | comparing first {n_dates} target dates',flush=True)
        subset=cache.loc[cache.data_period.eq(season)].copy()
        schedule,m1=load_schedule(year)
        pbp,m2=load_pbp(year)
        ss,ee=validate_and_join(pbp,schedule,subset)
        game=per_game_components(ss,ee)
        game.insert(0,'season',season)
        game_frames.append(game)
        manifest.extend([m1,m2])
        for window,g in [('all_available_dates',game), (f'first_{n_dates}_dates',game.loc[game.date_rank.le(n_dates)])]:
            summary_rows.append(summarize(g,ee,season,window))
            event_frames.append(event_types(g,ee,season,window))
        print(f'P5 {season}: {len(ss)} target games, {int(game.pbp_events.gt(0).sum())} with PBP, '+
              f'{int(game.positive_xg_event_count.gt(0).sum())} with positive xG',flush=True)
    summary=pd.DataFrame(summary_rows)
    comps=measure_and_compare(summary)
    games=pd.concat(game_frames,ignore_index=True)
    events=pd.concat(event_frames,ignore_index=True)
    write_csv(pd.DataFrame(manifest),'source_file_inventory.csv')
    write_csv(summary,'season_pbp_xg_scale.csv')
    write_csv(events,'event_type_xg_composition.csv')
    write_csv(games,'per_game_raw_xg_components.csv')
    write_csv(comps,'early_season_xg_compression_stages.csv')
    write_text(findings(comps,summary,n_dates))


if __name__=='__main__':
    try:main()
    except (ValueError, RuntimeError, ImportError, KeyError, OSError, TypeError) as exc:
        sys.exit('ERROR: '+str(exc))
