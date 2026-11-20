#!/usr/bin/env python3
"""Independent V5 historical feature-readiness and timestamp-contract audit.

Examines local caches, SportsDataverse P5 inputs, original as-of goalie/lineup
source CSVs, and merged game files. This is not a model, data pull, or training
script. It refuses to call a reconstructed/cutoff-stamped observation an
authenticated historical pregame capture. Does not change production or V5.

Usage: python audit_my_probability_v7_features.py [--root path]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

SEASONS = (2022, 2023, 2024, 2025, 2026)
TYPES = ('xg', 'goalie', 'lineup')
CUTOFF_KIND = {'goalie':'goalie', 'lineup':'lineup'}
SOURCE_GLOBS = {
    'goalie': ('sdv/goalie/season_*_game_goalie_features_asof.csv', 'sdv/goalie/latest_game_goalie_features_asof.csv'),
    'lineup': ('sdv/lineup-strength/season_*_game_lineup_features_asof.csv', 'sdv/lineup-strength/latest_game_lineup_features_asof.csv'),
}
P5 = ('schedules','play_by_play','shifts','goalie_boxscores','game_rosters')


def required_columns(df, needed, location):
    absent = set(needed) - set(df.columns)
    if absent:
        raise ValueError(f'{location}: missing required columns {sorted(absent)}')


def read_csv(path, needed=()):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    required_columns(df, needed, path)
    return df


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f'WROTE {path} | rows={len(df)}', flush=True)


def parse_utc(x):
    if x is None or not str(x).strip():
        return None
    try:
        d = datetime.fromisoformat(str(x).strip().replace('Z', '+00:00'))
        if d.tzinfo is None:
            return None
        return d.astimezone(timezone.utc)
    except (ValueError, OverflowError, TypeError):
        return None


def official_id(s):
    s = str(s).strip()
    if not re.fullmatch(r'20\d{8}', s):
        raise ValueError(f'Noncanonical official NHL game ID: {s!r}')
    return s


def year_for_id(s):
    return int(official_id(s)[:4])


def validate_date(s):
    s = str(s).strip()
    try:
        return datetime.strptime(s.replace('-','_'), '%Y_%m_%d').date()
    except ValueError as exc:
        raise ValueError(f'Invalid game date: {s!r}') from exc


def indexed_targets(root):
    entries = {}
    source_files = []
    for year in (2022, 2023, 2024):
        p = root/'my_probability'/'independent_v5'/'official_schedule_cache'/f'season_{year}_regular_games.csv'
        if p.is_file():
            source_files.append(p)
    evaluation = root/'my_probability'/'independent_v5'/'reports'/'chronological_2025_2026_evaluation.csv'
    if evaluation.is_file():
        source_files.append(evaluation)
    # Historical rows from merged games are a fallback; date equality checked.
    source_files.extend(sorted((root/'01_merge').glob('*_NHL_merged.csv')))
    for path in source_files:
        frame = read_csv(path, ('game_id', 'game_date'))
        if path.name == 'chronological_2025_2026_evaluation.csv' and 'data_period' not in frame.columns:
            raise ValueError('V5 evaluation lacks data_period; cannot assess holdout seasons')
        for game in frame.itertuples(index=False):
            gid = official_id(getattr(game,'game_id'))
            year = year_for_id(gid)
            if year not in SEASONS or gid[4:6] != '02':
                continue
            d = validate_date(getattr(game,'game_date'))
            prior = entries.get(gid)
            if prior is not None and prior['date'] != d:
                raise ValueError(f'Conflicting historical game dates: {gid}, {prior["date"]} vs {d}')
            entries[gid] = {'date':d, 'year':year}
    if not entries:
        raise RuntimeError('No official V5 history or merged slate data found. Run V5 first.')
    return entries


def p5_coverage(root, entries):
    out=[]
    for year in SEASONS:
        base=root/'research'/'sdv_history'
        files={kind:base/kind/f'season_{year}.parquet' for kind in P5}
        state={kind:files[kind].is_file() and files[kind].stat().st_size>0 for kind in P5}
        # From repository build_sdv_history.py: shifts begins SDV END year 2025.
        shifts_supported=(year+1 >= 2025)
        out.append({
            'season':f'{year}_{year+1}', 'target_games':sum(r['year']==year for r in entries.values()),
            **{f'{kind}_exists':state[kind] for kind in P5},
            **{f'{kind}_path':str(files[kind]) for kind in P5},
            'shifts_supported_by_research_builder':shifts_supported,
            'xg_prior_only_reconstruction_possible':state['schedules'] and state['play_by_play'],
            'goalie_prior_pbp_shifts_possible':shifts_supported and state['play_by_play'] and state['shifts'],
            'lineup_prior_pbp_shifts_possible':shifts_supported and state['play_by_play'] and state['shifts'],
            'pregame_capture_authenticated':False,
        })
    return pd.DataFrame(out)


def audit_snapshot(row, family):
    start = parse_utc(row.get('pregame_cutoff_utc'))
    decision = parse_utc(row.get(f'{family}_decision_cutoff_utc'))
    snapshot = parse_utc(row.get(f'{family}_snapshot_as_of_utc'))
    if start is None or decision is None or snapshot is None:
        return 'missing_or_invalid_timestamp'
    if decision != start-timedelta(minutes=60):
        return 'decision_cutoff_not_T_minus_60'
    if snapshot > decision:
        return 'snapshot_after_decision_cutoff'
    source = row.get(f'{family}_decision_cutoff_source','').strip()
    if source != 'fixed_60_minutes_before_puck_drop':
        # Check actual repo constants; nonconforming values must be inspected.
        return f'unknown_cutoff_source:{source[:50]}'
    if family=='goalie':
        for side in ('home','away'):
            obs = parse_utc(row.get(f'{side}_goalie_status_observed_at'))
            if obs is None: return 'missing_or_invalid_goalie_observation'
            if obs > snapshot: return 'goalie_observed_after_snapshot'
            if obs > decision: return 'goalie_observed_after_decision_cutoff'
    else:
        for side in ('home','away'):
            status=str(row.get(f'{side}_lineup_status','')).strip().lower()
            obs=parse_utc(row.get(f'{side}_lineup_observed_at'))
            # Unknown lineup status may have no usable observation.
            if status not in ('unknown','') and obs is None:
                return 'lineup_status_missing_observation'
            if obs is not None and obs > snapshot:
                return 'lineup_observed_after_snapshot'
            if obs is not None and obs > decision:
                return 'lineup_observed_after_decision_cutoff'
    return 'passes_stored_timestamp_contract'


def validate_source_files(root, entries):
    details=[]
    for family in ('goalie','lineup'):
        for pattern in SOURCE_GLOBS[family]:
            for path in sorted(root.glob(pattern)):
                df=read_csv(path, ('game_id','game_date','pregame_cutoff_utc',
                    f'{family}_decision_cutoff_utc', f'{family}_snapshot_as_of_utc',
                    f'{family}_decision_cutoff_source'))
                for row in df.to_dict('records'):
                    gid=official_id(row['game_id'])
                    yr=year_for_id(gid)
                    if yr not in SEASONS or gid[4:6]!='02':continue
                    d=validate_date(row['game_date'])
                    indexed=entries.get(gid)
                    result=audit_snapshot(row, family)
                    if indexed is not None and indexed['date']!=d:
                        result='date_disagrees_with_official_history'
                    metric = 'starter_gsax_differential' if family=='goalie' else 'skater_war_differential'
                    is_finite=pd.to_numeric(pd.Series([row.get(metric,'')]),errors='coerce').notna().iloc[0]
                    if family=='lineup':
                        # Count ANY one of the six lineup/player metric differentials.
                        is_finite=any(pd.notna(pd.to_numeric(pd.Series([row.get(k+'_differential','')]),errors='coerce').iloc[0]) for k in
                          ('skater_rapm','skater_war','pp_value','pk_value','forward_line_strength','defense_pair_strength'))
                    details.append({
                      'family':family, 'season':f'{yr}_{yr+1}', 'game_id':gid,
                      'in_v5_target_set':indexed is not None,'source_file':str(path),
                      'date':d.isoformat(), 'timestamp_contract':result,
                      'numeric_feature_present':bool(is_finite),
                      'home_status':row.get(f'home_{family}_status',''),
                      'away_status':row.get(f'away_{family}_status',''),
                      'status_source_home':row.get(f'home_{family}_status_source',''),
                      'status_source_away':row.get(f'away_{family}_status_source',''),
                      'snapshot_equals_T_minus_60':parse_utc(row.get(f'{family}_snapshot_as_of_utc'))==parse_utc(row.get(f'{family}_decision_cutoff_utc')),
                      # Strict timestamp stamps are not authenticated historical captures.
                      'historical_capture_authenticated':False,
                    })
    return pd.DataFrame(details)


def merged_coverage(root, entries):
    files=sorted((root/'01_merge').glob('*_NHL_merged.csv'))
    data=[]
    feats=('adj_xg_net_differential','starter_gsax_differential',
           'skater_rapm_differential','skater_war_differential',
           'pp_value_differential','pk_value_differential',
           'forward_line_strength_differential','defense_pair_strength_differential')
    for path in files:
        df=read_csv(path,('game_id','game_date'))
        for col in feats:
            if col not in df:df[col]=''
        for record in df.to_dict('records'):
            gid=official_id(record['game_id']);year=year_for_id(gid)
            if year not in SEASONS or gid[4:6]!='02':continue
            if gid in entries and validate_date(record['game_date'])!=entries[gid]['date']:
                raise ValueError(f'Merged/official game date disagreement: {gid}')
            item={'season':f'{year}_{year+1}','game_id':gid,'source_file':str(path)}
            for col in feats:
                item[col+'_present']=pd.notna(pd.to_numeric(pd.Series([record[col]]),errors='coerce').iloc[0])
            item['goalie_status_known']=all(str(record.get(f'{s}_goalie_status','')).lower() not in ('','unknown') for s in ('home','away'))
            item['lineup_status_known']=all(str(record.get(f'{s}_lineup_status','')).lower() not in ('','unknown') for s in ('home','away'))
            data.append(item)
    return pd.DataFrame(data)


def summarise(entries,p5,snap,merged):
    rows=[]
    for year in SEASONS:
        season=f'{year}_{year+1}'
        targets={g for g,v in entries.items() if v['year']==year}
        sub=snap.loc[snap['season'].eq(season)] if not snap.empty else pd.DataFrame()
        merged_sub=merged.loc[merged['season'].eq(season)] if not merged.empty else pd.DataFrame()
        pv=p5.loc[p5.season.eq(season)].iloc[0]
        for feat in TYPES:
            x=sub.loc[sub.family.eq(feat)] if feat!='xg' and not sub.empty else pd.DataFrame()
            ok=x.loc[x.timestamp_contract.eq('passes_stored_timestamp_contract')] if not x.empty else pd.DataFrame()
            matching=ok.loc[ok.game_id.isin(targets)] if not ok.empty else pd.DataFrame()
            source_games=len(set(matching.game_id)) if not matching.empty else 0
            numeric_games=len(set(matching.loc[matching.numeric_feature_present,'game_id'])) if not matching.empty else 0
            merged_feat={'xg':'adj_xg_net_differential','goalie':'starter_gsax_differential','lineup':'skater_war_differential'}[feat]+'_present'
            observed_merged=int(merged_sub[merged_feat].sum()) if not merged_sub.empty else 0
            xg_possible=bool(pv.xg_prior_only_reconstruction_possible)
            if feat=='xg':
                evaluation='can_reconstruct_asof_only_from_P5' if xg_possible else 'missing_P5_schedule_or_PBP'
                numeric_games=0 # must not equate existence of parquet with feature derivation
            elif year<2024:
                evaluation='no_historical_shift_support_in_SDVP5_builder'
            elif not (pv.play_by_play_exists and pv.shifts_exists):
                evaluation='missing_P5_PBP_or_shifts'
            elif numeric_games==0:
                evaluation='no_pregame_numeric_snapshot_matching_V5_games'
            else:
                evaluation='candidate_prior_only_research_requires_coverage_and_lineage'
            rows.append({'season':season,'feature':feat,'v5_target_games':len(targets),
                'p5_inputs_exist': bool(pv.schedules_exists and pv.play_by_play_exists) if feat=='xg' else bool(pv.play_by_play_exists and pv.shifts_exists),
                'source_snapshot_rows':len(x),'passing_snapshot_rows':len(ok),
                'matching_games_with_timestamp_contract':source_games,'matching_games_with_numeric':numeric_games,
                'daily_merged_rows':len(merged_sub), 'merged_rows_with_numeric':observed_merged,
                'snapshot_capture_authenticated':False,'readiness':evaluation})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2],help='NHL root (defaults to this script location)')
    args=parser.parse_args()
    root=args.root.resolve()
    out=root/'my_probability'/'independent_v7_feature_audit'/'reports'
    target=indexed_targets(root)
    p5=p5_coverage(root,target)
    snapshots=validate_source_files(root,target)
    merged=merged_coverage(root,target)
    coverage=summarise(target,p5,snapshots,merged)
    write(p5,out/'p5_historical_source_inventory.csv')
    write(snapshots,out/'goalie_lineup_asof_row_contracts.csv')
    write(merged,out/'merged_2026_feature_presence.csv')
    write(coverage,out/'historical_feature_readiness.csv')
    errors= snapshots.loc[snapshots.timestamp_contract.ne('passes_stored_timestamp_contract')] if not snapshots.empty else pd.DataFrame()
    action=[]
    for year in SEASONS:
        p=p5.loc[p5.season.eq(f'{year}_{year+1}')].iloc[0]
        action.append({
           'season':p.season,
           'xg_action':'build_P5_schedule_and_PBP_then_strict_prior_date_xg' if not p.xg_prior_only_reconstruction_possible else 'derive_strict_prior_date_xg_and_join_by_game_id',
           'goalie_action':'historical_shift_source_unavailable_in_P5_builder' if year<2024 else ('inspect_prior_PBP_shift_derived_goalie_and_original_snapshot_lineage' if p.goalie_prior_pbp_shifts_possible else 'source_PBP_and_shifts_before_ablation'),
           'lineup_action':'historical_shift_source_unavailable_in_P5_builder' if year<2024 else ('inspect_prior_PBP_shift_derived_lineup_and_original_snapshot_lineage' if p.lineup_prior_pbp_shifts_possible else 'source_PBP_and_shifts_before_ablation'),
           'model_training_changed':False,
        })
    write(pd.DataFrame(action),out/'reconstruction_actions.csv')
    findings=[
      f'V5 official game coverage: '+', '.join(f'{y}-{str(y+1)[-2:]}: {sum(v["year"]==y for v in target.values())}' for y in SEASONS),
      f'Historical source snapshot rows: {len(snapshots)}. Failing timestamp contracts: {len(errors)}.',
      'The SDV P5 history builder supports shifts from NHL start year 2024 (SDV END year 2025); 2022-23 and 2023-24 prior shift-driven goalie/lineup data cannot be reconstructed through this builder.',
      'Historical cutoff-labelled goalie/lineup snapshots are not independently authenticated as observed pregame; source code can label retrospections with the T-60 cutoff.',
      'P5 schedules and PBP can support xG as-of reconstruction only when actually present, using source_game_date strictly earlier than target_date.',
      'No feature entered V5. Do not fit xG, goalies or lineup by imputing entire older seasons to missing; that creates a season/feature availability confound.',
      'Next safe experiment: populate P5 schedules/PBP for 2022-24; rebuild strict prior-only xG across ALL training and validation seasons; select the xG ablation using 2024-25 only.',
      'Do not select a new model on 2025-27 holdout results and do not integrate with betting.',
    ]
    out.mkdir(parents=True,exist_ok=True)
    (out/'findings.txt').write_text('\n'.join(findings)+'\n',encoding='utf-8')
    print('\n'.join(findings),flush=True)
    print(f'WROTE {out/"findings.txt"}')

if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,KeyError) as e:sys.exit(f'ERROR: {e}')
