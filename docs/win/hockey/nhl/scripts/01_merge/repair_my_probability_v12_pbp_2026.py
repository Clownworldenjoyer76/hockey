#!/usr/bin/env python3
"""V12: verify and selectively repair stale 2026-27 SDV-P5 play-by-play.

A single --repair run audits the local PBP, fetches a fresh SDV 2027 PBP,
normalizes it with the repository's existing P5 builder, and adopts it ONLY if
coverage of independently completed NHL games is at least 95%, at least 20
additional completed game IDs are supplied, all old completed IDs remain,
source dates match the official P5 schedule, and every adopted completed game
has positive xG events. No NHL model is fitted. V5 and production unchanged.

Writes audit, manifest, and original-file backup under:
 NHL/my_probability/independent_v12_pbp_repair/
If quality gates fail, the original P5 dataset remains untouched.

Usage: python repair_my_probability_v12_pbp_2026.py --repair
       python repair_my_probability_v12_pbp_2026.py --self-test
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

YEAR = 2026
SEASON = '2026_2027'
MIN_COVERAGE = 0.95
MIN_ADDED_GAMES = 20
VERSION = 'v12_20261120_2_official_scores_fix'


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def sha256(path: Path) -> str:
    dig = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            dig.update(block)
    return dig.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)


def canonical(value: object) -> str:
    v = str(value).strip()
    if v.endswith('.0') and v[:-2].isdigit():
        v = v[:-2]
    require(bool(re.fullmatch(r'\d{10}', v)), f'Unusable official game_id: {v[:40]!r}')
    return v


def completed_official_ids(folder: Path, *, minimum_games: int = 40) -> tuple[set[str], dict[str, str], int]:
    """Read official NHL final scores transformed by this repository's Stage 05.

    The V5 official_schedule_cache contains only 2022–24 seasons, so it cannot
    serve as a current-season final-score source. Final-score CSVs are the
    independently retrieved completed-game artifacts already used in grading.
    Every game must be a final regular-season result with an unambiguous date.
    """
    require(folder.is_dir(), f'Missing official final-score directory: {folder}')
    paths = sorted(folder.glob('2026_*_NHL_final_scores.csv'))
    require(paths, f'No 2026-27 official final-score files in {folder}')
    ids: set[str] = set()
    dates: dict[str, str] = {}
    for path in paths:
        filename_day = path.name[:10]
        require(bool(re.fullmatch(r'2026_\d{2}_\d{2}', filename_day)),
                f'Invalid final-score filename date: {path}')
        try:
            datetime.strptime(filename_day, '%Y_%m_%d')
        except ValueError as exc:
            raise ValueError(f'Invalid final-score filename date: {path}') from exc
        with path.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or ())
            require({'game_id', 'home_score', 'away_score', 'game_date'}.issubset(cols),
                    f'Official final-score file missing columns: {path}')
            for row in reader:
                gid = canonical(row['game_id'])
                if not gid.startswith(f'{YEAR}02'):
                    # Ignore preseason and playoffs. Only regular-season games
                    # are eligible for this model's completion gate.
                    continue
                stamp = str(row['game_date']).strip().replace('-', '_')
                require(stamp == filename_day,
                        f'Official result date disagrees with file: {gid} {stamp} in {path.name}')
                try:
                    datetime.strptime(stamp, '%Y_%m_%d')
                    h, away = int(row['home_score']), int(row['away_score'])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'Unverifiable final score/date: {gid}') from exc
                require(h >= 0 and away >= 0 and h != away,
                        f'Invalid final game score (including tie): {gid}')
                require(gid not in ids, f'Duplicate final-score game across files: {gid}')
                ids.add(gid)
                dates[gid] = stamp.replace('_', '-')
    require(len(ids) >= minimum_games,
            f'Official regular-season completed-game results too small ({len(ids)} games)')
    return ids, dates, len(paths)


def quality_gate(*, old_ids: set[str], new_ids: set[str], completed: set[str],
                 invalid_dates: int, newly_positive_xg: set[str], schedule_missing: set[str]) -> tuple[bool, list[str], dict]:
    old_matched = old_ids & completed
    new_matched = new_ids & completed
    added = new_matched - old_matched
    vanished = old_matched - new_matched
    fraction = len(new_matched) / len(completed)
    reasons=[]
    if schedule_missing:
        reasons.append('P5 schedule missing completed game IDs')
    if invalid_dates:
        reasons.append('Source date mismatch/missing source date')
    if vanished:
        reasons.append('New source lost previously covered completed games')
    if len(added) < MIN_ADDED_GAMES:
        reasons.append('Fresh source has insufficient newly covered games')
    if fraction < MIN_COVERAGE:
        reasons.append('Fresh source covers fewer than 95% of completed games')
    if new_matched - newly_positive_xg:
        reasons.append('Fresh PBP lacks positive xG events for completed games')
    return not reasons, reasons, {
        'completed_games': len(completed),
        'old_matched_completed': len(old_matched),
        'new_matched_completed': len(new_matched),
        'newly_matched_completed': len(added),
        'lost_completed': len(vanished),
        'fresh_completed_coverage': round(fraction, 6),
        'completed_missing_from_p5_schedule': sorted(schedule_missing)[:50],
        'newly_covered_sample': sorted(added)[:30],
        'lost_sample': sorted(vanished)[:30],
        'fresh_missing_positive_xg_sample': sorted(new_matched - newly_positive_xg)[:30],
        'bad_date_count': invalid_dates,
    }


def self_test():
    completed={f'202602{i:04d}' for i in range(100)}
    old=set(sorted(completed)[:7])
    new=set(completed)
    accepted,reasons,stats=quality_gate(old_ids=old,new_ids=new,completed=completed,
                                        invalid_dates=0,newly_positive_xg=new,schedule_missing=set())
    require(accepted and not reasons and stats['newly_matched_completed']==93,'Full refresh rejected')
    accepted,reasons,_=quality_gate(old_ids=old,new_ids=old,completed=completed,
                                    invalid_dates=0,newly_positive_xg=old,schedule_missing=set())
    require(not accepted and len(reasons)>=2,'Stale refresh accepted')
    accepted,reasons,_=quality_gate(old_ids=old,new_ids=new,completed=completed,
                                    invalid_dates=1,newly_positive_xg=new,schedule_missing=set())
    require(not accepted and any('date' in s.lower() for s in reasons),'Wrong date accepted')
    accepted,reasons,_=quality_gate(old_ids=old,new_ids=new-old,completed=completed,
                                    invalid_dates=0,newly_positive_xg=new,schedule_missing=set())
    require(not accepted and any('lost' in s for s in reasons),'Lost historical game accepted')
    require(canonical('2026020001.0')=='2026020001','NHL ID normalization failed')
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp)
        f=folder/'2026_11_19_NHL_final_scores.csv'
        f.write_text('game_id,game_date,home_score,away_score\n'
                     '2026020347,2026_11_19,3,4\n'
                     '2026020348,2026_11_19,5,6\n', encoding='utf-8')
        parsed, dates, count=completed_official_ids(folder,minimum_games=2)
        require(len(parsed)==2 and dates['2026020347']=='2026-11-19' and count==1,
                'Stage05 final-score file loader failed')
        try:
            (folder/'2026_11_20_NHL_final_scores.csv').write_text(
                'game_id,game_date,home_score,away_score\n2026020347,2026_11_20,3,4\n',encoding='utf-8')
            completed_official_ids(folder,minimum_games=2)
        except ValueError as exc:
            require('Duplicate final-score game' in str(exc),'Wrong duplicate guard failure')
        else:
            raise AssertionError('Duplicate final-score game was accepted')
    print('SELF-TEST PASS | Stage05 score loader, duplicate detection, original coverage gates')


def import_builder(root: Path):
    research=root / 'scripts' / 'research'
    require((research/'build_sdv_history.py').is_file(),f'Missing repository builder: {research}')
    sys.path.insert(0, str(research))
    os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR','load_as_storage')
    try:
        import polars as pl
        import build_sdv_history as builder
    except ImportError as exc:
        raise RuntimeError(f'Requires repository dependencies (Polars, SportsDataverse): {exc}') from exc
    builder.enforce_sdv_pin()
    return pl, builder


def game_profile(pl, pbp, schedule_ids: set[str], schedule_dates: dict[str,str], complete: set[str], label: str) -> dict:
    require(set(['game_id','source_game_date','xg']).issubset(pbp.columns),
            f'{label}: required PBP fields missing')
    rows=pbp.select(
        pl.col('game_id').cast(pl.String).str.strip_chars().alias('game_id'),
        pl.col('source_game_date').cast(pl.String).alias('source_game_date'),
        pl.col('xg').cast(pl.Float64,strict=False).alias('xg')
    )
    # The builder must already have canonicalized the ID; do not use fuzzy matching.
    all_ids=set(rows.get_column('game_id').drop_nulls().to_list())
    bad_ids=[s for s in all_ids if not re.fullmatch(r'\d{10}',s)]
    require(not bad_ids, f'{label}: invalid official NHL game IDs: {bad_ids[:5]}')
    regular={s for s in all_ids if s.startswith('202602')}
    unknown=all_ids - schedule_ids
    dated=rows.filter(pl.col('game_id').is_in(list(all_ids))).select(['game_id','source_game_date'])
    minmax=dated.filter(pl.col('source_game_date').is_not_null()).get_column('source_game_date')
    day_by_id=dated.group_by('game_id').agg(pl.col('source_game_date').drop_nulls().unique().alias('days'))
    wrong=[]
    for rec in day_by_id.iter_rows(named=True):
        gid=rec['game_id']; days=rec['days']
        if gid in schedule_dates and (len(days)!=1 or days[0]!=schedule_dates[gid]):
            wrong.append(gid)
    # Old data is merely audited; the fresh source is always strictly validated.
    positive=rows.filter((pl.col('xg')>0) & (pl.col('game_id').is_in(list(complete))))
    positive_ids=set(positive.get_column('game_id').unique().to_list())
    return {
        'all_ids':all_ids,'regular_ids':regular,'unknown_schedule_ids':unknown,
        'invalid_source_date_ids':wrong,'positive_completed_ids':positive_ids,
        'event_rows':int(rows.height), 'regular_game_ids':len(regular),
        'latest_event_source_date':str(max(minmax)) if len(minmax) else None,
        'regular_completed_covered':len(regular & complete),
        'completed_positive_xg_covered':len(regular & complete & positive_ids),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repair',action='store_true',help='Fetch and adopt only a fully validated fresher 2026 PBP')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    root=Path(__file__).resolve().parents[2]
    require(root.name=='nhl', f'Expected script under NHL/scripts/01_merge, got {root}')
    output=root/'my_probability'/'independent_v12_pbp_repair'
    audit=output/'reports'/'repair_manifest.json'
    source=root/'research'/'sdv_history'
    schedule_path=source/'schedules'/f'season_{YEAR}.parquet'
    pbp_path=source/'play_by_play'/f'season_{YEAR}.parquet'
    official_path=root/'05_final_scores'/'final_scores'
    complete, official_dates, completed_files=completed_official_ids(official_path)
    pl,builder=import_builder(root)
    require(schedule_path.is_file() and pbp_path.is_file(),
            f'Missing local P5 schedule/PBP. Expected {schedule_path} and {pbp_path}')
    schedule=builder.prepare_schedule(pl.read_parquet(schedule_path),YEAR)
    date_map={str(g):str(d) for g,d in schedule.select(['game_id','source_game_date']).iter_rows()}
    schedule_ids=set(date_map)
    date_conflicts={gid:(day, date_map.get(gid)) for gid,day in official_dates.items()
                    if gid in date_map and date_map[gid]!=day}
    require(not date_conflicts,
            f'Official final-score game date conflicts with P5 schedule: {list(date_conflicts.items())[:5]}')
    existing=pl.read_parquet(pbp_path)
    old=game_profile(pl,existing,schedule_ids,date_map,complete,'cached')
    report={'version':VERSION,'season':SEASON,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'mode':'guarded_refresh' if args.repair else 'audit_only',
            'completed_score_source_directory':str(official_path),
            'completed_score_source':'stage05_official_nhl_final_scores',
            'completed_score_files':completed_files, 'completed_score_games':len(complete),
            'p5_schedule':str(schedule_path),'p5_pbp':str(pbp_path),
            'p5_old_sha256':sha256(pbp_path),
            'old_pbp':{k:v for k,v in old.items() if not isinstance(v,set) and not k.endswith('_ids')},
            'old_schedule_unknown_game_ids_sample':sorted(old['unknown_schedule_ids'])[:20],
            'status':'audit_only_no_change'}
    print(f'BEFORE | official completed={len(complete)} | existing P5 completed={old["regular_completed_covered"]} | latest PBP date={old["latest_event_source_date"]}',flush=True)
    if not args.repair:
        atomic_json(audit,report)
        print(f'WROTE {audit}\nNO CHANGES. Rerun with --repair to fetch and safely validate a fresh PBP.')
        return
    missing=complete-schedule_ids
    if missing:
        report['status']='blocked_incomplete_p5_schedule'
        report['schedule_missing_completed_sample']=sorted(missing)[:50]
        atomic_json(audit,report)
        raise ValueError(f'P5 schedule lacks {len(missing)} completed IDs; cannot safely rebuild from cached schedule')
    print('FETCH | SportsDataverse 2027 PBP; original P5 file untouched',flush=True)
    spec=builder.SPEC_BY_NAME['play_by_play']
    raw=builder.load_sdv_frame(spec,YEAR)
    require(not raw.is_empty(),'SDV returned an empty PBP: old dataset preserved')
    # Same normalization and source-date guard as the repository history builder.
    probe,_=builder.recover_legacy_game_id(raw,spec=spec,season=YEAR,schedule=schedule)
    require('game_id' in probe.columns,'Fresh PBP lacks canonical game_id')
    candidate,_,_=builder.attach_source_dates(probe,spec=spec,season=YEAR,
                                              schedule=schedule,schedule_dates=builder.schedule_date_lookup(schedule))
    builder.validate_pbp_research_columns(candidate)
    fresh=game_profile(pl,candidate,schedule_ids,date_map,complete,'fresh')
    accept, reasons, numbers=quality_gate(
        old_ids=old['regular_ids'],new_ids=fresh['regular_ids'],completed=complete,
        invalid_dates=len(fresh['invalid_source_date_ids'])+len(fresh['unknown_schedule_ids']),
        newly_positive_xg=fresh['positive_completed_ids'],schedule_missing=missing)
    report['fresh_pbp']={k:v for k,v in fresh.items() if not isinstance(v,set) and not k.endswith('_ids')}
    report['fresh_unknown_schedule_game_ids_sample']=sorted(fresh['unknown_schedule_ids'])[:20]
    report['fresh_bad_source_date_game_ids_sample']=fresh['invalid_source_date_ids'][:20]
    report['quality_gates']=numbers
    report['rejection_reasons']=reasons
    print(f'FRESH | completed={numbers["new_matched_completed"]}/{len(complete)} '
          f'| added={numbers["newly_matched_completed"]} '
          f'| lost={numbers["lost_completed"]} '
          f'| positive_xG={fresh["completed_positive_xg_covered"]}',flush=True)
    if not accept:
        report['status']='rejected_old_p5_preserved'
        atomic_json(audit,report)
        print('REJECT | original P5 unchanged | '+ '; '.join(reasons),flush=True)
        print(f'WROTE {audit}')
        return
    # No source mutation until the refresh is completely normalized and vetted.
    staging=output/'staging'/f'season_{YEAR}_play_by_play.parquet'
    staging.parent.mkdir(parents=True,exist_ok=True)
    builder.write_parquet_atomic(candidate,staging)
    staged=game_profile(pl,pl.read_parquet(staging),schedule_ids,date_map,complete,'staged')
    require(staged['regular_ids']==fresh['regular_ids'] and staged['event_rows']==fresh['event_rows'],
            'Round-trip Parquet profile changed; original untouched')
    backup=output/'backups'/f'season_{YEAR}_play_by_play_prior_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.parquet'
    backup.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(pbp_path,backup)
    require(sha256(backup)==report['p5_old_sha256'],'Backup verification failed; old dataset unchanged')
    report['backup_path']=str(backup)
    report['backup_sha256']=sha256(backup)
    # Staging and original live on same G: filesystem in the standard layout.
    try:
        staging.replace(pbp_path)
        adopted=game_profile(pl,pl.read_parquet(pbp_path),schedule_ids,date_map,complete,'adopted')
        require(adopted['regular_ids']==fresh['regular_ids'] and adopted['event_rows']==fresh['event_rows'],
                'Post-adoption verification failed')
    except Exception:
        shutil.copy2(backup,pbp_path)
        report['status']='failed_after_staging_restored_backup'
        atomic_json(audit,report)
        raise
    report['new_p5_sha256']=sha256(pbp_path)
    report['status']='validated_p5_2026_pbp_refreshed'
    atomic_json(audit,report)
    print(f'ADOPTED | {pbp_path}\nBACKUP | {backup}\nWROTE {audit}',flush=True)
    print('V8 cache fingerprint will trigger xG regeneration on next V8 execution; V5 unchanged.',flush=True)
    print('SOURCE LIMIT: retrospective PBP does not independently authenticate pregame capture timestamps.',flush=True)


if __name__=='__main__':
    try:
        main()
    except (ValueError,RuntimeError,OSError,ImportError,KeyError,TypeError) as exc:
        sys.exit('ERROR: '+str(exc))
