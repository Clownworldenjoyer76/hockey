#!/usr/bin/env python3
"""Immutable prospective NHL my_probability observation and postgame grading.

Install, and only execute, via a user-run PowerShell command to:
 G:\\hockey\\docs\\win\\hockey\\nhl\\scripts\\01_merge\\prospective_my_probability.py

Usage from any working directory:
 python prospective_my_probability.py record
 python prospective_my_probability.py record --date YYYY_MM_DD
 python prospective_my_probability.py grade

Record prerequisites: pre-existing merged game data, v1+v3 predictors, and
valid pregame odds timestamps. v3 is invoked only AFTER initial pregame checks.
Capture time must precede puck drop, quotes cannot be stale or future-dated,
source CSV cannot change during calculation, and model training history must
end strictly before the scheduled game date. A copied row of the merged inputs
is preserved beside every prediction for future independent feature audits.

All persistent new files are restricted to NHL/my_probability/prospective/.
v3's existing *isolated* research outputs are updated by its --only predict
invocation; production pipeline artifacts and configuration are never modified.
No sportsbook activity, selections, or betting is performed.

Gradings are postgame summaries of predictions captured before puck drop. No ROI is calculated, since a source timestamp does not verify quote
executability. No thresholds are searched or optimized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MERGE = ROOT / '01_merge'
FINAL = ROOT / '05_final_scores' / 'final_scores'
SCRIPT_DIR = ROOT / 'scripts' / '01_merge'
V1 = SCRIPT_DIR / 'predict_my_probability.py'
V3 = SCRIPT_DIR / 'predict_my_probability_v3.py'
OUT = ROOT / 'my_probability' / 'prospective'
PRED = ROOT / 'my_probability' / 'v3' / 'predictions'
NY = ZoneInfo('America/New_York')
UTC = timezone.utc
PREDICT_FIELDS = (
    'game_id', 'game_date', 'home_team', 'away_team',
    'my_probability_home', 'my_probability_away',
    'raw_v3_probability_home', 'vig_free_home_prob',
    'home_decimal', 'away_decimal', 'market_blend_weight',
    'model_version', 'max_source_result_date', 'forecast_type'
)
MERGED_FIELDS = (
    'game_id', 'game_date', 'game_time', 'home_team', 'away_team',
    'pulled_at', 'home_dk_moneyline_american', 'away_dk_moneyline_american',
)


def required(frame: pd.DataFrame, cols, label):
    missing = sorted(set(cols) - set(frame.columns))
    if missing:
        raise ValueError(f'{label}: missing {missing}')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canon(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame['game_id'] = frame['game_id'].astype('string').str.strip()
    if (frame['game_id'].isna().any()
        or not frame['game_id'].str.fullmatch(r'\d{10}').all()
        or frame['game_id'].duplicated().any()):
        raise ValueError('Game IDs must be unique official 10-digit IDs')
    return frame


def read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, dtype={'game_id': 'string'}, low_memory=False)


def dates(frame: pd.DataFrame) -> pd.Series:
    raw = frame['game_date'].astype('string').str.replace('_', '-', regex=False)
    result = pd.to_datetime(raw + ' ' + frame['game_time'].astype('string'), errors='coerce')
    if result.isna().any():
        raise ValueError('Invalid game_date/game_time')
    return result.dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')


def as_utc(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col.astype('string').str.strip(), utc=True, errors='coerce', format='mixed')


def check_rows(frame: pd.DataFrame, captured_at: pd.Timestamp, max_age_hours: float):
    frame = canon(frame)
    required(frame, MERGED_FIELDS, 'Stage 01 merged')
    frame = frame.copy()
    frame['_game_start_utc'] = dates(frame)
    frame['_odds_pulled_utc'] = as_utc(frame['pulled_at'])
    frame['_reason'] = ''
    for idx, row in frame.iterrows():
        start = row['_game_start_utc']
        pulled = row['_odds_pulled_utc']
        reason = None
        if pd.isna(start):
            reason = 'invalid_game_start'
        elif captured_at >= start:
            reason = 'capture_not_pregame'
        elif pd.isna(pulled):
            reason = 'missing_odds_timestamp'
        elif pulled > captured_at:
            reason = 'future_dated_odds_pull'
        elif pulled >= start:
            reason = 'odds_not_pregame'
        elif (captured_at - pulled).total_seconds() > max_age_hours * 3600:
            reason = 'odds_stale'
        else:
            try:
                h = float(row['home_dk_moneyline_american'])
                a = float(row['away_dk_moneyline_american'])
                if not (math.isfinite(h) and math.isfinite(a) and h != 0 and a != 0):
                    reason = 'invalid_moneyline'
            except (TypeError, ValueError, OverflowError):
                reason = 'invalid_moneyline'
        if reason is not None:
            frame.at[idx, '_reason'] = reason
    return frame


def snapshot_id(moment: datetime, file_digest: str) -> str:
    return moment.strftime('%Y%m%dT%H%M%S%fZ') + '_' + file_digest[:12]


def record(date_arg: str | None, max_age_hours: float, no_xg: bool):
    now = datetime.now(UTC)
    date_label = date_arg or now.astimezone(NY).strftime('%Y_%m_%d')
    try:
        day = datetime.strptime(date_label, '%Y_%m_%d').date()
    except ValueError as exc:
        raise ValueError('Date must use YYYY_MM_DD') from exc
    if day > now.astimezone(NY).date():
        raise ValueError('Future date not yet observable; wait for date of game')
    src = MERGE / f'{date_label}_NHL_merged.csv'
    before_hash = sha(src)
    staged = read(src)
    if staged.empty:
        raise ValueError('Empty merged slate')
    required(staged, MERGED_FIELDS, str(src))
    if not staged['game_date'].astype(str).eq(date_label).all():
        raise ValueError('Merged file contains games outside target date')
    initial = check_rows(staged, pd.Timestamp(now), max_age_hours)
    accepted = initial.loc[initial['_reason'].eq('')]
    print('PRECHECK', f'date={date_label}', f'games={len(initial)}',
          f'eligible={len(accepted)}', f'excluded={len(initial)-len(accepted)}')
    if accepted.empty:
        raise RuntimeError('No future games with fresh pregame odds; no snapshot saved')
    for key, count in initial.loc[initial['_reason'].ne(''), '_reason'].value_counts().items():
        print('EXCLUDED', f'{key}={count}')
    for needed in (V1, V3):
        if not needed.is_file():
            raise FileNotFoundError(f'Missing model dependency {needed}')
    # v3 has prior-day target leakage guards; --only predict skips report rewrite.
    command = [sys.executable, str(V3), '--only', 'predict', '--date', date_label]
    if no_xg:
        command.append('--skip-xg')
    print('RUN', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT.parents[3], check=True)
    pred_file = PRED / f'{date_label}_NHL_my_probability_v3.csv'
    produced = canon(read(pred_file))
    required(produced, PREDICT_FIELDS, str(pred_file))
    if not produced['game_date'].astype(str).eq(date_label).all():
        raise ValueError('V3 returned wrong forecast date')
    if sha(src) != before_hash:
        raise RuntimeError('Merged source changed during model execution; no snapshot saved')
    # Recheck the actual capture time, not model-start time.
    captured = datetime.now(UTC)
    current = check_rows(staged, pd.Timestamp(captured), max_age_hours)
    valid = current.loc[current['_reason'].eq('')].copy()
    if valid.empty:
        raise RuntimeError('All games became ineligible during prediction; no snapshot saved')
    valid = valid.drop(columns=['_game_start_utc', '_odds_pulled_utc', '_reason'])
    matched = valid.merge(produced, on='game_id', how='left', validate='one_to_one',
                          suffixes=('', '_prediction'), indicator=True)
    if not matched['_merge'].eq('both').all():
        raise ValueError('Predictions missing for eligible games')
    for key in ('game_date', 'home_team', 'away_team'):
        if not matched[key].astype('string').eq(matched[f'{key}_prediction'].astype('string')).all():
            raise ValueError(f'Prediction/source mismatch in {key}')
    if not matched['forecast_type'].astype(str).eq('future_fixture_unverified_latest_price').all():
        raise ValueError('V3 contains historical reconstruction, not a future forecast')
    before = pd.to_datetime(matched['max_source_result_date'], errors='coerce')
    targets = pd.to_datetime(matched['game_date'].astype(str).str.replace('_', '-',regex=False),errors='coerce')
    if before.isna().any() or not (before < targets).all():
        raise ValueError('Model training cutoff is not strictly prior to target date')
    p = pd.to_numeric(matched['my_probability_home'],errors='raise').to_numpy(float)
    q = pd.to_numeric(matched['my_probability_away'],errors='raise').to_numpy(float)
    raw = pd.to_numeric(matched['raw_v3_probability_home'],errors='raise').to_numpy(float)
    if not (np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(raw).all()
            and ((p >= 0) & (p <= 1)).all() and ((raw >= 0) & (raw <= 1)).all()
            and np.allclose(p + q, 1., rtol=0, atol=1e-8)):
        raise ValueError('Invalid final probability values')
    # Preserve original merged source columns under source_ prefix for auditing.
    source_columns = list(staged.columns)
    records = pd.DataFrame({f'source_{key}': matched[key] for key in source_columns})
    for key in produced.columns:
        if key in ('game_id', 'game_date', 'home_team', 'away_team'):
            records[key] = matched[key]
        else:
            records['prediction_' + key] = matched[key]
    records.insert(0, 'captured_at_utc', captured.isoformat())
    records.insert(0, 'snapshot_id', snapshot_id(captured, before_hash))
    # Data provenance: pulled_at is source-reported, not independently authenticated.
    records['capture_before_game_start'] = True
    records['source_quote_timestamp_pre_capture'] = True
    records['quote_executability_independently_verified'] = False
    records['starter_identity_pregame_verified'] = False
    subdir = OUT / 'snapshots' / date_label
    subdir.mkdir(parents=True, exist_ok=True)
    name = records['snapshot_id'].iloc[0]
    outfile = subdir / f'{name}.csv'
    manifest = subdir / f'{name}.json'
    # Refuse overwrite of immutable observations.
    with outfile.open('x', encoding='utf-8', newline='') as stream:
        records.to_csv(stream,index=False)
    meta = dict(
        snapshot_id=name, captured_at_utc=captured.isoformat(),
        game_date=date_label, eligible_games=len(records),
        excluded_games=len(current)-len(records),
        exclusion_reasons=current.loc[current['_reason'].ne(''), '_reason'].value_counts().to_dict(),
        source_file=str(src), source_sha256=before_hash,
        model_file=str(V3), model_sha256=sha(V3),
        model_dependency_sha256=sha(V1), produced_prediction_sha256=sha(pred_file),
        snapshot_sha256=sha(outfile), max_quote_age_hours=max_age_hours,
        quote_executability_independently_verified=False,
        evaluation_status='prospective_ungraded',
    )
    with manifest.open('x',encoding='utf-8') as stream:
        json.dump(meta,stream,indent=2,sort_keys=True)
    print('RECORDED', outfile, f'games={len(records)}')
    print('MANIFEST', manifest)
    print('NO BETTING INTEGRATION. No claims of executable odds.')


def prob_loss(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p,dtype=float),1e-8,1.-1e-8)
    y = np.asarray(y,dtype=float)
    return float(np.mean((p-y)**2)), float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))


def grade():
    files = sorted((OUT / 'snapshots').glob('*/*.csv'))
    if not files:
        raise FileNotFoundError('No prospective snapshots to grade')
    frames=[]
    for f in files:
        mf=f.with_suffix('.json')
        if not mf.is_file():
            raise ValueError(f'Unmatched snapshot manifest: {f}')
        meta=json.loads(mf.read_text(encoding='utf-8'))
        if sha(f) != meta.get('snapshot_sha256'):
            raise ValueError(f'Prospective snapshot mutated: {f}')
        frame=canon(read(f))
        required(frame,['captured_at_utc','source_game_time','game_date',
                        'prediction_my_probability_home',
                        'prediction_raw_v3_probability_home',
                        'prediction_vig_free_home_prob'],str(f))
        frame['snapshot_file']=str(f)
        frames.append(frame)
    data=pd.concat(frames,ignore_index=True)
    # If several snapshots are available for one game, evaluate the first only;
    # the rule is fixed in advance, not chosen after seeing the final score.
    data=data.sort_values(['captured_at_utc','game_id']).drop_duplicates('game_id',keep='first')
    final_files=sorted(FINAL.glob('*_NHL_final_scores.csv'))
    if not final_files:
        raise FileNotFoundError('No final-score files available')
    final=pd.concat([read(p) for p in final_files],ignore_index=True)
    required(final,['game_id','game_date','home_score','away_score'],str(FINAL))
    final=canon(final)
    scored = data.merge(final[['game_id','game_date','home_score','away_score']],on='game_id',how='left',validate='one_to_one',suffixes=('', '_result'))
    scored['home_score']=pd.to_numeric(scored['home_score'],errors='coerce')
    scored['away_score']=pd.to_numeric(scored['away_score'],errors='coerce')
    valid=scored['home_score'].notna() & scored['away_score'].notna()
    graded=scored.loc[valid].copy()
    if graded.empty:
        print('NO COMPLETED PROSPECTIVE GAMES; no grading artifacts changed')
        return
    if (graded['home_score'] == graded['away_score']).any():
        raise ValueError('Tied binary final game encountered')
    graded['actual_home_win']=(graded['home_score']>graded['away_score']).astype(int)
    report=[]
    for label,col in [
        ('vig_free_market','prediction_vig_free_home_prob'),
        ('raw_v3','prediction_raw_v3_probability_home'),
        ('guarded_v3','prediction_my_probability_home'),
    ]:
        values=pd.to_numeric(graded[col],errors='raise').to_numpy(float)
        if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
            raise ValueError(f'Invalid prospective probability {col}')
        brier, logloss=prob_loss(graded['actual_home_win'].to_numpy(float),values)
        report.append({'model':label,'completed_games':len(graded),
                       'brier':brier,'log_loss':logloss,
                       'odds_executability_independently_verified':False})
    report_df=pd.DataFrame(report)
    out=OUT/'reports'; out.mkdir(parents=True,exist_ok=True)
    # Grading reports are derived/rebuildable and may be updated; snapshots cannot.
    graded[['game_id','game_date','captured_at_utc','actual_home_win',
            'prediction_vig_free_home_prob','prediction_raw_v3_probability_home',
            'prediction_my_probability_home','snapshot_file']].to_csv(out/'prospective_graded_games.csv',index=False)
    report_df.to_csv(out/'prospective_performance.csv',index=False)
    print(report_df.to_string(index=False))
    print('REPORT',out/'prospective_performance.csv')
    print('CAUTION: Quote availability and executability not independently verified.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    rec=sub.add_parser('record',help='Capture today\'s actual pregame model and merged features')
    rec.add_argument('--date',help='YYYY_MM_DD, default local NY current date')
    rec.add_argument('--max-odds-age-hours',type=float,default=12.0)
    rec.add_argument('--skip-xg',action='store_true',help='Skip optional local historical xG')
    sub.add_parser('grade',help='Compare immutable early snapshots with completed scores')
    args=parser.parse_args()
    if args.action=='record':
        if not (0.1<=args.max_odds_age_hours<=48):
            parser.error('Allowed odds age 0.1..48 hours')
        record(args.date,args.max_odds_age_hours,args.skip_xg)
    else:
        grade()


if __name__ == '__main__':
    try:
        main()
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc
