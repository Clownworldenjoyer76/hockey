#!/usr/bin/env python3
"""Independent NHL historical model, 2022-23 onward.

Run beside predict_my_probability.py, predict_my_probability_v3.py, and
predict_my_probability_independent.py. Uses prior finalized NHL scores, Elo,
rest, and recent form only. No sportsbook/provider prediction enters fitting.

Official NHL schedule API is also used by this repository's pull_nhl_schedule.py.
Past schedules are retrieved POST-game. They are NEVER interpreted as pregame
snapshots: scores only influence strictly LATER target dates.

The 2024-25 season selects hyperparameters; 2025-26 and 2026-27 are
chronologically evaluated without tuning on their results. 2026-27 scores
are already present in the local final_scores folder. Existing merged slate
is used for game IDs, names, and market benchmarking ONLY.

Outputs: NHL/my_probability/independent_v5/; no existing production writes.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

import predict_my_probability as v1
import predict_my_probability_v3 as v3
from predict_my_probability_independent import RidgeLogit, brier, log_loss

NHL = v1.NHL
OUT = NHL / 'my_probability' / 'independent_v5'
RAW = OUT / 'official_schedule_cache'
SRC = 'https://api-web.nhle.com/v1/schedule/{date}'
MODEL_ID = 'independent_v5_multiseason_20261119'
EPS = 1e-7
PENALTIES = (2.0, 15.0, 60.0, 200.0)
FEATURE_SETS = {
    'elo_only': ('elo_diff_asof',),
    'elo_and_rest': ('elo_diff_asof', 'rest_diff_asof', 'back_to_back_diff_asof', 'games_in_four_diff_asof'),
    'elo_form': ('elo_diff_asof', 'rest_diff_asof', 'back_to_back_diff_asof',
                 'games_in_four_diff_asof', 'win_rate_last10_diff_asof',
                 'goals_diff_last10_diff_asof', 'goals_for_last10_diff_asof',
                 'goals_against_last10_diff_asof', 'prior_games_min_asof'),
    'form_only': ('rest_diff_asof', 'back_to_back_diff_asof',
                  'games_in_four_diff_asof', 'win_rate_last10_diff_asof',
                  'goals_diff_last10_diff_asof', 'goals_for_last10_diff_asof',
                  'goals_against_last10_diff_asof', 'prior_games_min_asof'),
}


def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f'WROTE {path} | rows={len(frame)}', flush=True)


def localized(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get('default') or value.get('en') or '').strip()
    return ''


def team_id(team):
    """Use stable abbreviations; old ARI and new UTA are distinct until linked."""
    if not isinstance(team, dict):
        return ''
    return str(team.get('abbrev') or '').strip().upper()


def canonical_id(value):
    s = str(value).strip()
    if not re.fullmatch(r'\d{10}', s):
        raise ValueError(f'Invalid NHL official game ID: {s[:30]}')
    return s


def parse_completed(game, expected_season):
    if not isinstance(game, dict) or game.get('gameType') not in (2, '2'):
        return None
    gid = canonical_id(game.get('id'))
    if not gid.startswith(str(expected_season) + '02'):
        return None
    if str(game.get('gameState') or '').upper() not in ('OFF', 'FINAL'):
        return None
    home, away = game.get('homeTeam') or {}, game.get('awayTeam') or {}
    if not isinstance(home, dict) or not isinstance(away, dict):
        raise ValueError(f'Malformed NHL team block: {gid}')
    h, a = home.get('score'), away.get('score')
    if isinstance(h, bool) or isinstance(a, bool) or not isinstance(h, int) or not isinstance(a, int):
        # A completed schedule entry without both scores is not usable.
        return None
    if h == a or h < 0 or a < 0:
        raise ValueError(f'Impossible finalized score: {gid} {h}-{a}')
    day = str(game.get('gameDate') or '')
    try:
        dt = date.fromisoformat(day)
    except ValueError as exc:
        raise ValueError(f'Invalid gameDate on {gid}: {day}') from exc
    if not (date(expected_season, 9, 1) <= dt <= date(expected_season + 1, 7, 1)):
        raise ValueError(f'Game date outside expected season: {gid} {day}')
    ha, aa = team_id(home), team_id(away)
    if not ha or not aa or ha == aa:
        raise ValueError(f'Unusable team abbreviation: {gid} {ha}-{aa}')
    return {
        'game_id': gid, 'game_date': day.replace('-', '_'),
        'game_date_dt': pd.Timestamp(day),
        'data_period': f'{expected_season}_{expected_season+1}',
        'home_team_full': ha, 'away_team_full': aa,
        'home_team': ha, 'away_team': aa,
        'home_score': int(h), 'away_score': int(a),
        'target': float(h > a),
        'official_score_state': str(game.get('gameState')),
        'outcome_source': 'NHL_official_schedule_postgame',
        'pregame_prices_verified': False,
    }


def extract_games(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected an official NHL schedule JSON object')
    days = payload.get('gameWeek')
    if not isinstance(days, list):
        raise ValueError('NHL schedule API gameWeek missing or invalid')
    found = []
    for day in days:
        if not isinstance(day, dict):
            continue
        for game in day.get('games') or []:
            if isinstance(game, dict):
                found.append(game)
    return found


def fetch_week(session, day, offline=False):
    folder = RAW / 'weeks'
    target = folder / f'{day.isoformat()}.json'
    if target.is_file():
        with target.open('r', encoding='utf-8') as fh:
            payload = json.load(fh)
        extract_games(payload)
        return payload
    if offline:
        raise FileNotFoundError(f'Missing cached official NHL week: {target}')
    url = SRC.format(date=day.isoformat())
    last = None
    for attempt in range(3):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            extract_games(payload)
            folder.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix('.tmp')
            with temp.open('w', encoding='utf-8') as fh:
                json.dump(payload, fh, separators=(',', ':'))
            temp.replace(target)
            return payload
        except (requests.RequestException, ValueError, OSError) as exc:
            last = exc
            if attempt < 2:
                time.sleep(1 + attempt * 2)
    raise RuntimeError(f'Official NHL schedule request failed: {url}: {last}')


def load_old_season(session, season_start, offline=False):
    cache = RAW / f'season_{season_start}_regular_games.csv'
    if cache.is_file():
        saved = pd.read_csv(cache, dtype={'game_id': 'string'})
        validate_old(saved, season_start)
        saved['game_date_dt'] = pd.to_datetime(saved['game_date'].str.replace('_', '-', regex=False))
        return saved
    # Intervals of six days overlap official gameWeek responses (seven days).
    # The 1 Sep to 1 Jul window covers early/late season rescheduling.
    start, stop = date(season_start, 9, 1), date(season_start + 1, 7, 1)
    games, seen = {}, 0
    day = start
    while day <= stop:
        for game in extract_games(fetch_week(session, day, offline=offline)):
            record = parse_completed(game, season_start)
            if record is None:
                continue
            gid = record['game_id']
            if gid in games:
                old = games[gid]
                for key in ('home_score', 'away_score', 'home_team_full', 'away_team_full', 'game_date'):
                    if old[key] != record[key]:
                        raise ValueError(f'Conflicting official historical game {gid}: {key}')
            else:
                games[gid] = record
        seen += 1
        if seen % 12 == 0:
            print(f'OFFICIAL HISTORY {season_start}: fetched/cached {seen} week windows, {len(games)} games', flush=True)
        day += timedelta(days=6)
    out = pd.DataFrame(games.values())
    validate_old(out, season_start)
    save(out.drop(columns='game_date_dt'), cache)
    return out.sort_values(['game_date_dt', 'game_id']).reset_index(drop=True)


def validate_old(frame, season):
    required = {'game_id', 'game_date', 'home_team_full', 'away_team_full',
                'home_score', 'away_score', 'target', 'data_period'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Historical archive season {season} missing columns: {required-set(frame.columns)}')
    ids = frame['game_id'].astype('string')
    if ids.isna().any() or not ids.str.fullmatch(str(season) + r'02\d{4}').all() or ids.duplicated().any():
        raise ValueError(f'Historical archive season {season} has duplicate/invalid official game IDs')
    if len(frame) < 1100 or len(frame) > 1500:
        raise ValueError(f'Season {season} incomplete/unusual: {len(frame)} regular games (require 1100-1500); refusing fit')
    for side in ('home', 'away'):
        values = pd.to_numeric(frame[side+'_score'], errors='coerce')
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f'Season {season} has missing/invalid {side} goals')
    if (pd.to_numeric(frame['home_score']) == pd.to_numeric(frame['away_score'])).any():
        raise ValueError(f'Season {season} contains tied final scores')
    if not np.array_equal((pd.to_numeric(frame['home_score']) > pd.to_numeric(frame['away_score'])).astype(int),
                          pd.to_numeric(frame['target']).astype(int)):
        raise ValueError(f'Season {season} winner labels conflict with official final scores')
    if frame['data_period'].astype(str).ne(f'{season}_{season+1}').any():
        raise ValueError(f'Season {season} label mismatch')


def historical_and_current(older):
    local, slate = v1.load_history()
    local, slate, aliases = v3.scores_and_names(local, slate)
    # An abbreviation is a stable historical team identity; handle common
    # renamed abbreviations explicitly, without fuzzy team name matching.
    for frame in (local, slate):
        for side in ('home', 'away'):
            col = f'{side}_team_abbr'
            if frame[col].isna().any():
                raise ValueError(f'Missing official team abbreviation: {col}')
            frame[f'{side}_team_full'] = frame[col].astype(str).str.upper().replace({'STL': 'STL'})
    older = older.copy()
    older['game_date_dt'] = pd.to_datetime(older['game_date'].astype(str).str.replace('_', '-', regex=False))
    basecols = ['game_id', 'game_date', 'game_date_dt', 'data_period',
                'home_team_full', 'away_team_full', 'home_score', 'away_score', 'target']
    old = older[basecols].copy()
    new = local[basecols].copy()
    history = pd.concat([old, new], ignore_index=True).sort_values(['game_date_dt', 'game_id']).reset_index(drop=True)
    if history.game_id.duplicated().any():
        raise ValueError('Duplicate official game ID across older/current seasons')
    # Optional last-season market fields only enter reports, never learning.
    for frame in (local, slate):
        for field in ('vig_free_home_prob', 'home_decimal', 'away_decimal', 'drat_home_win_prob', 'sdv_home_win_prob'):
            if field not in frame:
                frame[field] = np.nan
    if slate.game_id.duplicated().any():
        raise ValueError('Duplicate target slate game IDs')
    # Build all pregame form AND Elo in one chronological pass. The v3 form
    # generator snapshots every game on a date BEFORE applying ANY same-day final.
    form = v3.build_pregame_form(history, slate)
    elo = prior_elo(history, slate)
    history = history.merge(form, on='game_id', validate='one_to_one', how='left').merge(elo, on='game_id', validate='one_to_one', how='left')
    slate = slate.merge(form, on='game_id', validate='one_to_one', how='left').merge(elo, on='game_id', validate='one_to_one', how='left')
    check = history[['form_cutoff_exclusive', 'game_date_dt']].copy()
    if (pd.to_datetime(check.form_cutoff_exclusive) != check.game_date_dt).any() or history['elo_diff_asof'].isna().any():
        raise ValueError('As-of feature cutoff mismatch')
    return history, slate, local


def prior_elo(history, current, k=20., offseason_retention=.45, home_advantage=65.):
    frames = [history, current]
    combined = (pd.concat([f[['game_id', 'game_date_dt', 'data_period',
                              'home_team_full', 'away_team_full', 'home_score',
                              'away_score', 'target']] for f in frames], ignore_index=True)
                .drop_duplicates('game_id').sort_values(['game_date_dt','game_id']))
    ratings = defaultdict(lambda: 1500.)
    last_season, records = None, []
    for day, games in combined.groupby('game_date_dt', sort=True):
        season = str(games.data_period.iloc[0])
        if last_season is not None and season != last_season:
            ratings = defaultdict(lambda:1500., {t: 1500.+offseason_retention*(v-1500.) for t,v in ratings.items()})
        last_season = season
        for row in games.itertuples(index=False):
            records.append({'game_id':row.game_id, 'elo_diff_asof': (ratings[row.home_team_full]-ratings[row.away_team_full])/400.,
                            'elo_feature_asof_rule':'results_before_target_calendar_day'})
        for row in games.itertuples(index=False):
            if not np.isfinite(row.target):
                continue
            delta = ratings[row.home_team_full]-ratings[row.away_team_full]
            exp_home = 1./(1.+10.**(-(delta+home_advantage)/400.))
            mov = abs(float(row.home_score)-float(row.away_score))
            change = k*(float(row.target)-exp_home)*(1.+.13*max(0., mov-1.))
            ratings[row.home_team_full] += change
            ratings[row.away_team_full] -= change
    output = pd.DataFrame(records)
    if output.game_id.duplicated().any() or len(output) != len(combined):
        raise ValueError('Elo output does not preserve one row per game')
    return output


def select_model(history):
    first = history.loc[history.data_period.isin(['2022_2023','2023_2024'])]
    valid = history.loc[history.data_period.eq('2024_2025')]
    if len(first) < 2200 or len(valid) < 1100:
        raise ValueError('Insufficient 2022-24 train or 2024-25 validation games')
    if first.game_date_dt.max() >= valid.game_date_dt.min():
        raise ValueError('Historical model selection temporal leakage')
    rows = []
    for name, cols in FEATURE_SETS.items():
        for pen in PENALTIES:
            p = RidgeLogit(cols, pen).fit(first).predict(valid)
            rows.append({'model':name, 'penalty':pen, 'training_games':len(first),
                         'train_max_date':first.game_date_dt.max().strftime('%Y-%m-%d'),
                         'validation_games':len(valid),
                         'validation_first_date':valid.game_date_dt.min().strftime('%Y-%m-%d'),
                         'brier':brier(valid.target,p), 'log_loss':log_loss(valid.target,p)})
    report = pd.DataFrame(rows).sort_values(['brier','log_loss','model','penalty'])
    winner = report.iloc[0]
    return report, str(winner.model), float(winner.penalty)


def evaluate_future(history, name, penalty):
    fields = ['game_id','game_date','game_date_dt','data_period','target','home_team_full','away_team_full']
    results = []
    for season in ('2025_2026', '2026_2027'):
        targets = history.loc[history.data_period.eq(season)]
        if targets.empty:
            raise ValueError(f'Missing completed chronological test season {season}')
        for i,(day, group) in enumerate(targets.groupby('game_date_dt', sort=True),1):
            prior = history.loc[history.game_date_dt < day]
            if len(prior) < 2000 or prior.game_date_dt.max() >= day:
                raise RuntimeError('Training history missing or includes same/future game results')
            predicted = RidgeLogit(FEATURE_SETS[name],penalty).fit(prior).predict(group)
            frame = group[fields].copy()
            frame['my_probability_home'] = predicted
            frame['my_probability_away'] = 1.-predicted
            frame['training_games_before_target'] = len(prior)
            frame['last_training_game_date'] = prior.game_date_dt.max().strftime('%Y-%m-%d')
            results.append(frame)
            if i % 45 == 0:
                print(f'CHRONOLOGICAL {season}: {i} game dates evaluated', flush=True)
    return pd.concat(results, ignore_index=True)


def report_metrics(pred, local, bootstrap_n=1000):
    reference = local[['game_id','vig_free_home_prob','drat_home_win_prob','sdv_home_win_prob','pregame_prices_verified']].drop_duplicates('game_id')
    out = pred.merge(reference,on='game_id',how='left',validate='one_to_one')
    rows=[]
    rng = np.random.default_rng(20261119)
    for season, group in out.groupby('data_period',sort=True):
        y = group.target.to_numpy(float)
        market = group.vig_free_home_prob.to_numpy(float)
        have_market=np.isfinite(market)
        for label,col in [('independent_v5','my_probability_home'),('vig_free_market','vig_free_home_prob'),
                          ('D_Ratings','drat_home_win_prob'),('SportsDataverse','sdv_home_win_prob')]:
            use=group[col].notna()
            if label == 'independent_v5':
                use = pd.Series(np.ones(len(group),bool),index=group.index)
            sub=group.loc[use]
            if len(sub)==0:
                continue
            values=sub[col].to_numpy(float)
            yy=sub.target.to_numpy(float)
            common=sub.loc[sub.vig_free_home_prob.notna()]
            advantage=np.nan;low=np.nan;high=np.nan
            if label=='independent_v5' and len(common):
                diff=(common.my_probability_home.to_numpy(float)-common.target.to_numpy(float))**2 - (common.vig_free_home_prob.to_numpy(float)-common.target.to_numpy(float))**2
                advantage=float(diff.mean())
                groups=[grp.index.to_numpy() for _,grp in common.reset_index(drop=True).groupby('game_date_dt')]
                vals=[]
                for _ in range(bootstrap_n):
                    indices=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))])
                    vals.append(float(diff[indices].mean()))
                low,high=np.quantile(vals,[.025,.975])
            rows.append({'season':season,'model':label,'games':len(sub),
                         'brier':round(brier(yy,values),7),'log_loss':round(log_loss(yy,values),7),
                         'brier_minus_market_paired':advantage,'paired_date_95ci_low':low,
                         'paired_date_95ci_high':high,
                         'market_prices_provenance':'2025_unverified' if season=='2025_2026' else '2026_source_timestamp_checked_not_independently_authenticated',
                         'used_market_or_provider_inputs_in_model':False})
    return out, pd.DataFrame(rows)


def forecast(history, current, name, penalty, selected_date):
    frame=current.loc[current.game_date_dt.eq(selected_date)]
    if frame.empty:
        raise ValueError(f'No current merged games for {selected_date:%Y-%m-%d}')
    prior=history.loc[history.game_date_dt<selected_date]
    if len(prior)<2000 or prior.game_date_dt.max()>=selected_date:
        raise ValueError('Insufficient strictly prior training history for requested date')
    probs=RidgeLogit(FEATURE_SETS[name],penalty).fit(prior).predict(frame)
    out=frame[['game_id','game_date','game_time','home_team','away_team','vig_free_home_prob']].copy()
    out['my_probability_home']=probs
    out['my_probability_away']=1.-probs
    out['my_probability_minus_market']=probs-out.vig_free_home_prob.to_numpy(float)
    out['winner']=np.where(probs>=.5,out.home_team,out.away_team)
    out['train_games_before_target']=len(prior)
    out['max_training_game_date']=prior.game_date_dt.max().strftime('%Y-%m-%d')
    out['model']=MODEL_ID
    out['market_fallback_used']=False
    out['market_or_provider_features_used']=False
    out['generated_at_utc']=datetime.now(timezone.utc).isoformat()
    return out


def test_without_network():
    # Cheap deterministic regression tests for score parsing and chronology.
    example={'id':2024020001,'gameType':2,'gameState':'OFF','gameDate':'2024-10-09',
             'homeTeam':{'abbrev':'BOS','score':3},'awayTeam':{'abbrev':'MTL','score':2}}
    assert parse_completed(example,2024)['target']==1.
    assert parse_completed({**example,'gameState':'FUT'},2024) is None
    assert parse_completed({**example,'gameType':3},2024) is None
    try:
        parse_completed({**example,'homeTeam':{'abbrev':'BOS','score':2}},2024)
    except ValueError:
        pass
    else:
        raise AssertionError('tied postgame score should not be accepted')
    print('SELF-TEST PASS: official ID, final-state filter, outcome, invalid score')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--date',help='YYYY_MM_DD; defaults latest merged date')
    ap.add_argument('--offline',action='store_true',help='Require all historical source files from local cache; never fetch')
    ap.add_argument('--self-test',action='store_true',help='Run parser and temporal regression tests; no network')
    args=ap.parse_args()
    if args.self_test:
        test_without_network();return
    session=requests.Session()
    session.headers.update({'Accept':'application/json','User-Agent':'NHL-research-oddsfree-history/1.0'})
    originals=[]
    for year in (2022,2023,2024):
        originals.append(load_old_season(session,year,offline=args.offline))
    old=pd.concat(originals,ignore_index=True)
    history,current,local=historical_and_current(old)
    selection,winning_model,reg=select_model(history)
    pred=evaluate_future(history,winning_model,reg)
    out,metrics=report_metrics(pred,local)
    target_date=(pd.Timestamp(args.date.replace('_','-')) if args.date else current.game_date_dt.max())
    future=forecast(history,current,winning_model,reg,target_date)
    reports=OUT/'reports'
    save(pd.DataFrame([{'season':y,'games':len(a), 'source':'NHL_official_postgame_schedule','captured_at':'postgame_not_pregame','older_history_complete':True} for y,a in zip((2022,2023,2024),originals)]),reports/'official_history_coverage.csv')
    save(selection,reports/'model_selection_2024_validation.csv')
    save(pd.DataFrame([{'model':winning_model,'penalty':reg,'selected_on':'2024_2025_only','evaluation_seasons':'2025_2026;2026_2027','odds_used_in_fit':False,'source_forecast_used_in_fit':False,'historical_results_asof_rule':'source_game_date < target_date','feature_pre_game_provenance':'derived_only_from_prior_completed_scores','model_version':MODEL_ID}]),reports/'selection_provenance.csv')
    save(out.drop(columns='game_date_dt'),reports/'chronological_2025_2026_evaluation.csv')
    save(metrics,reports/'chronological_summary.csv')
    path=OUT/'predictions'/f'{target_date:%Y_%m_%d}_NHL_my_probability_independent_v5.csv'
    save(future,path)
    print('SELECTED (2024-25 ONLY):', winning_model,'penalty',reg)
    print('TESTS (2025-26; 2026-27 HOLDOUT):')
    print(metrics[['season','model','games','brier','log_loss','brier_minus_market_paired','paired_date_95ci_low','paired_date_95ci_high']].to_string(index=False))
    print('NO BET SELECTION; no market fallback. Market is benchmark only.')
    print('2025-26 market odds timestamps unverified. 2026 source timestamps not independently authenticated.')

if __name__=='__main__':
    try:
        main()
    except (ValueError,RuntimeError,OSError,ImportError,KeyError) as e:
        sys.exit(f'ERROR: {e}')
