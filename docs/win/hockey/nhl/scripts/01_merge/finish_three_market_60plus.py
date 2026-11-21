#!/usr/bin/env python3
"""Finish the user's historical 60%+ three-market check using existing files.

User's sole historical test: strictly increasing actual win percentages in
60–<65%, 65–<70%, and >=70% probability bands (no minimum group-size gate).

Reads previously written totals-candidate and three-market reports. Lists ALL
candidate totals models passing both 2025–26 and 2026–27 retrospectively,
uses a fully disclosed tie-break to nominate one research-only candidate,
and scores the saved prediction slate with historical goal totals only.

D-Ratings/SDV forecast probabilities and sportsbook prices are never inputs.
Existing ML/puck-line outputs and all production files remain untouched.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import test_independent_totals_distribution as totals

ROOT = totals.ROOT
RESEARCH = ROOT / 'my_probability' / 'independent_research'
SOURCE = RESEARCH / 'three_market_60plus'
TOTALS_SOURCE = SOURCE / 'totals_distribution_test'
PUCK_SOURCE = SOURCE / 'goal_count_correction'
ML_SOURCE = RESEARCH / 'probability_win_rate_60plus' / 'form_gate'
OUT = SOURCE / 'completed_three_market_60plus'
PERIODS = ('2025_2026', '2026_2027')
BANDS = ('60-<65%', '65-<70%', '70%+')


def check(test, msg):
    if not bool(test):
        raise ValueError(msg)


def bool_col(column):
    vals = column.astype(str).str.strip().str.lower()
    check(vals.isin(('true', 'false')).all(), 'Invalid boolean value in stored report')
    return vals.eq('true')


def read_csv(path):
    check(path.is_file(), f'Missing prior report: {path}')
    return pd.read_csv(path, low_memory=False)


def save_csv(df, rel):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.pending')
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    print(f'WROTE {p} | rows={len(df)}', flush=True)


def report_from_cached():
    all_models = read_csv(TOTALS_SOURCE / 'all_totals_models_60plus.csv')
    all_bands = read_csv(TOTALS_SOURCE / 'all_totals_models_probability_bands.csv')
    prior_markets = read_csv(PUCK_SOURCE / 'all_three_markets_user_goal.csv')

    needed = {'season', 'candidate', 'strict_increase_60plus', 'bands_populated',
              'minimum_step_pp', 'games_at_60plus', 'band_counts', 'actual_win_rates_pct'}
    check(needed.issubset(all_models.columns),
          f'Missing totals columns: {sorted(needed-set(all_models.columns))}')
    need_bands = {'season', 'candidate', 'band', 'games', 'wins', 'actual_win_pct'}
    check(need_bands.issubset(all_bands.columns),
          f'Missing totals band columns: {sorted(need_bands-set(all_bands.columns))}')
    required_markets = {'season', 'market', 'candidate', 'strict_increase_60plus',
                        'band_counts', 'actual_win_rates_pct'}
    check(required_markets.issubset(prior_markets.columns),
          f'Missing ML/puck columns: {sorted(required_markets-set(prior_markets.columns))}')

    models = all_models.copy()
    models['strict_increase_60plus'] = bool_col(models['strict_increase_60plus'])
    models['bands_populated'] = bool_col(models['bands_populated'])
    models = models[models.season.isin(PERIODS)].copy()
    check(not models.duplicated(['season','candidate']).any(), 'Duplicate totals candidate/season')
    check(set(models.season) == set(PERIODS), 'Missing totals historical season')

    # Recompute win-rate band progression from observed wins/games; don't trust
    # the previously stored PASS flags without checking them against raw counts.
    for r in models.itertuples():
        x = all_bands[(all_bands.season == r.season) &
                      (all_bands.candidate == r.candidate)]
        check(len(x) == 3 and set(x.band) == set(BANDS),
              f'Incomplete bands: {r.season} {r.candidate}')
        x = x.set_index('band').loc[list(BANDS)]
        n = pd.to_numeric(x.games, errors='coerce').to_numpy(dtype=float)
        wins = pd.to_numeric(x.wins, errors='coerce').to_numpy(dtype=float)
        check(np.isfinite(n).all() and (n >= 0).all(), 'Bad game counts')
        check(np.isfinite(wins).all() and (wins >= 0).all() and (wins <= n).all(), 'Bad win counts')
        valid = bool((n > 0).all())
        rates = 100 * wins / np.where(n > 0, n, 1)
        passed = valid and bool(np.all(np.diff(rates) > 0))
        check(bool(r.strict_increase_60plus) == passed,
              f'Incorrect preexisting PASS flag for {r.season} {r.candidate}')
        check(bool(r.bands_populated) == valid,
              f'Incorrect preexisting band populated flag for {r.season} {r.candidate}')
        check(int(n.sum()) == int(r.games_at_60plus),
              f'Coverage mismatch for {r.season} {r.candidate}')

    groups = models.groupby('candidate', sort=True)
    qualified=[]
    for candidate, group in groups:
        if len(group) != 2 or set(group.season) != set(PERIODS):
            continue
        if not group.strict_increase_60plus.all():
            continue
        qualified.append({
            'candidate':candidate,
            'min_observed_increase_pp':float(group.minimum_step_pp.min()),
            'total_games_at_60plus':int(group.games_at_60plus.sum()),
            '2025_band_counts':str(group.loc[group.season=='2025_2026','band_counts'].iloc[0]),
            '2025_win_rates_pct':str(group.loc[group.season=='2025_2026','actual_win_rates_pct'].iloc[0]),
            '2026_band_counts':str(group.loc[group.season=='2026_2027','band_counts'].iloc[0]),
            '2026_win_rates_pct':str(group.loc[group.season=='2026_2027','actual_win_rates_pct'].iloc[0]),
        })
    candidates = pd.DataFrame(qualified)
    check(not candidates.empty, 'No totals candidate passes all three bands in both seasons')
    candidates = candidates.sort_values(
        ['min_observed_increase_pp','total_games_at_60plus','candidate'],
        ascending=[False,False,True],kind='stable').reset_index(drop=True)
    # Sorting here only nominates a research candidate. All passing candidates
    # are exported for the owner; no files are automatically promoted.
    winner = str(candidates.iloc[0].candidate)

    existing = prior_markets[prior_markets.season.isin(PERIODS) &
                             prior_markets.market.isin(['moneyline','puck_line'])].copy()
    check(len(existing) == 4 and
          not existing.duplicated(['season','market']).any(),
          'Expected exactly two seasons of moneyline and puck-line results')
    existing['strict_increase_60plus'] = bool_col(existing['strict_increase_60plus'])
    check(existing.strict_increase_60plus.all(),
          'Moneyline or puck-line did not pass in one of the existing reports')
    selected_totals = models.loc[models.candidate.eq(winner),
          ['season','candidate','strict_increase_60plus','band_counts','actual_win_rates_pct']].copy()
    selected_totals['market'] = 'total'
    status = pd.concat([existing[['season','market','candidate','strict_increase_60plus',
                                   'band_counts','actual_win_rates_pct']],
                        selected_totals[['season','market','candidate','strict_increase_60plus',
                                         'band_counts','actual_win_rates_pct']]], ignore_index=True)
    status = status.sort_values(['season','market']).reset_index(drop=True)
    check(len(status) == 6 and status.strict_increase_60plus.all(),
          'Not all 3 markets pass 60%+ retrospective progression')
    return candidates, status, winner


def make_slate(winner):
    """Refit only as needed for an already saved slate; no historical re-evaluation."""
    saved = totals.prior.old.base.load_saved_slate(None)
    if saved is None:
        print('No saved V8 slate: candidate report only',flush=True)
        return
    need={'game_id','game_date','home_team','away_team'}
    check(need.issubset(saved.columns), 'Saved V8 slate missing identifiers')
    check(not saved.game_id.astype(str).duplicated().any(), 'Duplicate saved slate game IDs')
    history, slate = totals.prior.old.base.load_cached_features()
    history, slate = totals.prior.old.add_prior_goal_rates(history, slate)
    history = totals.prior.with_goal_features(history)
    ids=set(saved.game_id.astype(str))
    current = slate[slate.game_id.astype(str).isin(ids)].copy()
    check(len(current)==len(ids) and set(current.game_id.astype(str))==ids,
          'Saved slate mismatch with cached features')
    current = totals.prior.old.attach_saved_slate_lines(current, totals.prior.old.get_lines())
    current = totals.prior.with_goal_features(current)
    current = current[current.total.between(3,10)].copy()
    check(not current.empty, 'No supported totals contract lines on saved slate')
    date = pd.to_datetime(current.game_date_dt)
    check(date.nunique()==1, 'Mixed dates in saved slate')
    first_date = date.iloc[0]
    train = history[history.game_date_dt < first_date].copy()
    check(not train.empty and pd.to_datetime(train.game_date_dt).max()<first_date,
          'Historical training date invalid')
    if winner=='unchanged_independent_total':
        base_train=totals.prior.old.make_training(train,'total')
        model=totals.prior.old.RidgeLogit(totals.prior.old.TOTAL_FEATURES,totals.prior.old.PENALTY).fit(base_train)
        base_p=model.predict(current)
    else:
        base_p=None
    p=totals.calculate_candidates(train,current,base_p)[winner]
    check(np.isfinite(p).all() and np.all((p>0)&(p<1)), 'Invalid selected totals slate probabilities')
    total_df=current[['game_id','game_date','home_team','away_team','total']].copy()
    total_df['market']='total'
    total_df['model']=winner
    total_df['contract_line']=current.total.to_numpy(float)
    total_df['side1']='over';total_df['side2']='under'
    total_df['prob_side1']=p;total_df['prob_side2']=1-p
    total_df['predicted_side']=np.where(p>=.5,'over','under')
    total_df['predicted_win_pct']=100*np.maximum(p,1-p)

    # Existing ML + puck-line research predictions are unchanged and copied
    # into a *separate* consolidated research report for convenience.
    ml_path = ML_SOURCE/'predictions'/'latest_saved_v8_experimental_form_gate.csv'
    puck_path = PUCK_SOURCE/'predictions'/f'{pd.Timestamp(first_date).strftime("%Y_%m_%d")}_independent_puck_total_candidate.csv'
    check(ml_path.is_file(), f'Existing moneyline research prediction missing: {ml_path}')
    check(puck_path.is_file(), f'Existing puck-line research prediction missing: {puck_path}')
    ml = pd.read_csv(ml_path,dtype={'game_id':str})
    puck = pd.read_csv(puck_path,dtype={'game_id':str})
    check({'game_id','experimental_home_probability','experimental_away_probability'}.issubset(ml.columns),
          'Moneyline research prediction schema changed')
    need_puck={'game_id','market','contract_line','prob_side1','prob_side2',
               'predicted_side','predicted_win_pct'}
    check(need_puck.issubset(puck.columns), 'Puck research prediction schema changed')
    puck=puck.loc[puck.market.eq('puck_line')].copy()
    # Confirm no cross-slate joins; match by actual game ID.
    check(set(total_df.game_id.astype(str)).issubset(set(ml.game_id.astype(str))),
          'Moneyline research slate does not contain selected totals game IDs')
    check(set(total_df.game_id.astype(str)).issubset(set(puck.game_id.astype(str))),
          'Puck research slate does not contain selected totals game IDs')
    ml['market']='moneyline'
    ml['model']='experimental_v8_form_gate'
    ml['contract_line']=np.nan
    ml['side1']='home';ml['side2']='away'
    ml['prob_side1']=ml.experimental_home_probability
    ml['prob_side2']=ml.experimental_away_probability
    ml['predicted_side']=np.where(ml.prob_side1>=.5,'home','away')
    ml['predicted_win_pct']=100*np.maximum(ml.prob_side1,ml.prob_side2)
    if 'model' not in puck: puck['model']='blend_50pct_goals_alpha_5'
    if 'side1' not in puck:puck['side1']='home_covers'
    if 'side2' not in puck:puck['side2']='away_covers'
    final_cols=['game_id','game_date','home_team','away_team','market','model',
                'contract_line','side1','side2','prob_side1','prob_side2',
                'predicted_side','predicted_win_pct']
    all_slate=pd.concat([ml[final_cols],puck[final_cols],total_df[final_cols]],ignore_index=True)
    check(not all_slate.duplicated(['game_id','market']).any(),
          'Duplicate three-market game/contract')
    all_slate['research_only']=True
    save_csv(total_df, 'predictions/totals_historical_ascent_candidate.csv')
    save_csv(all_slate,'predictions/three_market_60plus_research_slate.csv')


def run(skip_slate=False):
    candidates, status, winner = report_from_cached()
    save_csv(candidates,'totals_candidates_passing_both_seasons.csv')
    save_csv(status,'three_market_60plus_historical_progression.csv')
    print('\nTOTALS CANDIDATES PASSING BOTH HISTORICAL SEASONS:',flush=True)
    print(candidates.to_string(index=False),flush=True)
    print('\nTHREE-MARKET RESULT WITH RESEARCH NOMINEE:',flush=True)
    print(status[['season','market','band_counts','actual_win_rates_pct']].to_string(index=False),flush=True)
    if not skip_slate: make_slate(winner)
    data={'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'criterion':'60-<65% < 65-<70% < 70%+ actual win rates separately for each market and each evaluated season',
          'minimum_sample_size_gate':False,
          'totals_candidates_passing_both':int(len(candidates)),
          'research_nominee':winner,
          'nomination_rule':'maximizes smallest observed positive win-rate increase across the two historical seasons, then 60%+ game count; research only',
          'both_2025_and_2026_seasons_inspected_to_nominate':True,
          'independent_future_validation_claimed':False,
          'three_market_historical_60plus_pass':True,
          'production_changed':False,
          'existing_ML_and_puck_line_files_changed':False,
          'source':'previously written local 2025 and 2026 scored-game reports'}
    path=OUT/'manifest.json';path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
    print(f'WROTE {path}',flush=True)
    print('HISTORICAL 60%+ ASCENT: PASS for all three markets with the nominated totals candidate',flush=True)
    print('Selection is retrospective; no model deployed.',flush=True)


def self_test():
    with __import__('tempfile').TemporaryDirectory() as td:
        td=Path(td)
        orig=(globals()['TOTALS_SOURCE'],globals()['PUCK_SOURCE'],globals()['OUT'])
        globals()['TOTALS_SOURCE']=td/'t';globals()['PUCK_SOURCE']=td/'p';globals()['OUT']=td/'out'
        globals()['TOTALS_SOURCE'].mkdir();globals()['PUCK_SOURCE'].mkdir()
        model_rows=[];band_rows=[]
        for season in PERIODS:
            for name,rates in [('pass_a',[.51,.58,.65] if season==PERIODS[0] else [.46,.56,.64]),
                               ('pass_b',[.55,.57,.60] if season==PERIODS[0] else [.50,.55,.61]),
                               ('fail_c',[.66,.59,.62])]:
                n=[40,30,20]
                wins=[int(round(r*n_i)) for r,n_i in zip(rates,n)]
                actual=[100*w/num for w,num in zip(wins,n)]
                qualified=actual[0]<actual[1]<actual[2]
                model_rows.append({'season':season,'candidate':name,
                     'strict_increase_60plus':qualified,'bands_populated':True,
                     'minimum_step_pp':min(np.diff(actual)),'games_at_60plus':sum(n),
                     'band_counts':str(n),'actual_win_rates_pct':str(actual)})
                for band,num,win,rate in zip(BANDS,n,wins,actual):
                    band_rows.append({'season':season,'candidate':name,'band':band,
                                      'games':num,'wins':win,'actual_win_pct':rate})
        pd.DataFrame(model_rows).to_csv(globals()['TOTALS_SOURCE']/'all_totals_models_60plus.csv',index=False)
        pd.DataFrame(band_rows).to_csv(globals()['TOTALS_SOURCE']/'all_totals_models_probability_bands.csv',index=False)
        puck_rows=[]
        for season in PERIODS:
            for market in ('moneyline','puck_line'):
                puck_rows.append({'season':season,'market':market,'candidate':'existing',
                                  'strict_increase_60plus':True,'band_counts':'[10, 10, 10]',
                                  'actual_win_rates_pct':'[60, 70, 80]'})
        pd.DataFrame(puck_rows).to_csv(globals()['PUCK_SOURCE']/'all_three_markets_user_goal.csv',index=False)
        try:
            c,s,w=report_from_cached()
            check(len(c)==2 and len(s)==6,'Candidate filtering smoke mismatch')
            check(w in ['pass_a','pass_b'],'Candidate nomination incorrect')
            run(skip_slate=True)
            check((globals()['OUT']/'three_market_60plus_historical_progression.csv').is_file(),
                  'Did not write progress report')
        finally:
            globals()['TOTALS_SOURCE'],globals()['PUCK_SOURCE'],globals()['OUT']=orig
    print('SELF-TEST PASS: band verification, candidate filter, three-market integration, no group-size gate')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--no-slate',action='store_true')
    args=parser.parse_args()
    try:
        if args.self_test:self_test()
        else:run(skip_slate=args.no_slate)
    except (ValueError,KeyError,FileNotFoundError,RuntimeError,OSError) as exc:
        raise SystemExit(f'ERROR: {exc}')
