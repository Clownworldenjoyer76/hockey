#!/usr/bin/env python3
"""Offline counterfactual: test a specific recent-form check on NHL >=70% forecasts.

Uses EXISTING V8 predictions and pregame-only Elo/xG/form/rest features.
If V8 gives a favorite >=70% but its recent 10-game wins advantage is <0.15
AND its average recent goal-difference advantage is <1.0, reduce the
favorite probability to 62.5%. The predicted winning side does not change.
This is an exploratory test proposed AFTER inspecting four 2026-27 losses,
NOT an untouched future validation, and never an automatically adopted model.

Run beside analyze_v8_70plus_losses.py and improve_probability_win_rate_60plus.py.
No external services, D-Ratings/SDV/odds inputs, or production changes.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import improve_probability_win_rate_60plus as base

OUTPUT = base.ROOT / 'my_probability' / 'independent_research' / 'probability_win_rate_60plus' / 'form_gate'
WIN_DIFF_THRESHOLD = 0.15
GOAL_DIFF_THRESHOLD = 1.0
MIN_ORIGINAL_CONFIDENCE = 0.70
ADJUSTED_CONFIDENCE = 0.625
BANDS = ((.60, .65, '60-<65%'), (.65, .70, '65-<70%'), (.70, 1.000001, '70%+'))


def require(condition: bool, msg: str) -> None:
    if not condition:
        raise ValueError(msg)


def write_csv(frame: pd.DataFrame, name: str) -> None:
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    frame.to_csv(temp, index=False)
    temp.replace(path)
    print(f'WROTE {path} | rows={len(frame)}', flush=True)


def write_json(data: dict, name: str) -> None:
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(data, indent=2, default=str) + '\n', encoding='utf-8')
    temp.replace(path)
    print(f'WROTE {path}', flush=True)


def apply_rule(frame: pd.DataFrame, source_col: str = 'v8_xg_home') -> pd.DataFrame:
    """Use as-of inputs only; do not look at game outcome to assign probabilities."""
    required = {'game_id', source_col, 'favorite_last10_win_diff', 'favorite_last10_goal_diff'}
    require(required.issubset(frame.columns), f'Missing columns: {required-set(frame.columns)}')
    p = pd.to_numeric(frame[source_col], errors='coerce').to_numpy(dtype=float)
    require(np.isfinite(p).all() and np.all((p > 0.) & (p < 1.)), 'Invalid V8 probabilities')
    win = pd.to_numeric(frame.favorite_last10_win_diff, errors='coerce').to_numpy(float)
    goal = pd.to_numeric(frame.favorite_last10_goal_diff, errors='coerce').to_numpy(float)
    favorite_home = p >= .5
    before = np.maximum(p, 1. - p)
    flagged = ((before >= MIN_ORIGINAL_CONFIDENCE) & np.isfinite(win) & np.isfinite(goal)
               & (win < WIN_DIFF_THRESHOLD) & (goal < GOAL_DIFF_THRESHOLD))
    after_conf = np.where(flagged, ADJUSTED_CONFIDENCE, before)
    after = np.where(favorite_home, after_conf, 1. - after_conf)
    require(np.isfinite(after).all() and np.all((after > 0.) & (after < 1.)), 'Invalid experimental probabilities')
    require(np.array_equal(after >= .5, favorite_home), 'Changed predicted winning team')
    require(np.array_equal(before >= .60, after_conf >= .60), 'Lost or added 60%+ prediction coverage')
    result = frame.copy()
    result['source_confidence'] = before
    result['recent_form_flag'] = flagged
    result['adjusted_home_probability'] = after
    result['adjusted_away_probability'] = 1. - after
    result['adjusted_favorite_confidence'] = after_conf
    return result


def band_table(frame: pd.DataFrame, period: str, source: str, pcol: str) -> tuple[pd.DataFrame, dict]:
    p = frame[pcol].to_numpy(float)
    y = frame.target.to_numpy(float)
    require(np.isin(y, (0., 1.)).all(), 'Invalid game outcomes')
    conf = np.maximum(p, 1. - p)
    won = np.where(p >= .5, y, 1. - y)
    rows = []
    for low, high, label in BANDS:
        selected = (conf >= low) & (conf < high)
        n = int(selected.sum())
        wins = int(won[selected].sum())
        rows.append({'season': period, 'model': source, 'probability_band': label,
                     'games': n, 'wins': wins, 'losses': n - wins,
                     'actual_win_pct': round(100*wins/n, 2) if n else np.nan,
                     'mean_predicted_pct': round(100*conf[selected].mean(), 2) if n else np.nan})
    out = pd.DataFrame(rows)
    rates = out.actual_win_pct.to_numpy(float)
    ascending = bool(np.isfinite(rates).all() and np.all(np.diff(rates) > 0.))
    total = int(out.games.sum())
    summary = {'season': period, 'model': source, 'games_at_60plus': total,
               'win_rates_strictly_ascending': ascending,
               'band_sizes': [int(k) for k in out.games],
               'win_rates_pct': [float(k) if pd.notna(k) else None for k in out.actual_win_pct],
               'brier_all_games': float(np.mean((p-y)**2))}
    return out, summary


def historical_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    validation = base.load_predictions(base.V8_REPORTS/'2024_ablation_predictions.csv', '2024_2025')
    replay = base.load_predictions(base.V8_REPORTS/'2025_2027_chronological_replay.csv')
    require(set(replay.data_period)=={'2025_2026','2026_2027'},'Missing historical evaluation seasons')
    predictions = pd.concat([validation, replay], ignore_index=True)
    require(not predictions.game_id.duplicated().any(), 'Duplicate predictions')
    history, _slate = base.load_cached_features()
    matched = base.make_features(predictions, history)
    revised = apply_rule(matched)
    grouped = []
    summary = []
    for period, season in revised.groupby('data_period', sort=True):
        for name, col in [('unchanged_V8', 'v8_xg_home'), ('experimental_form_gate', 'adjusted_home_probability')]:
            groups, stats = band_table(season, str(period), name, col)
            grouped.append(groups); summary.append(stats)
            print(f'{period} {name}: counts={stats["band_sizes"]}, win%={stats["win_rates_pct"]}, '
                  f'ascending={stats["win_rates_strictly_ascending"]}', flush=True)
        flag = season.loc[season.recent_form_flag]
        f_won = flag.favorite_won.to_numpy(float)
        print(f'{period} confidence-reduced >=70%: {len(flag)} games | wins={int(f_won.sum())} '
              f'losses={int((1-f_won).sum())}', flush=True)
    audit = pd.DataFrame(summary)
    changed = bool(revised.recent_form_flag.any())
    return revised, pd.concat(grouped, ignore_index=True), audit, changed


def make_slate() -> pd.DataFrame | None:
    saved = base.load_saved_slate(None)
    if saved is None:
        print('SLATE: no saved V8 forecast found; historical test only', flush=True)
        return None
    _, slate_features = base.load_cached_features()
    candidates = slate_features.loc[slate_features.game_id.astype(str).isin(saved.game_id.astype(str))].copy()
    require(len(candidates)==len(saved) and set(candidates.game_id.astype(str))==set(saved.game_id.astype(str)),
            'Saved V8 slate and existing pregame feature cache do not match; no slate written')
    existing = base.make_features(saved[['game_id','game_date_dt','v8_xg_home']], candidates)
    adjusted = apply_rule(existing)
    out = saved[['game_id', 'game_date', 'home_team', 'away_team', 'v8_xg_home']].copy()
    changes = adjusted.set_index('game_id')
    out['form_check_flag'] = out.game_id.map(changes.recent_form_flag).astype(bool)
    out['experimental_home_probability'] = out.game_id.map(changes.adjusted_home_probability)
    out['experimental_away_probability'] = 1. - out.experimental_home_probability
    out['research_only'] = True
    out['goal_confirmed_on_unseen_games'] = False
    return out


def self_test() -> None:
    values = pd.DataFrame({'game_id':['1','2','3','4','5','6'],
                           'v8_xg_home':[.8, .2, .7, .68, .75, .79],
                           'favorite_last10_win_diff':[.1,.1,.15,0.,np.nan,.10],
                           'favorite_last10_goal_diff':[.5,.5,.5,.5,.4,1.1],
                           'target':[1.,0.,1.,0.,0.,1.]})
    changed = apply_rule(values)
    assert changed.recent_form_flag.tolist() == [True, True, False, False, False, False]
    assert np.allclose(changed.adjusted_home_probability.to_numpy(), [.625,.375,.7,.68,.75,.79])
    assert np.allclose(changed.adjusted_home_probability + changed.adjusted_away_probability, 1.)
    assert (changed.adjusted_home_probability.ge(.5).to_numpy()==values.v8_xg_home.ge(.5).to_numpy()).all()
    assert int(changed.source_confidence.ge(.60).sum())==int(changed.adjusted_favorite_confidence.ge(.60).sum())
    sample = pd.DataFrame({'game_id':list(map(str,range(48))),
                           'target': np.concatenate([np.r_[np.ones(n),np.zeros(12-n)] for n in (7,9,11,11)]),
                           'v8_xg_home': np.concatenate([np.repeat(p,12) for p in (.62,.675,.725,.8)])})
    table, stats = band_table(sample,'synthetic','baseline','v8_xg_home')
    assert table.games.tolist()==[12,12,24] and stats['win_rates_strictly_ascending']
    print('SELF-TEST PASS: home/away orientation, 70% boundary, missing feature, 60%+ coverage, band progression',flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--no-slate', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    print('TESTING TARGETED 70%+ FORM CONFIRMATION; EXISTING LOCAL V8 DATA ONLY',flush=True)
    revised, grouped, summary, changed = historical_experiment()
    write_csv(revised[['game_id','game_date_dt','data_period','target','v8_xg_home',
                       'adjusted_home_probability','recent_form_flag','source_confidence',
                       'adjusted_favorite_confidence','favorite_last10_win_diff','favorite_last10_goal_diff']],
              '2024_2027_game_by_game_form_gate.csv')
    write_csv(grouped, '2024_2027_60plus_band_win_rates.csv')
    write_csv(summary, '2024_2027_band_progression_summary.csv')
    flagged = revised.loc[revised.recent_form_flag,
                          ['game_id','game_date_dt','data_period','v8_xg_home','target','favorite_won',
                           'favorite_last10_win_diff','favorite_last10_goal_diff','adjusted_home_probability']]
    write_csv(flagged, 'games_confidence_reduced.csv')
    print('EXPERIMENTAL RULE: if V8 favorite >=70%, recent win advantage <0.15, and recent goal '
          'advantage <1.0: reduce favorite to 62.5%. No favorite flips or predictions below 60%.',flush=True)
    if not args.no_slate:
        try:
            slate = make_slate()
            if slate is not None:
                write_csv(slate, 'predictions/latest_saved_v8_experimental_form_gate.csv')
        except (ValueError, FileNotFoundError) as e:
            print(f'SLATE NOT WRITTEN: {e}',flush=True)
    accepted = bool(summary.loc[summary.model=='experimental_form_gate','win_rates_strictly_ascending'].all())
    write_json({'created_utc': datetime.now(timezone.utc).isoformat(),
                'origin':'2026-27 observed >=70% losses motivated this test; retrospective, not independent confirmation',
                'fixed_probability_bands':['60-<65%','65-<70%','70%+'],
                'rule':{'min_original_confidence':MIN_ORIGINAL_CONFIDENCE,
                        'win_diff_below':WIN_DIFF_THRESHOLD,'goal_diff_below':GOAL_DIFF_THRESHOLD,
                        'cap_to':ADJUSTED_CONFIDENCE},
                'results_from_existing_models':True,
                '60plus_counts_preserved':True,
                'all_seasons_show_strictly_ascending_win_rates':accepted,
                'changed_any_probabilities':changed,
                'no_automatic_adoption':True,
                'no_provider_probabilities_or_odds_used_as_inputs':True,
                'production_unchanged':True},'manifest.json')
    print(f'RESULT: {"ASCENDING ALL THREE HISTORICAL SEASONS" if accepted else "NOT ASCENDING ALL THREE HISTORICAL SEASONS"}',flush=True)
    print('Research-only rule created after seeing 2026-27 losses; no prospective validity claimed.',flush=True)


if __name__=='__main__':
    main()
