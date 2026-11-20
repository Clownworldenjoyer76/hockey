#!/usr/bin/env python3
"""Audit whether higher independent V8 NHL win probabilities actually win more often.

Uses existing V8 2024-25 validation and 2025-27 historical replay CSVs.
Does not refit, blend, scrape, replace, or use D-Ratings, SDV, or market data.
Writes separate game-level, equal-count quintile, fixed-bin and summary CSVs.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {'game_id', 'data_period', 'target', 'v8_xg_home'}


def fail(message: str) -> None:
    raise ValueError(message)


def validated_predictions(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        fail(f'Missing V8 {label} data: {path}')
    df = pd.read_csv(path, dtype={'game_id': str, 'data_period': str})
    if not REQUIRED.issubset(df.columns):
        fail(f'V8 {label} CSV missing required columns: {sorted(REQUIRED - set(df.columns))}')
    if df.empty:
        fail(f'V8 {label} CSV is empty: {path}')
    if df.game_id.isna().any() or df.game_id.duplicated().any():
        fail(f'V8 {label} CSV contains missing or duplicate game_id')
    if df.data_period.isna().any():
        fail(f'V8 {label} CSV missing season')
    for col in ('target', 'v8_xg_home'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df[col].to_numpy()).all():
            fail(f'V8 {label} CSV has missing or nonfinite {col}')
    if not df.target.isin([0., 1.]).all():
        fail(f'V8 {label} outcomes must be 0 (away won) or 1 (home won)')
    if not df.v8_xg_home.between(0, 1, inclusive='both').all():
        fail(f'V8 {label} home probabilities must be 0 to 1')
    if len(df.data_period.unique()) != (1 if label == 'validation' else 2):
        fail(f'Unexpected seasons in {label}: {sorted(df.data_period.unique())}')
    if label == 'validation' and set(df.data_period.unique()) != {'2024_2025'}:
        fail(f'Validation must be 2024_2025, got {sorted(df.data_period.unique())}')
    if label == 'replay' and set(df.data_period.unique()) != {'2025_2026', '2026_2027'}:
        fail(f'Replay must have 2025_2026 and 2026_2027, got {sorted(df.data_period.unique())}')
    df = df[['game_id', 'game_date', 'data_period', 'target', 'v8_xg_home']].copy() if 'game_date' in df else df[['game_id', 'data_period', 'target', 'v8_xg_home']].copy()
    df['favored_side'] = np.where(df.v8_xg_home >= .5, 'home', 'away')
    df['predicted_favorite_win_probability'] = np.maximum(df.v8_xg_home, 1 - df.v8_xg_home)
    df['favorite_won'] = np.where(df.favored_side.eq('home'), df.target, 1 - df.target).astype(int)
    df['predicted_home_probability'] = df.v8_xg_home
    df['predicted_away_probability'] = 1 - df.v8_xg_home
    return df


def auc(prob: pd.Series, outcome: pd.Series) -> float:
    y = outcome.to_numpy(dtype=int)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if not n1 or not n0:
        return float('nan')
    ranks = prob.rank(method='average').to_numpy(dtype=float)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def quintiles(g: pd.DataFrame, season: str) -> pd.DataFrame:
    ordered = g.sort_values(['predicted_favorite_win_probability', 'game_id'], kind='mergesort').copy()
    # Split into equal-size groups; no boundary selection based on game outcomes.
    ordered['rank_group'] = np.floor(np.arange(len(ordered)) * 5 / len(ordered)).astype(int) + 1
    rows = []
    for ix, b in ordered.groupby('rank_group', sort=True):
        rows.append({
            'season': season,
            'probability_rank_group': int(ix),
            'games': len(b),
            'probability_low_pct': round(100 * b.predicted_favorite_win_probability.min(), 2),
            'probability_high_pct': round(100 * b.predicted_favorite_win_probability.max(), 2),
            'average_predicted_win_pct': round(100 * b.predicted_favorite_win_probability.mean(), 2),
            'actual_win_pct': round(100 * b.favorite_won.mean(), 2),
            'wins': int(b.favorite_won.sum()),
            'probability_minus_actual_pp': round(100 * (b.predicted_favorite_win_probability.mean() - b.favorite_won.mean()), 2),
        })
    return pd.DataFrame(rows)


def fixed_bins(g: pd.DataFrame, season: str) -> pd.DataFrame:
    breaks = np.arange(.50, 1.001, .05)
    rows = []
    for lo in breaks[:-1]:
        hi = min(1., round(lo + .05, 8))
        mask = (g.predicted_favorite_win_probability >= lo - 1e-9) & (
            (g.predicted_favorite_win_probability < hi - 1e-9) if hi < 1 else
            (g.predicted_favorite_win_probability <= hi))
        b = g.loc[mask]
        rows.append({
            'season': season,
            'probability_range': f'{round(lo*100):02d}-{round(hi*100):02d}%',
            'games': len(b),
            'wins': int(b.favorite_won.sum()),
            'avg_predicted_win_pct': round(100 * b.predicted_favorite_win_probability.mean(), 2) if len(b) else float('nan'),
            'actual_win_pct': round(100 * b.favorite_won.mean(), 2) if len(b) else float('nan'),
            'evidence_note': 'less_than_30_games' if len(b) < 30 else '30_or_more_games',
        })
    if sum(x['games'] for x in rows) != len(g):
        fail(f'Fixed-bin allocation error in {season}')
    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quintile_frames = []
    bin_frames = []
    summaries = []
    for season, g in df.groupby('data_period', sort=True):
        q = quintiles(g, season)
        b = fixed_bins(g, season)
        quintile_frames.append(q)
        bin_frames.append(b)
        actual = q.actual_win_pct.to_numpy(float)
        decreases = np.flatnonzero(np.diff(actual) < 0)
        summaries.append({
            'season': season,
            'games': len(g),
            'favorite_wins': int(g.favorite_won.sum()),
            'favorite_win_rate_pct': round(100 * g.favorite_won.mean(), 2),
            'avg_favorite_probability_pct': round(100 * g.predicted_favorite_win_probability.mean(), 2),
            'highest_quintile_actual_win_pct': float(actual[-1]),
            'lowest_quintile_actual_win_pct': float(actual[0]),
            'quintile_win_rate_step_decreases': int(len(decreases)),
            'strictly_monotone_quintiles': bool(not len(decreases)),
            'ranking_auc': round(auc(g.predicted_favorite_win_probability, g.favorite_won), 5),
            'interpretation': 'higher_ranking_auc_is_better; 0.5_is_no_discrimination; bin_wins_vary_randomly',
        })
    return (pd.concat(quintile_frames, ignore_index=True),
            pd.concat(bin_frames, ignore_index=True), pd.DataFrame(summaries))


def self_test() -> None:
    mock = pd.DataFrame({
        'game_id': [f'{i:010d}' for i in range(20)],
        'data_period': ['2025_2026'] * 20,
        'target': [0,1]*10,
        'v8_xg_home': [0.5 + .02*i for i in range(20)],
    })
    # Construct favorite fields without requiring production files.
    mock['favored_side'] = 'home'
    mock['predicted_favorite_win_probability'] = mock.v8_xg_home
    mock['favorite_won'] = mock.target
    q, fixed, summary = analyze(mock)
    assert len(q) == 5 and q.games.sum() == len(mock)
    assert fixed.games.sum() == len(mock)
    assert summary.games.iloc[0] == len(mock)
    assert auc(pd.Series([.1,.2,.3,.4]), pd.Series([0,0,1,1])) == 1.
    assert auc(pd.Series([.1,.2,.3,.4]), pd.Series([1,1,0,0])) == 0.
    print('SELF-TEST PASS: quintile ordering, all-game counts, fixed bins, ranking AUC')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2],
                    help='NHL project directory, default inferred from scripts/01_merge placement')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    root = args.root
    base = root / 'my_probability' / 'independent_v8_xg' / 'reports'
    valid = validated_predictions(base / '2024_ablation_predictions.csv', 'validation')
    replay = validated_predictions(base / '2025_2027_chronological_replay.csv', 'replay')
    combined = pd.concat([valid, replay], ignore_index=True)
    if combined.game_id.duplicated().any():
        fail('Duplicate game ID across validation and replay')
    five, fixed, summary = analyze(combined)
    output = root / 'my_probability' / 'independent_research' / 'probability_win_rate_audit'
    output.mkdir(parents=True, exist_ok=True)
    files = [
        (combined.sort_values(['data_period', 'predicted_favorite_win_probability']), output / 'game_level_probability_and_outcome.csv'),
        (five, output / 'probability_rank_quintiles.csv'),
        (fixed, output / 'fixed_5pct_probability_buckets.csv'),
        (summary, output / 'probability_ranking_summary.csv'),
    ]
    for frame, path in files:
        temp = path.with_suffix('.tmp')
        frame.to_csv(temp, index=False)
        temp.replace(path)
        print(f'WROTE {path} | rows={len(frame)}')
    print('\nDOES HIGHER PROBABILITY CORRESPOND TO HIGHER WIN RATE?')
    for season, q in five.groupby('season', sort=True):
        print(f'\n{season}:')
        print(q[['probability_rank_group','games','average_predicted_win_pct','actual_win_pct']].to_string(index=False))
        row = summary.loc[summary.season.eq(season)].iloc[0]
        print(f'Group-to-group win-rate drops: {int(row.quintile_win_rate_step_decreases)} of 4; ranking AUC={row.ranking_auc:.3f}')
    print('\nAll data from independent V8 and actual outcomes. No D-Ratings, SDV, sportsbook, retraining, or production changes.')
    print('A small group can show a lower actual win rate by chance; higher probability is not a guaranteed outcome.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, pd.errors.ParserError) as exc:
        sys.exit(f'ERROR: {exc}')
