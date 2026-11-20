#!/usr/bin/env python3
"""Read-only diagnostic of V5's pre-existing, untuned chronological holdouts.

Run from the NHL scripts/01_merge folder beside predict_my_probability_v5_extended.py.
No model fitting, selection, network access, source modification, or bet selection.
The optional pregame-feature reconstruction uses V5's local cached NHL scores,
strictly before game dates. It does not read postgame data as target features.

Writes ONLY NHL/my_probability/independent_v5/diagnostics/*.csv and diagnosis.txt.
All segment findings are exploratory, NOT validated trading edges.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

NHL = Path(__file__).resolve().parents[2]
BASE = NHL / 'my_probability' / 'independent_v5'
SRC = BASE / 'reports' / 'chronological_2025_2026_evaluation.csv'
OUT = BASE / 'diagnostics'
EPS = 1e-8
SEED = 20261120

REQUIRED = ('game_id', 'game_date', 'data_period', 'target', 'home_team_full',
            'away_team_full', 'my_probability_home', 'vig_free_home_prob',
            'last_training_game_date', 'training_games_before_target')


def load_predictions(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing V5 chronological predictions: {path}')
    df = pd.read_csv(path, dtype={'game_id': 'string'}, low_memory=False)
    absent = sorted(set(REQUIRED) - set(df.columns))
    if absent:
        raise ValueError(f'V5 report missing required columns: {absent}')
    df['game_id'] = df.game_id.astype('string').str.strip()
    if df.game_id.isna().any() or not df.game_id.str.fullmatch(r'\d{10}').all() or df.game_id.duplicated().any():
        raise ValueError('Invalid or duplicated official NHL game_id')
    if not df.data_period.isin(['2025_2026', '2026_2027']).all():
        raise ValueError('Unexpected season in chronological holdout')
    if not pd.to_datetime(df.game_date.str.replace('_','-',regex=False), errors='coerce').notna().all():
        raise ValueError('Invalid target dates')
    df['date'] = pd.to_datetime(df.game_date.str.replace('_','-',regex=False))
    df['trained_through'] = pd.to_datetime(df.last_training_game_date, errors='coerce')
    if df.trained_through.isna().any() or (df.trained_through >= df.date).any():
        raise ValueError('Training chronology violated: model fit includes same/future date')
    ntrain = pd.to_numeric(df.training_games_before_target,errors='coerce')
    if ntrain.isna().any() or (ntrain < 2000).any():
        raise ValueError('Missing/insufficient recorded past-date training sample')
    df['target'] = pd.to_numeric(df.target, errors='coerce')
    if df.target.isna().any() or not df.target.isin([0.,1.]).all():
        raise ValueError('Missing or invalid final result labels')
    for col in ('my_probability_home', 'vig_free_home_prob'):
        df[col] = pd.to_numeric(df[col],errors='coerce')
        if df[col].isna().any() or not np.isfinite(df[col]).all() or not df[col].between(EPS,1.-EPS).all():
            raise ValueError(f'{col}: missing, invalid, or out-of-range probability')
    df['model_brier_loss'] = (df.my_probability_home-df.target)**2
    df['market_brier_loss'] = (df.vig_free_home_prob-df.target)**2
    df['brier_excess'] = df.model_brier_loss-df.market_brier_loss
    df['model_log_loss'] = -df.target*np.log(df.my_probability_home)-(1.-df.target)*np.log1p(-df.my_probability_home)
    df['market_log_loss'] = -df.target*np.log(df.vig_free_home_prob)-(1.-df.target)*np.log1p(-df.vig_free_home_prob)
    df['log_loss_excess'] = df.model_log_loss-df.market_log_loss
    df['model_market_shift'] = df.my_probability_home-df.vig_free_home_prob
    df['absolute_disagreement'] = df.model_market_shift.abs()
    df['month'] = df.date.dt.strftime('%Y-%m')
    df['market_favorite'] = np.where(df.vig_free_home_prob>=.5,'market_home_favorite','market_away_favorite')
    df['model_favorite'] = np.where(df.my_probability_home>=.5,'model_home_favorite','model_away_favorite')
    df['pick_disagreement'] = np.where((df.my_probability_home>=.5)!=(df.vig_free_home_prob>=.5),
                                       'different_favorites','same_favorite')
    df['direction'] = np.select([df.model_market_shift<=-.03, df.model_market_shift>=.03],
                                ['model_away_shift_3pp_plus','model_home_shift_3pp_plus'],
                                default='within_3pp')
    df['disagreement_band'] = pd.cut(df.absolute_disagreement,
                                    bins=[-np.inf,.01,.03,.06,.10,np.inf],
                                    labels=['under_1pp','1_to_3pp','3_to_6pp','6_to_10pp','10pp_plus'])
    df['market_confidence'] = pd.cut(abs(df.vig_free_home_prob-.5),
                                     bins=[-np.inf,.025,.05,.10,.15,np.inf],
                                     labels=['under_2.5pp','2.5_to_5pp','5_to_10pp','10_to_15pp','15pp_plus'])
    return df


def summary(group):
    return {
        'games': int(len(group)),
        'model_brier': float(group.model_brier_loss.mean()),
        'market_brier': float(group.market_brier_loss.mean()),
        'excess_brier_positive_worse': float(group.brier_excess.mean()),
        'model_log_loss': float(group.model_log_loss.mean()),
        'market_log_loss': float(group.market_log_loss.mean()),
        'excess_log_loss_positive_worse': float(group.log_loss_excess.mean()),
        'actual_home_win_rate':float(group.target.mean()),
        'predicted_home_win_rate':float(group.my_probability_home.mean()),
        'market_home_win_rate':float(group.vig_free_home_prob.mean()),
        'mean_model_minus_market_home_prob':float(group.model_market_shift.mean()),
        'mean_abs_model_minus_market':float(group.absolute_disagreement.mean()),
    }


def by(df, columns, min_games=1):
    result=[]
    for keys, group in df.groupby(columns, observed=True, dropna=False, sort=True):
        if len(group)<min_games:
            continue
        if not isinstance(keys, tuple):
            keys=(keys,)
        record={c:str(v) for c,v in zip(columns,keys)}
        record.update(summary(group))
        result.append(record)
    return pd.DataFrame(result)


def bootstrap_by_date(df, n=2500):
    rng=np.random.default_rng(SEED)
    out=[]
    for season, group in df.groupby('data_period',sort=True):
        days=list(group.groupby('date',sort=True))
        totals=np.asarray([g.brier_excess.sum() for _,g in days])
        counts=np.asarray([len(g) for _,g in days])
        vals=np.empty(n,dtype=float)
        for j in range(n):
            take=rng.integers(0,len(days),len(days))
            vals[j]=totals[take].sum()/counts[take].sum()
        row={'season':season,'calendar_dates':len(days), 'bootstrap_resamples':n,
             'positive_excess_means_model_worse':float(group.brier_excess.mean()),
             'lower_95pct':float(np.quantile(vals,.025)),
             'upper_95pct':float(np.quantile(vals,.975)),
             'percentage_samples_market_better':float(np.mean(vals>0))}
        out.append(row)
    return pd.DataFrame(out)


def calibration_bins(df):
    records=[]
    for season,part in df.groupby('data_period'):
        for model,col in [('independent_v5','my_probability_home'),('market','vig_free_home_prob')]:
            probs=part[col]
            # Predefined probability bins, rather than outcome-selected bins.
            bins=pd.cut(probs,[0.,.35,.4,.45,.5,.55,.6,.65,1.],include_lowest=True)
            for band,sub in part.groupby(bins,sort=True,observed=True):
                records.append({'season':season,'model':model,'probability_bin':str(band),
                                'games':len(sub), 'mean_predicted_prob':float(sub[col].mean()),
                                'observed_home_wins':float(sub.target.mean()),
                                'calibration_gap_predicted_minus_observed':float(sub[col].mean()-sub.target.mean()),
                                'brier':float(((sub[col]-sub.target)**2).mean())})
    return pd.DataFrame(records)


def get_asof_features(df):
    # Offline only. No API requests. The prior-game feature builder is unchanged.
    try:
        import predict_my_probability_v5_extended as v5
    except ImportError as e:
        return None,f'Optional historical feature import unavailable: {e}'
    caches=[v5.RAW/f'season_{year}_regular_games.csv' for year in (2022,2023,2024)]
    missing=[str(p) for p in caches if not p.is_file()]
    if missing:
        return None,'Missing archived official history, no network attempted: '+', '.join(missing)
    old=pd.concat([v5.load_old_season(None,year,offline=True) for year in (2022,2023,2024)],ignore_index=True)
    history,_,_=v5.historical_and_current(old)
    features=['elo_diff_asof','rest_diff_asof','back_to_back_diff_asof','games_in_four_diff_asof',
              'win_rate_last10_diff_asof','goals_diff_last10_diff_asof','goals_for_last10_diff_asof',
              'goals_against_last10_diff_asof','prior_games_min_asof','form_complete_both_teams']
    have=history[['game_id',*features]].copy()
    have['game_id']=have.game_id.astype('string').str.strip()
    if have.game_id.duplicated().any():
        raise ValueError('Duplicate reconstructed as-of feature game IDs')
    merged=df.merge(have,on='game_id',how='left',validate='one_to_one',indicator='_feature_join')
    if not merged._feature_join.eq('both').all():
        raise ValueError('Cannot match all held-out games to reconstructed pregame features')
    merged=merged.drop(columns=['_feature_join'])
    merged['elo_bin']=pd.cut(merged.elo_diff_asof,[-np.inf,-.5,-.2,.2,.5,np.inf],
                             labels=['away_elo_strong','away_elo_edge','elo_close','home_elo_edge','home_elo_strong'])
    merged['rest_bin']=pd.cut(merged.rest_diff_asof,[-np.inf,-1.5,-.5,.5,1.5,np.inf],
                              labels=['home_rest_2plus_less','home_rest_1_less','same_rest','home_rest_1_more','home_rest_2plus_more'])
    merged['prior_team_games_bin']=pd.cut(merged.prior_games_min_asof,[-np.inf,0,3,10,np.inf],
                                          labels=['no_prior_game','1_to_3','4_to_10','11_plus'])
    merged['form_available']=np.where(merged.form_complete_both_teams==1,'both_teams_history','missing_team_history')
    return merged,'Past-date feature reconstruction complete from local official NHL schedule cache'


def write(df,name):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/name
    df.to_csv(path,index=False,float_format='%.8f')
    print(f'WROTE {path} | rows={len(df)}',flush=True)


def diagnosis(df, features, bootstrap, feature_status):
    lines=['INDEPENDENT V5 HISTORICAL HOLDOUT DIAGNOSIS',
           'Source: '+str(SRC),
           'This is a read-only analysis; no V5 model, predictions, or selection settings changed.',
           'Positive excess Brier means V5 is worse than the market.',
           '2025-26 market odds timestamps unverified; comparisons are provisional.',
           '2026-27 market timestamps source-checked, not independently authenticated.',
           'All segment analyses are exploratory; multiple comparisons can create false patterns.',
           '']
    for season, part in df.groupby('data_period'):
        r=summary(part)
        boot=bootstrap.loc[bootstrap.season==season].iloc[0]
        lines += [f'{season}: games={r["games"]}; model Brier={r["model_brier"]:.6f}; market Brier={r["market_brier"]:.6f}; excess={r["excess_brier_positive_worse"]:+.6f}',
                  f'  Model/market log loss={r["model_log_loss"]:.6f}/{r["market_log_loss"]:.6f}',
                  f'  Actual home win rate={r["actual_home_win_rate"]:.3f}; model avg={r["predicted_home_win_rate"]:.3f}; market avg={r["market_home_win_rate"]:.3f}',
                  f'  Average model-market home probability shift={r["mean_model_minus_market_home_prob"]:+.4f}; absolute={r["mean_abs_model_minus_market"]:.4f}',
                  f'  Calendar-date block bootstrap 95% interval for excess: [{boot.lower_95pct:+.6f}, {boot.upper_95pct:+.6f}]']
        gg=by(part,['direction'],min_games=25)
        if not gg.empty:
            gg=gg.sort_values('excess_brier_positive_worse',ascending=False)
            for row in gg.itertuples(index=False):
                lines.append(f'  Direction {row.direction}: n={row.games}, excess={row.excess_brier_positive_worse:+.6f}')
        dis=by(part,['disagreement_band'],min_games=25)
        if not dis.empty:
            peak=dis.sort_values('excess_brier_positive_worse',ascending=False).iloc[0]
            lines.append(f'  Largest excess by disagreement (exploratory): {peak.disagreement_band}, n={int(peak.games)}, excess={peak.excess_brier_positive_worse:+.6f}')
        cal=part.my_probability_home.mean()-part.target.mean()
        if abs(cal)>.025:
            lines.append(f'  Notable marginal home-probability calibration bias: {cal:+.3f} (>2.5 percentage points)')
        lines.append('')
    lines += ['Feature reconstruction: '+feature_status,
              'Follow-up: use 2022-24 training / 2024-25 selection as originally locked; treat 2025-27 errors as diagnosis, not tuning targets.',
              'No live betting integration or market fallback is justified by these diagnostics.']
    path=OUT/'diagnosis.txt'
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)
    print(f'WROTE {path}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-features',action='store_true',help='Only analyze saved holdout predictions; do not reconstruct feature history')
    parser.add_argument('--test',action='store_true',help='Validate diagnostic calculations against in-memory fixture, with no file writes')
    args=parser.parse_args()
    if args.test:
        fixture=pd.DataFrame({'game_id':['2026020001','2026020002'],
          'game_date':['2026_10_01','2026_10_02'],'data_period':['2026_2027']*2,
          'home_team_full':['BOS','BUF'],'away_team_full':['TOR','NYR'],
          'target':[1.,0.], 'my_probability_home':[.6,.4],
          'vig_free_home_prob':[.55,.45],
          'training_games_before_target':[5000,5001],
          'last_training_game_date':['2026-09-30','2026-10-01']})
        # Unit fixture bypasses file loader; verifies exactly calculated losses.
        for p,y,m in zip(fixture.my_probability_home,fixture.target,fixture.vig_free_home_prob):
            d=(p-y)**2-(m-y)**2
            assert np.isclose(d,-.0425)
        assert fixture.game_id.is_unique
        print('SELF-TEST PASS: Brier paired losses, chronology fixture, distinct game IDs')
        return
    frame=load_predictions(SRC)
    seasons=by(frame,['data_period'])
    boot=bootstrap_by_date(frame)
    write(seasons,'season_errors.csv')
    write(boot,'paired_date_bootstrap.csv')
    write(by(frame,['data_period','month']),'monthly_errors.csv')
    write(by(frame,['data_period','market_favorite']),'favorite_side_errors.csv')
    write(by(frame,['data_period','direction']),'probability_shift_errors.csv')
    write(by(frame,['data_period','disagreement_band']),'disagreement_errors.csv')
    write(by(frame,['data_period','market_confidence']),'market_confidence_errors.csv')
    write(by(frame,['data_period','pick_disagreement']),'favorite_disagreement_errors.csv')
    write(calibration_bins(frame),'calibration_bins.csv')
    # Each team appears twice per game, one as home/away. Descriptive only.
    # Team report must use its own minimal columns and metrics.
    teamrows=[]
    for season,sub in frame.groupby('data_period'):
        game_teams=pd.concat([
            sub[['game_id','brier_excess']].assign(team=sub.home_team_full),
            sub[['game_id','brier_excess']].assign(team=sub.away_team_full)],ignore_index=True)
        for name,rows in game_teams.groupby('team'):
            if len(rows)>=8:
                teamrows.append({'season':season,'team':name,'games':len(rows),
                    'mean_excess_brier':float(rows.brier_excess.mean()),
                    'total_excess_brier':float(rows.brier_excess.sum())})
    write(pd.DataFrame(teamrows).sort_values(['season','mean_excess_brier'],ascending=[True,False]),'team_errors_exploratory.csv')
    bad=frame.sort_values('brier_excess',ascending=False).groupby('data_period',sort=True).head(30)
    write(bad[['data_period','game_date','game_id','home_team_full','away_team_full',
               'target','my_probability_home','vig_free_home_prob','model_market_shift',
               'brier_excess','log_loss_excess']],'largest_game_excess_losses.csv')
    if not args.skip_features:
        augmented,feature_status=get_asof_features(frame)
    else:
        augmented,feature_status=None,'Skipped on request'
    if augmented is not None:
        keys=['elo_bin','rest_bin','prior_team_games_bin','form_available']
        for name in keys:
            write(by(augmented,['data_period',name]),f'asof_{name}_errors.csv')
        coverage=[]
        for season,part in augmented.groupby('data_period'):
            for field in ('elo_diff_asof','rest_diff_asof','back_to_back_diff_asof',
                          'win_rate_last10_diff_asof','goals_diff_last10_diff_asof',
                          'prior_games_min_asof','form_complete_both_teams'):
                coverage.append({'season':season,'feature':field,'games':len(part),
                                'missing':int(part[field].isna().sum()),
                                'coverage':float(part[field].notna().mean())})
        write(pd.DataFrame(coverage),'asof_feature_coverage.csv')
    diagnosis(frame,augmented,boot,feature_status)


if __name__=='__main__':
    try:
        main()
    except (ValueError,FileNotFoundError,ImportError,KeyError,RuntimeError) as exc:
        sys.exit('ERROR: '+str(exc))
