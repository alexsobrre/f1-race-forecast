"""Race-level, chronological F1 podium forecasting through FastF1/Jolpica."""
from pathlib import Path
import json
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import fastf1
from fastf1.ergast import Ergast
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
AS_OF = '2026-10-12'
SEED = 42
FEATURES = ['driver_finish5', 'driver_grid5', 'driver_podium5', 'driver_completion5',
            'team_finish10', 'team_podium10', 'season_podium', 'circuit_finish3',
            'experience', 'history_count', 'field_size']


def setup():
    for folder in ['cache', 'data/raw', 'data/processed', 'reports', 'figures', 'forecasts', '.vscode']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(ROOT / 'cache'))


def download(refresh=False):
    """Cache season-level records and paginate; do not silently accept partial races."""
    setup()
    api = Ergast(limit=1000)
    all_results = []
    for season in range(2018, 2027):
        target = ROOT / f'data/raw/results_{season}.csv'
        if target.exists() and not refresh:
            frame = pd.read_csv(target)
        else:
            parts = []
            response = api.get_race_results(season=season)
            expected = response.total_results
            count = 0
            while True:
                for i, result in enumerate(response.content):
                    result = result.copy()
                    info = response.description.iloc[i]
                    for key in ['season', 'round', 'raceName', 'raceDate', 'circuitId']:
                        result[key] = info[key]
                    parts.append(result)
                    count += len(result)
                if count >= expected:
                    break
                time.sleep(.4)
                response = response.get_next_result_page()
            frame = pd.concat(parts, ignore_index=True)
            assert len(frame) == expected, (season, len(frame), expected)
            columns = ['season', 'round', 'raceName', 'raceDate', 'circuitId', 'driverId',
                       'givenName', 'familyName', 'constructorId', 'constructorName',
                       'position', 'grid', 'status', 'points', 'laps']
            frame[columns].to_csv(target, index=False)
            frame = frame[columns]
        print(f'{season}: {len(frame)} driver-race records', flush=True)
        all_results.append(frame)
    data = pd.concat(all_results, ignore_index=True)
    data['raceDate'] = pd.to_datetime(data.raceDate)
    data = data[data.raceDate <= pd.Timestamp(AS_OF)].copy()
    assert not data.duplicated(['season', 'round', 'driverId']).any()
    data['race_id'] = data.season.astype(str) + '-' + data['round'].astype(str)
    data['field_size'] = data.groupby('race_id').driverId.transform('size')
    podium_counts = data.assign(podium=data.position <= 3).groupby('race_id').podium.sum()
    assert podium_counts.eq(3).all(), 'Source race has incomplete podium results'
    data = data.sort_values(['raceDate', 'season', 'round', 'position']).reset_index(drop=True)
    schedule_path = ROOT / 'data/raw/schedule_2026.csv'
    if schedule_path.exists() and not refresh:
        schedule = pd.read_csv(schedule_path)
    else:
        schedule = api.get_race_schedule(season=2026)
        schedule.to_csv(schedule_path, index=False)
    schedule['raceDate'] = pd.to_datetime(schedule.raceDate)
    future = schedule[schedule.raceDate > pd.Timestamp(AS_OF)].sort_values('raceDate')
    if future.empty:
        raise ValueError('No upcoming scheduled race after forecast cutoff')
    next_race = future.iloc[0]
    latest = data.iloc[-1]
    schedule_past = schedule[schedule.raceDate <= pd.Timestamp(AS_OF)]
    if schedule_past['round'].max() != data[data.season == 2026]['round'].max():
        raise ValueError('Latest scheduled race results unavailable: refuse stale forecast')
    manifest = {'as_of': AS_OF, 'retrieved_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
                'fastf1_version': fastf1.__version__, 'rows': len(data),
                'races': data.race_id.nunique(), 'latest_race': latest.raceName,
                'latest_date': latest.raceDate.date().isoformat(),
                'next_race': next_race.raceName, 'next_date': next_race.raceDate.date().isoformat(),
                'endpoint': 'FastF1 Ergast interface / api.jolpi.ca',
                'roster_assumption': 'Latest completed race entrants carried forward; provisional.'}
    (ROOT / 'data/manifest.json').write_text(json.dumps(manifest, indent=2))
    return data, next_race, manifest


def make_features(data, next_race):
    """Read state for every entrant before updating it with any current-race outcomes."""
    drivers, teams, circuits, seasons = (defaultdict(list) for _ in range(4))
    rows = []
    def calculate(row, race_date):
        d = drivers[row.driverId]
        t = teams[row.constructorId][-10:]
        c = circuits[(row.driverId, row.circuitId)][-3:]
        s = seasons[(row.driverId, row.season)]
        prior = 3 / row.field_size
        recent = d[-5:]
        return {'driver_finish5': np.mean([x['finish'] for x in recent]) if recent else .5,
                'driver_grid5': np.mean([x['grid'] for x in recent if np.isfinite(x['grid'])]) if any(np.isfinite(x['grid']) for x in recent) else .5,
                'driver_podium5': (sum(x['podium'] for x in recent) + 2 * prior) / (len(recent) + 2),
                'driver_completion5': np.mean([x['completed'] for x in recent]) if recent else .8,
                'team_finish10': np.mean([x['finish'] for x in t]) if t else .5,
                'team_podium10': (sum(x['podium'] for x in t) + 2 * prior) / (len(t) + 2),
                'season_podium': (sum(s) + 4 * prior) / (len(s) + 4),
                'circuit_finish3': np.mean(c) if c else .5,
                'experience': min(len(d), 150), 'history_count': len(recent),
                'field_size': row.field_size,
                'history_max_date': max((x['date'] for x in d), default=pd.NaT)}
    for race_id, race in data.groupby('race_id', sort=False):
        for row in race.itertuples():
            f = calculate(row, row.raceDate)
            f.update({'race_id': race_id, 'season': row.season, 'round': row.round,
                      'raceDate': row.raceDate, 'raceName': row.raceName, 'driverId': row.driverId,
                      'driver': row.givenName + ' ' + row.familyName, 'team': row.constructorName,
                      'podium': int(row.position <= 3), 'position': row.position})
            rows.append(f)
        # All entrants must see the same pre-race information boundary.
        for row in race.itertuples():
            record = {'finish': (row.position - 1) / (row.field_size - 1),
                      'grid': (row.grid - 1) / (row.field_size - 1) if row.grid > 0 else np.nan,
                      'podium': int(row.position <= 3),
                      'completed': int(row.status == 'Finished' or str(row.status).startswith('+')),
                      'date': row.raceDate}
            drivers[row.driverId].append(record)
            teams[row.constructorId].append(record)
            circuits[(row.driverId, row.circuitId)].append(record['finish'])
            seasons[(row.driverId, row.season)].append(record['podium'])
    historical = pd.DataFrame(rows)
    valid = historical.history_max_date.notna()
    assert (historical.loc[valid, 'history_max_date'] < historical.loc[valid, 'raceDate']).all()
    latest = data[data.race_id == data.iloc[-1].race_id].copy()
    future_rows = []
    for _, row in latest.iterrows():
        row['season'] = int(next_race.season)
        row['circuitId'] = next_race.circuitId
        f = calculate(row, next_race.raceDate)
        f.update({'driverId': row.driverId, 'driver': row.givenName + ' ' + row.familyName,
                  'team': row.constructorName, 'raceName': next_race.raceName,
                  'raceDate': next_race.raceDate})
        future_rows.append(f)
    historical.to_csv(ROOT / 'data/processed/driver_race_features.csv', index=False)
    future = pd.DataFrame(future_rows)
    assert (future.history_max_date < future.raceDate).all()
    future.to_csv(ROOT / 'data/processed/next_race_features.csv', index=False)
    return historical, future


def model(name):
    if name == 'Logistic regression':
        estimator = LogisticRegression(C=1., max_iter=2000)
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
    if name == 'Random forest':
        return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(
            n_estimators=120, max_depth=6, min_samples_leaf=15, random_state=SEED, n_jobs=2))
    raise ValueError(name)


def backtest(historical):
    """2024 development selects a model; 2025 and 2026 remain later evaluations."""
    candidates = ['Field prevalence', 'Recent podium form', 'Team podium form',
                  'Logistic regression', 'Random forest']
    rows = []
    for race_id, race in historical[historical.season >= 2024].groupby('race_id', sort=False):
        train = historical[(historical.season >= 2019) & (historical.raceDate < race.raceDate.iloc[0])]
        assert set(train.race_id).isdisjoint(set(race.race_id))
        print('Backtest', race_id, race.raceName.iloc[0], flush=True)
        for name in candidates:
            if name == 'Field prevalence':
                p = np.full(len(race), 3 / len(race))
            elif name == 'Recent podium form':
                p = race.driver_podium5.to_numpy()
            elif name == 'Team podium form':
                p = race.team_podium10.to_numpy()
            else:
                estimator = model(name)
                estimator.fit(train[FEATURES], train.podium)
                p = estimator.predict_proba(race[FEATURES])[:, 1]
            for (_, row), probability in zip(race.iterrows(), p):
                rows.append({'race_id': race_id, 'season': row.season,
                             'raceDate': row.raceDate, 'driver': row.driver,
                             'model': name, 'actual_podium': row.podium, 'p_podium': probability})
    predictions = pd.DataFrame(rows)
    predictions.to_csv(ROOT / 'reports/backtest_predictions.csv', index=False)
    metrics = evaluate(predictions)
    metrics.to_csv(ROOT / 'reports/backtest_metrics.csv', index=False)
    development = metrics[metrics.season == 2024].set_index('model')
    selected = development.loc[['Logistic regression', 'Random forest'], 'brier'].idxmin()
    # Selection uses probability accuracy on 2024 only. Baselines remain comparators.
    return predictions, metrics, selected


def evaluate(predictions):
    out = []
    for (season, name), group in predictions.groupby(['season', 'model']):
        race_metrics = []
        for _, race in group.groupby('race_id', sort=False):
            # Alphabetical tie-breaking is fixed, deterministic, and outcome-independent.
            ordered = race.sort_values(['p_podium', 'driver'], ascending=[False, True])
            hits = ordered.iloc[:3].actual_podium.sum()
            race_metrics.append((brier_score_loss(race.actual_podium, race.p_podium),
                                 log_loss(race.actual_podium, np.clip(race.p_podium, 1e-6, 1-1e-6), labels=[0,1]),
                                 hits, int(hits == 3)))
        arr = np.asarray(race_metrics)
        out.append({'season': season, 'model': name, 'races': len(arr),
                    'brier': arr[:,0].mean(), 'log_loss': arr[:,1].mean(),
                    'mean_podium_hits_of_3': arr[:,2].mean(), 'exact_podium_set_rate': arr[:,3].mean(),
                    'average_precision': average_precision_score(group.actual_podium, group.p_podium),
                    'roc_auc': roc_auc_score(group.actual_podium, group.p_podium)})
    return pd.DataFrame(out)


def race_bootstrap(predictions, selected, season=2025):
    """Paired race-block bootstrap: keep all drivers from a race together."""
    subset = predictions[predictions.season == season]
    losses = subset.assign(squared_error=(subset.p_podium-subset.actual_podium)**2).groupby(['race_id','model']).squared_error.mean().unstack()
    rng = np.random.default_rng(SEED)
    intervals = []
    for baseline in ['Recent podium form','Team podium form']:
        delta = (losses[selected]-losses[baseline]).to_numpy()
        samples = rng.choice(delta, size=(2000,len(delta)), replace=True).mean(axis=1)
        intervals.append({'season':season,'selected':selected,'baseline':baseline,
                          'brier_difference':delta.mean(),'ci_low':np.quantile(samples,.025),'ci_high':np.quantile(samples,.975)})
    result = pd.DataFrame(intervals)
    result.to_csv(ROOT/'reports/paired_race_bootstrap.csv',index=False)
    return result


def predict_next(historical, future, selected, manifest):
    train = historical[historical.season >= 2019]
    estimator = model(selected).fit(train[FEATURES], train.podium)
    forecast = future[['driverId','driver','team']].copy()
    forecast['podium_probability'] = estimator.predict_proba(future[FEATURES])[:,1]
    forecast['recent_form_probability'] = future.driver_podium5.to_numpy()
    forecast['team_form_probability'] = future.team_podium10.to_numpy()
    forecast = forecast.sort_values(['podium_probability','driver'],ascending=[False,True]).reset_index(drop=True)
    forecast.insert(0,'podium_likelihood_rank',np.arange(1,len(forecast)+1))
    forecast['forecast_as_of'] = AS_OF
    forecast['race'] = manifest['next_race']
    forecast['race_date'] = manifest['next_date']
    filename=f'{AS_OF}_preweekend_podium_forecast.csv'
    target = ROOT/'forecasts'/filename
    if target.exists():
        recorded = pd.read_csv(target)
        same_drivers = recorded.driverId.tolist() == forecast.driverId.tolist()
        same_probabilities = len(recorded) == len(forecast) and np.allclose(
            recorded.podium_probability, forecast.podium_probability, atol=1e-10, rtol=0)
        if not same_drivers or not same_probabilities:
            raise FileExistsError('Dated forecast already exists with different predictions. Preserve it and use a new explicitly versioned forecast.')
        # An identical rerun preserves the original prospective artifact.
    else:
        forecast.to_csv(target,index=False)
    return forecast, estimator
