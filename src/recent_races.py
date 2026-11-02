"""Score a preserved forecast and reconstruct Mexico without its race results."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
from forecast import FEATURES,model,make_features
ROOT=Path(__file__).resolve().parents[1]

def score(forecast,actual,label):
    joined=forecast.merge(actual[['driverId','position']],on='driverId',how='outer',validate='one_to_one',indicator=True)
    unforecast=joined.loc[joined._merge.eq('right_only'),'driverId'].tolist()
    nonstarters=joined.loc[joined._merge.eq('left_only'),'driverId'].tolist()
    # Nonstarters cannot make the podium. New entrants retain missing probabilities.
    scored=joined.podium_probability.notna()
    joined['actual_podium']=joined.position.le(3).astype(int)
    joined['squared_probability_error']=(joined.podium_probability-joined.actual_podium)**2
    joined=joined.sort_values(['podium_probability','driver'],ascending=[False,True])
    joined.to_csv(ROOT/f'reports/{label}_driver_evaluation.csv',index=False)
    return {'edition':label,'race':forecast.race.iloc[0],'race_date':str(forecast.race_date.iloc[0]),
        'forecast_status':'Preserved pre-race forecast' if label=='cota_2026' else 'Retrospective reconstruction created 2026-11-02',
        'predicted_podium_shortlist':'; '.join(joined.head(3).driver),
        'actual_podium':'; '.join(actual.sort_values('position').head(3).apply(lambda x:x.givenName+' '+x.familyName,axis=1)),
        'podium_hits_of_3':int(joined.head(3).actual_podium.sum()),'brier':float(joined.loc[scored,'squared_probability_error'].mean()),
        'brier_scope':'Forecast roster only; nonstarters scored as no podium; unforecast entrants excluded explicitly',
        'unforecast_entrants':'; '.join(unforecast),'forecast_nonstarters':'; '.join(nonstarters),
        'scored_drivers':int(scored.sum()),
        'recent_form_brier':float(((joined.recent_form_probability-joined.actual_podium)**2).mean()),
        'team_form_brier':float(((joined.team_form_probability-joined.actual_podium)**2).mean())}

def run(data):
    original=ROOT/'forecasts/2026-10-12_preweekend_podium_forecast.csv'
    before=hashlib.sha256(original.read_bytes()).hexdigest()
    saved=pd.read_csv(original)
    cota=data[(data.season==2026)&data.raceDate.eq('2026-10-25')]
    mexico=data[(data.season==2026)&data.raceDate.eq('2026-11-01')]
    assert len(cota)>0 and len(mexico)>0
    schedule=pd.read_csv(ROOT/'data/raw/schedule_2026.csv',parse_dates=['raceDate'])
    target=schedule[schedule.raceDate.eq('2026-11-01')].iloc[0]
    prior=data[data.raceDate<pd.Timestamp('2026-10-26')].copy()
    assert prior.raceDate.max()==pd.Timestamp('2026-10-25')
    h,f=make_features(prior,target)
    train=h[h.season>=2019]
    estimator=model('Random forest').fit(train[FEATURES],train.podium)
    retro=f[['driverId','driver','team']].copy()
    retro['podium_probability']=estimator.predict_proba(f[FEATURES])[:,1]
    retro['recent_form_probability']=f.driver_podium5.to_numpy()
    retro['team_form_probability']=f.team_podium10.to_numpy()
    retro=retro.sort_values(['podium_probability','driver'],ascending=[False,True]).reset_index(drop=True)
    retro.insert(0,'podium_likelihood_rank',np.arange(1,len(retro)+1))
    retro['forecast_as_of']='2026-10-26';retro['created_on']='2026-11-02'
    retro['forecast_status']='retrospective reconstruction; not issued before race'
    retro['race']=target.raceName;retro['race_date']='2026-11-01'
    retro.to_csv(ROOT/'forecasts/2026-11-02_reconstructed_2026-11-01_mexico.csv',index=False)
    result=pd.DataFrame([score(saved,cota,'cota_2026'),score(retro,mexico,'mexico_2026_reconstructed')])
    result.to_csv(ROOT/'reports/recent_race_evaluation.csv',index=False)
    (ROOT/'reports/forecast_audit_2026-11-02.json').write_text(json.dumps({'created_on':'2026-11-02','original_cota_forecast_sha256':before,'mexico_information_cutoff':'2026-10-26','mexico_latest_training_race':'2026-10-25','method':'Original random forest selected on 2024, unchanged settings; latest previous race roster','limitation':'Current revised snapshots; historical reconstruction is not a contemporaneously issued forecast; no current-weekend practice, qualifying, sprint or weather'},indent=2))
    assert hashlib.sha256(original.read_bytes()).hexdigest()==before
    return result
