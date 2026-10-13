"""Tests target actual forecast leakage risks, not model scores."""
import sys, tempfile, unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import forecast

class InformationBoundary(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.original_root=forecast.ROOT
        forecast.ROOT=Path(self.tmp.name)
        (forecast.ROOT/'data/processed').mkdir(parents=True)
        rows=[]
        for round_number in range(1,4):
            for driver in range(6):
                rows.append({'season':2025,'round':round_number,'race_id':f'2025-{round_number}',
                  'raceDate':pd.Timestamp(f'2025-03-{round_number*7:02}'), 'raceName':f'Race {round_number}',
                  'driverId':f'd{driver}','givenName':'Driver','familyName':str(driver),
                  'constructorId':f't{driver//2}','constructorName':f'Team {driver//2}',
                  'circuitId':'test','position':driver+1,'grid':driver+1,'status':'Finished','field_size':6})
        self.data=pd.DataFrame(rows)
        self.next=pd.Series({'season':2025,'circuitId':'test','raceDate':pd.Timestamp('2025-03-28'),'raceName':'Future'})
    def tearDown(self):
        forecast.ROOT=self.original_root
        self.tmp.cleanup()
    def test_current_race_outcomes_do_not_change_its_features(self):
        before,_=forecast.make_features(self.data,self.next)
        altered=self.data.copy()
        mask=altered['round']==2
        altered.loc[mask,'position']=7-altered.loc[mask,'position']
        after,_=forecast.make_features(altered,self.next)
        pd.testing.assert_frame_equal(before.loc[before['round']==2,forecast.FEATURES],after.loc[after['round']==2,forecast.FEATURES])
        self.assertFalse(before.loc[before['round']==3,forecast.FEATURES].equals(after.loc[after['round']==3,forecast.FEATURES]))
    def test_future_forecast_uses_latest_completed_race(self):
        _,before=forecast.make_features(self.data,self.next)
        altered=self.data.copy()
        mask=altered['round']==3
        altered.loc[mask,'position']=7-altered.loc[mask,'position']
        _,after=forecast.make_features(altered,self.next)
        self.assertFalse(before[forecast.FEATURES].equals(after[forecast.FEATURES]))
        self.assertTrue((after.history_max_date < after.raceDate).all())

if __name__=='__main__': unittest.main()
