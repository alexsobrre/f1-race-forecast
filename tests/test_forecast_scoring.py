"""A roster mismatch must remain visible rather than silently improving the score."""
import unittest,tempfile,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import recent_races
class RosterScoring(unittest.TestCase):
 def test_replacement_is_explicit_and_nonstarter_is_not_a_podium(self):
  with tempfile.TemporaryDirectory() as tmp:
   old=recent_races.ROOT;recent_races.ROOT=Path(tmp);(Path(tmp)/'reports').mkdir()
   try:
    f=pd.DataFrame({'driverId':['a','b','c'],'driver':['A','B','C'],'podium_probability':[.8,.5,.2],'recent_form_probability':[.7,.4,.1],'team_form_probability':[.6,.3,.1],'race':['Test']*3,'race_date':['2026-11-01']*3})
    actual=pd.DataFrame({'driverId':['a','b','new'],'position':[1,2,3],'givenName':['A','B','New'],'familyName':['Driver']*3})
    result=recent_races.score(f,actual,'mexico_2026_reconstructed')
    self.assertEqual(result['unforecast_entrants'],'new');self.assertEqual(result['forecast_nonstarters'],'c')
    self.assertEqual(result['podium_hits_of_3'],2);self.assertEqual(result['scored_drivers'],3)
    self.assertAlmostEqual(result['brier'],((.8-1)**2+(.5-1)**2+.2**2)/3)
    audit=pd.read_csv(Path(tmp)/'reports/mexico_2026_reconstructed_driver_evaluation.csv')
    self.assertTrue(audit.loc[audit.driverId=='new','podium_probability'].isna().all())
   finally:recent_races.ROOT=old
