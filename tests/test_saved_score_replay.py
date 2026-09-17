"""Deterministic boundary checks for saved-score queue replay."""
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research.replay import replay

class ReplayTests(unittest.TestCase):
    def run_replay(self,ids,sources,times,scores,capacity=1.):
        return replay(ids,np.asarray(sources,float),[np.asarray(t,float) for t in times],np.asarray(scores,float),capacity,'combined_catboost_ranker')

    def test_same_time_observed_and_ties(self):
        summary,decisions=self.run_replay(['b','a'],[0,0],[[1,2,3],[1,2,3]],[[0,0,0],[0,0,0]])
        self.assertEqual([r['thread_id'] for r in decisions],['a','b'])
        self.assertEqual(decisions[0]['observed_reactions'],1)
        self.assertEqual(decisions[0]['post_review_growth'],2)
        self.assertEqual(summary['future_growth_denominator'],4)
        self.assertEqual(summary['captured_post_review_growth'],3)

    def test_empty_slots_are_discarded(self):
        summary,decisions=self.run_replay(['a','b'],[0,100],[[2],[2]],[[0,0,0],[0,0,0]])
        self.assertEqual(summary['service_slots'],148)
        self.assertEqual([r['review_time_hours'] for r in decisions],[1.,101.])
        self.assertEqual(summary['reviewed'],2)

    def test_age_48_inclusive_and_no_duplicate_review(self):
        ids=[str(i).zfill(2) for i in range(49)]
        summary,decisions=self.run_replay(ids,[0]*49,[[50]]*49,np.zeros((49,3)))
        self.assertEqual(summary['reviewed'],48)
        self.assertEqual(summary['expired_or_unreviewed'],1)
        self.assertEqual(decisions[-1]['thread_age_hours'],48.)
        self.assertEqual(len({r['thread_id'] for r in decisions}),48)
        self.assertEqual(decisions[5]['snapshot_window_hours'],6)
        self.assertEqual(decisions[23]['snapshot_window_hours'],24)

    def test_noninteger_capacity_stops_on_service_grid(self):
        summary,decisions=self.run_replay(['a'],[0],[[2]],[[0,0,0]],.3)
        self.assertEqual(summary['service_slots'],15)
        self.assertEqual(summary['reviewed'],1)

if __name__=='__main__':unittest.main()
