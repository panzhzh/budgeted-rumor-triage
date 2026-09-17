"""Regression checks for high-growth metrics with tied or degenerate inputs."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from research.triage import _ranking_metrics

class RankingMetricTests(unittest.TestCase):
    def metrics(self, labels, scores):
        return _ranking_metrics(np.zeros(len(labels)), np.asarray(labels), np.asarray(scores))

    def test_constant_scores_retain_chance_performance(self):
        self.assertEqual(self.metrics([0, 0, 0, 1], [0.5] * 4),
                         {'roc_auc': 0.5, 'average_precision': 0.25})

    def test_all_positive_labels_have_defined_ap(self):
        self.assertEqual(self.metrics([1, 1, 1], [0, 0, 0]),
                         {'roc_auc': None, 'average_precision': 1.0})

    def test_no_positive_labels_follow_reporting_convention(self):
        self.assertEqual(self.metrics([0, 0, 0], [0, 1, 2]),
                         {'roc_auc': None, 'average_precision': None})

    def test_perfect_and_reversed_rankings(self):
        self.assertEqual(self.metrics([0, 1], [0, 1]),
                         {'roc_auc': 1.0, 'average_precision': 1.0})
        self.assertEqual(self.metrics([0, 1], [1, 0]),
                         {'roc_auc': 0.0, 'average_precision': 0.5})

    def test_nonfinite_scores_are_execution_errors(self):
        for invalid in [np.nan, np.inf, -np.inf]:
            with self.assertRaises(ValueError):
                self.metrics([0, 1], [0, invalid])

    def test_empty_and_misaligned_inputs(self):
        self.assertEqual(self.metrics([], []), {'roc_auc': None, 'average_precision': None})
        with self.assertRaises(ValueError):
            self.metrics([0, 1], [0])

if __name__ == '__main__':
    unittest.main()
