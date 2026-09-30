"""Защита временных границ и поправки взвешенных вероятностей."""
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from train import temporal_split, base_catboost
from evaluation import undo_class_weight, choose_threshold, metrics
from features import make_features, CATEGORICAL


class TemporalTests(unittest.TestCase):
    def test_date_boundaries(self):
        frame = pd.DataFrame({'visit_date': ['2021-05-19', '2021-10-31', '2021-11-01', '2021-11-30', '2021-12-01', '2021-12-31']})
        split = temporal_split(frame)
        for name, expected in [('train', [0, 1]), ('validation', [2, 3]), ('test', [4, 5])]:
            np.testing.assert_array_equal(np.flatnonzero(split[name]), expected)
        np.testing.assert_array_equal(sum(x.astype(int) for x in split.values()), np.ones(6))

    def test_weight_odds_inverse_and_endpoints(self):
        original = np.array([0, .01, .1, .5, 1.])
        weight = 30
        weighted = weight * original / (1 - original + weight * original)
        np.testing.assert_allclose(undo_class_weight(weighted, weight), original)
        np.testing.assert_allclose(undo_class_weight(original, 1), original)
        self.assertTrue((np.diff(undo_class_weight(np.linspace(0, 1, 101), weight)) >= 0).all())
        with self.assertRaises(ValueError):
            undo_class_weight(original, 0)

    def test_recall_policy(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
        probability = np.linspace(.01, .9, 10)
        threshold = choose_threshold(y, probability, 'recall_floor', .4)
        self.assertGreaterEqual(metrics(y, probability, threshold)['recall'], .4)

    def test_target_and_dates_never_enter_features(self):
        original = pd.DataFrame([{'utm_medium': 'organic', 'device_category': 'mobile'}])
        changed = original.assign(target=1, visit_date='2021-12-31', client_id='known', event_action='sub_submit_success')
        pd.testing.assert_frame_equal(make_features(original), make_features(changed))

    def test_randomized_search_explicit_fold(self):
        X = make_features(pd.DataFrame({'utm_medium': ['organic', 'cpc', 'banner'] * 30, 'geo_city': ['Moscow', 'Kazan'] * 45}))
        y = np.array([0, 1, 0, 0, 1, 0] * 15)
        fold = PredefinedSplit(np.r_[np.full(60, -1), np.zeros(30, dtype=int)])
        search = RandomizedSearchCV(base_catboost(), {'depth': [2, 3], 'iterations': [2]},
            n_iter=2, scoring={'roc_auc': 'roc_auc', 'average_precision': 'average_precision'},
            refit=False, cv=fold, error_score='raise', random_state=42)
        search.fit(X, y, cat_features=CATEGORICAL)
        self.assertEqual(len(search.cv_results_['params']), 2)
        self.assertTrue(np.isfinite(search.cv_results_['mean_test_roc_auc']).all())
        train, test = next(fold.split())
        self.assertEqual(max(train), 59)
        self.assertEqual(min(test), 60)


if __name__ == '__main__':
    unittest.main(verbosity=2)
