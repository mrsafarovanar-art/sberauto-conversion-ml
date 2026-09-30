"""Контракт, переносимость обработки и совпадение обучения/инференса."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from app import app
from features import RAW_FEATURES, make_features
from predict import Predictor

ROOT = Path(__file__).resolve().parent


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()
        cls.example = json.loads((ROOT/'examples/session.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_health(self):
        self.assertEqual(self.client.get('/health').json()['status'], 'ok')

    def test_binary_and_probability_agree(self):
        response = self.client.post('/predict', json=self.example)
        self.assertEqual(response.status_code, 200)
        self.assertIs(type(response.json()), int)
        details = self.client.post('/predict_proba', json=self.example).json()
        self.assertEqual(response.json(), int(details['probability'] >= details['threshold']))
        self.assertTrue(0 <= details['probability'] <= 1)

    def test_both_prediction_classes(self):
        for filename, expected in [('positive_prediction.json', 1), ('negative_prediction.json', 0)]:
            sample = json.loads((ROOT/'examples'/filename).read_text())
            self.assertEqual(self.client.post('/predict', json=sample).json(), expected)

    def test_missing_and_unknown_categories(self):
        for row in [{'utm_medium': 'new_unseen_channel'}, {'geo_city': 'Unknown City', 'device_os': None},
                    {'device_screen_resolution': 'bad-resolution', 'device_category': 'mobile'}]:
            response = self.client.post('/predict', json=row)
            self.assertEqual(response.status_code, 200)
            self.assertIn(response.json(), [0, 1])

    def test_invalid_payloads(self):
        for row in [{}, {'utm_medium': None}, {'device_model': 'only_unused_field'}, {'utm_medium': 123}, {'target': 1},
                    {'utm_medium': 'x'*513}, {'utm_medium': ['organic']}]:
            self.assertEqual(self.client.post('/predict', json=row).status_code, 422)
        self.assertEqual(self.client.post('/predict', content='{bad', headers={'Content-Type':'application/json'}).status_code, 422)

    def test_batch_parity_and_limits(self):
        single = self.client.post('/predict', json=self.example).json()
        self.assertEqual(self.client.post('/predict_batch', json={'sessions': [self.example]*3}).json(), [single]*3)
        for rows in [[], [self.example]*1001]:
            self.assertEqual(self.client.post('/predict_batch', json={'sessions':rows}).status_code, 422)

    def test_training_inference_parity(self):
        if not all((ROOT/'artifacts'/name).exists() for name in ['sessions_labeled.parquet', 'test_predictions.parquet']):
            self.skipTest('Requires prepared data and saved test predictions; see README.md')
        expected = pd.read_parquet(ROOT/'artifacts/test_predictions.parquet').head(100)
        data = pd.read_parquet(ROOT/'artifacts/sessions_labeled.parquet')
        sample = expected[['session_id']].merge(data, on='session_id', how='left', validate='one_to_one')
        actual = Predictor().probabilities(sample[RAW_FEATURES])
        np.testing.assert_allclose(actual, expected.probability, rtol=1e-12, atol=1e-12)

    def test_feature_order_and_uninformative_device_model(self):
        changed = dict(reversed(list(self.example.items())))
        changed['device_model'] = 'new_model'
        pd.testing.assert_frame_equal(make_features([self.example]), make_features([changed]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
