"""CLI: python predict.py --input examples/session.json -> 0 или 1."""
import argparse
import gzip
import json
import sys
import tempfile
from pathlib import Path
from catboost import CatBoostClassifier
import joblib
from features import make_features
from evaluation import undo_class_weight

ROOT = Path(__file__).resolve().parent


class Predictor:
    def __init__(self, artifacts_dir=None):
        folder = Path(artifacts_dir) if artifacts_dir else ROOT/'artifacts'
        self.metadata = json.loads((folder/'metadata.json').read_text())
        self.threshold = float(self.metadata['threshold'])
        self.positive_class_weight = float(self.metadata.get('positive_class_weight', 1.0))
        if self.metadata['model_type'] == 'CatBoostClassifier':
            self.model = CatBoostClassifier()
            compressed = folder/'model.cbm.gz'
            if compressed.exists():
                with tempfile.TemporaryDirectory(prefix='sberauto-model-') as temporary:
                    model_path = Path(temporary)/'model.cbm'
                    with gzip.open(compressed, 'rb') as source, model_path.open('wb') as target:
                        import shutil
                        shutil.copyfileobj(source, target)
                    self.model.load_model(str(model_path))
            else:
                self.model.load_model(str(folder/'model.cbm'))
        elif self.metadata['model_type'] == 'LogisticRegressionPipeline':
            self.model = joblib.load(folder/'model.joblib')
        else:
            raise ValueError('Неизвестный формат модели')

    def probabilities(self, rows):
        X = make_features(rows)
        if isinstance(self.model, CatBoostClassifier):
            raw = self.model.predict_proba(X, thread_count=1)[:, 1]
        else:
            raw = self.model.predict_proba(X)[:, 1]
        return undo_class_weight(raw, self.positive_class_weight)

    def predict(self, rows):
        return (self.probabilities(rows) >= self.threshold).astype(int).tolist()


def main():
    parser = argparse.ArgumentParser(description='Прогноз целевого действия по JSON с utm_*, device_*, geo_*')
    parser.add_argument('--input', type=Path, help='JSON-файл; если не указан, читается stdin')
    args = parser.parse_args()
    payload = json.loads(args.input.read_text() if args.input else sys.stdin.read())
    # Тот же валидатор, что в HTTP API.
    from schema import SessionInput
    row = SessionInput.model_validate(payload).model_dump()
    print(Predictor().predict([row])[0])


if __name__ == '__main__':
    main()
