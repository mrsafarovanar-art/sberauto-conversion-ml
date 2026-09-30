"""Чтение CSV частями, аудит данных и целевая переменная на уровне сессии."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TARGET_ACTIONS = [
    'sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
    'sub_custom_question_submit_click', 'sub_call_number_click',
    'sub_callback_submit_click', 'sub_submit_success', 'sub_car_request_submit_click',
]


def prepare_data(data_dir=None, force=False):
    data_dir = Path(data_dir) if data_dir else ROOT.parent
    cache = ROOT / 'artifacts/sessions_labeled.parquet'
    audit_path = ROOT / 'reports/data_audit.json'
    if cache.exists() and audit_path.exists() and not force:
        return pd.read_parquet(cache), json.loads(audit_path.read_text())
    sessions_file = data_dir / 'skillbox_diploma_main_dataset_sberautopodpiska/ga_sessions.csv'
    hits_file = data_dir / 'ga_hits-002.csv'
    sessions = pd.read_csv(sessions_file, dtype='string')
    audit = {'sessions_raw': len(sessions), 'sessions_columns': list(sessions.columns),
             'sessions_missing_fraction': sessions.isna().mean().to_dict(),
             'sessions_cardinality': sessions.nunique().to_dict(),
             'sessions_exact_duplicates': int(sessions.duplicated().sum())}
    sessions = sessions.drop_duplicates()
    if sessions.session_id.isna().any() or sessions.session_id.duplicated().any():
        raise ValueError('Пустые или конфликтующие session_id в таблице сессий')
    positives, observed = set(), set()
    missing, actions, daily = None, None, None
    hashes = []
    n = 0
    for chunk in pd.read_csv(hits_file, dtype='string', chunksize=250_000):
        n += len(chunk)
        missing = chunk.isna().sum() if missing is None else missing.add(chunk.isna().sum(), fill_value=0)
        counts = chunk.event_action.value_counts(dropna=False)
        actions = counts if actions is None else actions.add(counts, fill_value=0)
        counts = chunk.hit_date.value_counts()
        daily = counts if daily is None else daily.add(counts, fill_value=0)
        # Хеш используется только для аудита дублей, не для соединения/таргета.
        hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
        observed.update(chunk.session_id.dropna())
        positives.update(chunk.loc[chunk.event_action.isin(TARGET_ACTIONS), 'session_id'].dropna())
        if n % 1_000_000 == 0:
            print(f'Прочитано событий: {n:,}', flush=True)
    all_hashes = np.concatenate(hashes)
    audit['hits_hash_duplicate_rows'] = int(len(all_hashes) - len(np.unique(all_hashes)))
    del hashes, all_hashes
    ids = set(sessions.session_id)
    audit.update(hits_raw=n, hits_unique_sessions=len(observed),
                 hits_missing_fraction=(missing / n).to_dict(),
                 sessions_without_hits=len(ids - observed),
                 hit_sessions_without_session=len(observed - ids),
                 target_sessions_in_hits=len(positives),
                 target_sessions_without_session=len(positives - ids),
                 target_actions=TARGET_ACTIONS)
    # Бинарная агрегация max/any: повторы событий не размножают строки и не меняют y.
    sessions['has_hits'] = sessions.session_id.isin(observed)
    sessions['target'] = sessions.session_id.isin(positives).astype('int8')
    sessions['visit_date'] = pd.to_datetime(sessions.visit_date, errors='coerce')
    sessions['visit_number'] = pd.to_numeric(sessions.visit_number, errors='coerce')
    sessions['visit_hour'] = pd.to_datetime(sessions.visit_time, format='%H:%M:%S', errors='coerce').dt.hour
    audit['invalid_dates'] = int(sessions.visit_date.isna().sum())
    audit['invalid_visit_numbers'] = int(sessions.visit_number.isna().sum())
    audit['target_count_matched'] = int(sessions.target.sum())
    audit['date_min'] = str(sessions.visit_date.min().date())
    audit['date_max'] = str(sessions.visit_date.max().date())
    audit['conversion_all_sessions'] = float(sessions.target.mean())
    audit['conversion_observed_sessions'] = float(sessions.loc[sessions.has_hits, 'target'].mean())
    ROOT.joinpath('artifacts').mkdir(exist_ok=True)
    ROOT.joinpath('reports').mkdir(exist_ok=True)
    sessions.to_parquet(cache, index=False)
    actions.rename_axis('event_action').reset_index(name='count').to_csv(ROOT/'reports/event_counts.csv', index=False)
    daily.rename_axis('date').reset_index(name='count').sort_values('date').to_csv(ROOT/'reports/hits_daily.csv', index=False)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return sessions, audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    prepare_data(args.data_dir, args.force)
