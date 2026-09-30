"""Единый контракт признаков для обучения и инференса."""
import re
import pandas as pd

RAW_FEATURES = [
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
    'device_category', 'device_os', 'device_brand', 'device_model',
    'device_screen_resolution', 'device_browser', 'geo_country', 'geo_city',
]
# 99% пропусков: поле принимается API, но не используется моделью.
CATEGORICAL = [c for c in RAW_FEATURES if c != 'device_model']
NUMERIC = ['screen_width', 'screen_height', 'screen_ratio', 'screen_missing',
           'is_organic', 'is_social', 'utm_missing_count', 'device_missing_count']
SOCIAL_SOURCES = {'QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt',
                  'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHN BGo'.replace(' ', ''),
                  'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'}
MISSING = '__MISSING__'


def make_features(data):
    """Детерминированная обработка без обучения на будущих строках/таргете."""
    frame = pd.DataFrame(data).reindex(columns=RAW_FEATURES).copy()
    for col in RAW_FEATURES:
        frame[col] = frame[col].astype('string').str.strip().replace(
            {'': pd.NA, '(not set)': pd.NA, 'nan': pd.NA, 'None': pd.NA, '<NA>': pd.NA})
    out = frame[CATEGORICAL].fillna(MISSING).astype(str)
    screen = frame.device_screen_resolution.str.extract(r'^(\d{1,5})x(\d{1,5})$').astype(float)
    valid = screen[0].between(1, 20000) & screen[1].between(1, 20000)
    out['screen_width'] = screen[0].where(valid, 0).astype(float)
    out['screen_height'] = screen[1].where(valid, 0).astype(float)
    out['screen_ratio'] = (screen[0] / screen[1]).where(valid, 0).astype(float)
    out['screen_missing'] = (~valid).astype(int)
    out['is_organic'] = frame.utm_medium.isin(['organic', 'referral', '(none)']).astype(int)
    out['is_social'] = frame.utm_source.isin(SOCIAL_SOURCES).astype(int)
    out['utm_missing_count'] = frame[[c for c in RAW_FEATURES if c.startswith('utm_')]].isna().sum(axis=1)
    out['device_missing_count'] = frame[[c for c in CATEGORICAL if c.startswith('device_')]].isna().sum(axis=1)
    return out[CATEGORICAL + NUMERIC]
