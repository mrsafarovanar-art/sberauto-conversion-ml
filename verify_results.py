"""Независимые проверки сохранённого декабрьского результата."""
import hashlib
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parent

def verify():
    results=json.loads((ROOT/'reports/metrics.json').read_text())
    frozen=json.loads((ROOT/'reports/selection_frozen_before_test.json').read_text())
    metadata=json.loads((ROOT/'artifacts/metadata.json').read_text())
    selected=results['selected_model']
    winner=max(results['validation'],key=lambda name:(results['validation'][name]['roc_auc'],results['validation'][name]['pr_auc_ap']))
    assert selected==winner==frozen['selected_model']==metadata['selected_model']
    filename='model.cbm' if metadata['model_type']=='CatBoostClassifier' else 'model.joblib'
    model_path=ROOT/'artifacts'/filename
    if not model_path.exists() and filename=='model.cbm':
        with gzip.open(ROOT/'artifacts/model.cbm.gz','rb') as packed:
            model_bytes=packed.read()
    else:
        model_bytes=model_path.read_bytes()
    assert hashlib.sha256(model_bytes).hexdigest()==frozen['model_sha256']
    pred=pd.read_parquet(ROOT/'artifacts/test_predictions.parquet')
    source=pd.read_parquet(ROOT/'artifacts/sessions_labeled.parquet')
    december=source.loc[source.has_hits & source.visit_date.ge('2021-12-01') & source.visit_date.lt('2022-01-01')]
    assert set(pred.session_id)==set(december.session_id)
    assert pred.session_id.is_unique and pred.visit_date.ge('2021-12-01').all() and pred.visit_date.lt('2022-01-01').all()
    check=pred[['session_id','target']].merge(december[['session_id','target']],on='session_id',validate='one_to_one')
    assert check.target_x.eq(check.target_y).all()
    y=pred.target.to_numpy();p=pred.probability.to_numpy()
    assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all()
    n1=y.sum();n0=len(y)-n1
    auc=(rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
    expected=results['test'][selected]
    assert np.isclose(auc,expected['roc_auc'],atol=1e-12,rtol=0)
    predicted=(p>=frozen['threshold']).astype(int)
    np.testing.assert_array_equal(predicted,pred.prediction)
    tn=int(((y==0)&(predicted==0)).sum());fp=int(((y==0)&(predicted==1)).sum())
    fn=int(((y==1)&(predicted==0)).sum());tp=int(((y==1)&(predicted==1)).sum())
    assert [[tn,fp],[fn,tp]]==expected['confusion_matrix']
    assert np.isclose(tp/(tp+fp),expected['precision'])
    assert np.isclose(tp/(tp+fn),expected['recall'])
    train=source.loc[source.has_hits & source.visit_date.lt('2021-11-01')]
    weight=(train.target==0).sum()/(train.target==1).sum()
    assert np.isclose(weight,results['positive_class_weight_balanced'])
    assert results['candidates']['tuned_unweighted']['params']==results['candidates']['tuned_sqrt_balanced']['params']==results['candidates']['tuned_balanced']['params']
    data={'status':'passed','december_rows':len(pred),'independent_rank_sum_auc':float(auc),
          'confusion_matrix':[[tn,fp],[fn,tp]],'frozen_model_hash_matches':True,
          'selected_by_november_only':True,'december_membership_and_targets_match_source':True,
          'class_weights_match_train_only':True}
    (ROOT/'reports/independent_verification.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2))
    return data

if __name__=='__main__':verify()
