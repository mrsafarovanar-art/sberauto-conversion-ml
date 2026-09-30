"""Temporal validation, bounded RandomizedSearchCV and class-imbalance experiments."""
import argparse
import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.spatial.distance import jensenshannon
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features import RAW_FEATURES, CATEGORICAL, NUMERIC, make_features
from prepare import ROOT, prepare_data
from evaluation import metrics, choose_threshold, undo_class_weight

SEED = 42


def temporal_split(df):
    date = pd.to_datetime(df.visit_date, errors='raise')
    return {'train': (date < '2021-11-01').to_numpy(),
            'validation': ((date >= '2021-11-01') & (date < '2021-12-01')).to_numpy(),
            'test': ((date >= '2021-12-01') & (date < '2022-01-01')).to_numpy()}


def sample_indices(mask, y, limit):
    indices = np.flatnonzero(mask)
    if len(indices) <= limit:
        return indices
    return train_test_split(indices, train_size=limit, stratify=y[indices], random_state=SEED)[0]


def probability(model, X, weight=1.0):
    if isinstance(model, CatBoostClassifier):
        raw = model.predict_proba(X, thread_count=4)[:,1]
    else:
        raw = model.predict_proba(X)[:,1]
    return undo_class_weight(raw, weight)


def base_catboost(**params):
    config = dict(loss_function='Logloss', eval_metric='AUC', random_seed=SEED,
                  thread_count=4, allow_writing_files=False, verbose=False)
    config.update(params)
    return CatBoostClassifier(**config)


def drift_report(df, X, split):
    monthly = df.groupby(df.visit_date.dt.strftime('%Y-%m')).target.agg(['size','sum','mean']).reset_index()
    monthly.columns = ['month','sessions','target_sessions','CR']
    monthly.to_csv(ROOT/'reports/monthly_population.csv',index=False)
    rows = []
    for col in CATEGORICAL:
        train = X.loc[split['train'],col].value_counts(normalize=True)
        for name in ['validation','test']:
            other = X.loc[split[name],col].value_counts(normalize=True)
            categories = train.index.union(other.index)
            jsd = jensenshannon(train.reindex(categories,fill_value=0),other.reindex(categories,fill_value=0),base=2)**2
            rows.append({'feature':col,'period':name,'js_divergence_bits':float(jsd),
                         'unseen_fraction':float((~X.loc[split[name],col].isin(train.index)).mean())})
    pd.DataFrame(rows).to_csv(ROOT/'reports/feature_drift.csv',index=False)


def cluster_intervals(y, probabilities, clients, baseline, repetitions=100):
    codes, unique = pd.factorize(clients, sort=False)
    rng = np.random.default_rng(SEED)
    auc, delta = [], []
    for _ in range(repetitions):
        multiplicity = np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
        weights = multiplicity[codes]
        a = roc_auc_score(y, probabilities, sample_weight=weights)
        b = roc_auc_score(y, baseline, sample_weight=weights)
        auc.append(a); delta.append(a-b)
    return {'method':'client cluster bootstrap; percentile 95%; 100 repetitions, approximate',
            'auc_95':np.quantile(auc,[.025,.975]).tolist(),
            'auc_difference_vs_temporal_baseline_95':np.quantile(delta,[.025,.975]).tolist()}


def train_models(force=False):
    path = ROOT/'reports/metrics.json'
    if path.exists() and (ROOT/'artifacts/metadata.json').exists() and not force:
        return json.loads(path.read_text())
    started = time.perf_counter()
    artifacts = ROOT/'artifacts'; candidates_dir = artifacts/'candidates'; candidates_dir.mkdir(exist_ok=True)
    df,audit = prepare_data()
    df = df.loc[df.has_hits & df.visit_date.notna()].sort_values(['visit_date','session_id']).reset_index(drop=True)
    split = temporal_split(df)
    assert np.all(sum(mask.astype(int) for mask in split.values()) == 1)
    assert df.loc[split['train'],'visit_date'].max() < df.loc[split['validation'],'visit_date'].min()
    assert df.loc[split['validation'],'visit_date'].max() < df.loc[split['test'],'visit_date'].min()
    report = {'seed':SEED,'split_method':'train May-Oct; validation November; test December 2021',
              'selection_rule':'highest full-November ROC-AUC; AP breaks exact ties; no December model selection',
              'split':{},'validation':{},'test':{},'candidates':{},'threshold_rule':'maximum F1 on November',
              'retrospective_caveat':'Retrospective evaluation on historical data; prospective validation on later periods is required.'}
    for name,mask in split.items():
        part=df.loc[mask]
        report['split'][name]={'rows':len(part),'clients':int(part.client_id.nunique()),'positives':int(part.target.sum()),
            'positive_rate':float(part.target.mean()),'date_min':str(part.visit_date.min().date()),'date_max':str(part.visit_date.max().date())}
    train_clients=set(df.loc[split['train'],'client_id'])
    november_clients=set(df.loc[split['validation'],'client_id'])
    december_clients=set(df.loc[split['test'],'client_id'])
    report['client_overlap']={'train_validation':len(train_clients & november_clients),
                             'train_test':len(train_clients & december_clients),
                             'new_december_clients':len(december_clients-(train_clients|november_clients))}
    X=make_features(df); y=df.target.to_numpy()
    drift_report(df,X,split)
    print('SPLIT',json.dumps(report['split'],ensure_ascii=False),flush=True)
    xt,xv=X.loc[split['train']],X.loc[split['validation']]
    yt,yv=y[split['train']],y[split['validation']]
    ratio=float((yt==0).sum()/(yt==1).sum())
    report['positive_class_weight_balanced']=ratio
    validation_probs={}
    model_records={}

    def register(name,model,weight,params,elapsed):
        p=probability(model,xv,weight)
        validation_probs[name]=p
        threshold=choose_threshold(yv,p)
        report['validation'][name]=metrics(yv,p,threshold)
        report['candidates'][name]={'params':params,'positive_class_weight':weight,'seconds':elapsed,
            'tree_count':int(model.tree_count_) if isinstance(model,CatBoostClassifier) else None,
            'recall40_threshold':choose_threshold(yv,p,'recall_floor')}
        if isinstance(model,CatBoostClassifier):
            file=candidates_dir/f'{name}.cbm';model.save_model(str(file));kind='CatBoostClassifier'
        else:
            file=candidates_dir/f'{name}.joblib';joblib.dump(model,file);kind='LogisticRegressionPipeline'
        model_records[name]={'path':str(file),'model_type':kind,'weight':weight}
        print('VALIDATION',name,json.dumps(report['validation'][name]),flush=True)
        (ROOT/'reports/progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

    preprocessing=ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=30,sparse_output=True,dtype=np.float32),CATEGORICAL),
        ('num',StandardScaler(),NUMERIC)])
    linear=Pipeline([('preprocess',preprocessing),('model',LogisticRegression(C=1,max_iter=300,solver='liblinear',random_state=SEED))])
    t=time.perf_counter();linear.fit(xt,yt);register('logistic_regression',linear,1.0,{'C':1},time.perf_counter()-t)
    del linear

    baseline_params={'iterations':500,'depth':6,'learning_rate':.08,'l2_leaf_reg':5}
    def fit_cat(name,params,weight):
        model=base_catboost(**params,class_weights=[1,weight],verbose=100)
        t=time.perf_counter()
        model.fit(xt,yt,cat_features=CATEGORICAL,eval_set=(xv,yv),early_stopping_rounds=45)
        register(name,model,weight,params,time.perf_counter()-t)
        return model
    model=fit_cat('temporal_baseline',baseline_params,1.0); del model

    # One explicit train/validation fold. refit=False prevents training on validation.
    ti=sample_indices(split['train'],y,250000);vi=sample_indices(split['validation'],y,80000)
    search_indices=np.r_[ti,vi]
    fold=PredefinedSplit(np.r_[np.full(len(ti),-1),np.zeros(len(vi),dtype=int)])
    space={'iterations':[200,350], 'depth':[4,5,6], 'learning_rate':[.04,.08,.12],
           'l2_leaf_reg':[3,7,12], 'random_strength':[.5,1.5]}
    search=RandomizedSearchCV(base_catboost(),space,n_iter=6,
        scoring={'roc_auc':'roc_auc','average_precision':'average_precision'},
        refit=False,cv=fold,random_state=SEED,n_jobs=1,error_score='raise',verbose=2,return_train_score=False)
    t=time.perf_counter(); search.fit(X.iloc[search_indices],y[search_indices],cat_features=CATEGORICAL)
    cv=search.cv_results_
    trials=pd.DataFrame({'params':[json.dumps(p,sort_keys=True) for p in cv['params']],
                        'validation_roc_auc':cv['mean_test_roc_auc'],'validation_ap':cv['mean_test_average_precision'],
                        'fit_seconds':cv['mean_fit_time']})
    trials.to_csv(ROOT/'reports/random_search.csv',index=False)
    best_i=trials.sort_values(['validation_roc_auc','validation_ap'],ascending=False).index[0]
    best_params=cv['params'][best_i]
    report['random_search']={'space':space,'n_iter':6,'train_sample':len(ti),'validation_sample':len(vi),
                            'cv':'one chronological PredefinedSplit; refit=False','best_params':best_params,
                            'seconds':time.perf_counter()-t}
    print('SEARCH BEST',best_params,flush=True)
    del search
    for name,weight in [('tuned_unweighted',1.0),('tuned_sqrt_balanced',float(np.sqrt(ratio))),('tuned_balanced',ratio)]:
        model=fit_cat(name,best_params,weight);del model

    # Freeze model and threshold before any December labels/metrics enter selection.
    selected=max(report['validation'],key=lambda name:(report['validation'][name]['roc_auc'],report['validation'][name]['pr_auc_ap']))
    record=model_records[selected]
    threshold=report['validation'][selected]['threshold']
    metadata={'model_type':record['model_type'],'selected_model':selected,'threshold':threshold,
              'positive_class_weight':record['weight'],'probability_adjustment':'inverse class-weight odds; not fitted calibration',
              'threshold_rule':report['threshold_rule'],'split_method':report['split_method'],
              'input_features':RAW_FEATURES,'categorical_features':CATEGORICAL,'numeric_features':NUMERIC,
              'target_actions':audit['target_actions'],'fit_population':'May-Oct 2021 only',
              'validation_period':'November 2021','test_period':'December 2021','seed':SEED,
              'tree_count':report['candidates'][selected]['tree_count']}
    target=artifacts/('model.cbm' if record['model_type']=='CatBoostClassifier' else 'model.joblib')
    shutil.copyfile(record['path'],target)
    if record['model_type']=='CatBoostClassifier':
        with target.open('rb') as source, gzip.open(artifacts/'model.cbm.gz','wb',compresslevel=9) as packed:
            shutil.copyfileobj(source,packed)
    (artifacts/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    report['selected_model']=selected
    (ROOT/'reports/selection_frozen_before_test.json').write_text(json.dumps({'selected_model':selected,
        'threshold':threshold,'params':report['candidates'][selected],'selection_rule':report['selection_rule'],
        'model_sha256':hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))
    print('FROZEN',selected,threshold,flush=True)
    validation_predictions=pd.DataFrame({'target':yv,**validation_probs})
    validation_predictions.to_parquet(artifacts/'validation_predictions.parquet',index=False)

    def load(name):
        rec=model_records[name]
        if rec['model_type']=='CatBoostClassifier':
            model=CatBoostClassifier();model.load_model(rec['path']);return model
        return joblib.load(rec['path'])

    # Interpret selected model on validation, with all derived features recomputed.
    model=load(selected)
    rng=np.random.default_rng(SEED)
    ii=rng.choice(np.flatnonzero(split['validation']),12000,replace=False)
    raw=df.iloc[ii][RAW_FEATURES].reset_index(drop=True); yi=y[ii]
    reference=roc_auc_score(yi,probability(model,make_features(raw),record['weight']))
    importances=[]
    for col in RAW_FEATURES:
        delta=[]
        for _ in range(3):
            perm=raw.copy();perm[col]=perm[col].iloc[rng.permutation(len(raw))].to_numpy()
            delta.append(reference-roc_auc_score(yi,probability(model,make_features(perm),record['weight'])))
        importances.append({'feature':col,'auc_drop_mean':float(np.mean(delta)),'auc_drop_std':float(np.std(delta))})
    pd.DataFrame(importances).sort_values('auc_drop_mean',ascending=False).to_csv(ROOT/'reports/permutation_importance.csv',index=False)
    del model

    # All announced models are evaluated once on December for comparative reporting only.
    xe,ye=X.loc[split['test']],y[split['test']]
    test_probs={}
    threshold_policies=[]
    for name,rec in model_records.items():
        model=load(name);p=probability(model,xe,rec['weight']);test_probs[name]=p
        th=report['validation'][name]['threshold']
        report['test'][name]=metrics(ye,p,th)
        th40=report['candidates'][name]['recall40_threshold']
        for period,labels,probs in [('validation',yv,validation_probs[name]),('test',ye,p)]:
            for policy,cutoff in [('f1',th),('recall40',th40),('default_0.5',.5)]:
                row=metrics(labels,probs,cutoff)
                row.update(model=name,period=period,policy=policy);threshold_policies.append(row)
        if rec['weight']!=1:
            raw_p=model.predict_proba(xe,thread_count=4)[:,1]
            report['test'][name]['raw_weighted_brier_score']=metrics(ye,raw_p,.5)['brier_score']
        del model
    report['test']['constant_baseline']=metrics(ye,np.full(len(ye),yt.mean()),.5)
    pd.DataFrame(threshold_policies).drop(columns='confusion_matrix').to_csv(ROOT/'reports/threshold_policies.csv',index=False)
    report['test_default_threshold']=metrics(ye,test_probs[selected],.5)
    report['test_recall40_policy']=metrics(ye,test_probs[selected],report['candidates'][selected]['recall40_threshold'])
    pred=df.loc[split['test'],['session_id','client_id','visit_date','device_category','target']].copy()
    pred['probability']=test_probs[selected];pred['prediction']=(pred.probability>=threshold).astype(int)
    pred['client_cohort']=np.where(pred.client_id.isin(train_clients|november_clients),'returning','new')
    pred.to_parquet(artifacts/'test_predictions.parquet',index=False)
    pd.DataFrame({'target':ye,**test_probs}).to_parquet(artifacts/'test_model_probabilities.parquet',index=False)
    report['test_client_cohorts']={name:metrics(part.target,part.probability,threshold) for name,part in pred.groupby('client_cohort')}
    report['test_weekly']={str(week):metrics(part.target,part.probability,threshold) for week,part in pred.groupby(pred.visit_date.dt.to_period('W'))}
    report['bootstrap']=cluster_intervals(ye,test_probs[selected],pred.client_id,test_probs['temporal_baseline'])
    curves={};calibration=[]
    for name,probs in test_probs.items():
        fpr,tpr,_=roc_curve(ye,probs);prec,rec,_=precision_recall_curve(ye,probs)
        ri=np.unique(np.linspace(0,len(fpr)-1,min(1500,len(fpr))).astype(int))
        pi=np.unique(np.linspace(0,len(prec)-1,min(1500,len(prec))).astype(int))
        curves[name]={'fpr':fpr[ri].tolist(),'tpr':tpr[ri].tolist(),'precision':prec[pi].tolist(),'recall':rec[pi].tolist()}
        frac,mean=calibration_curve(ye,probs,n_bins=10,strategy='quantile')
        calibration.extend({'model':name,'mean_probability':float(p),'observed_rate':float(o)} for p,o in zip(mean,frac))
    (ROOT/'reports/curves.json').write_text(json.dumps(curves))
    pd.DataFrame(calibration).to_csv(ROOT/'reports/calibration.csv',index=False)
    test_indices=np.flatnonzero(split['test']);cp=test_probs[selected]
    for name,index in [('session',test_indices[0]),('positive_prediction',test_indices[np.argmax(cp)]),('negative_prediction',test_indices[np.argmin(cp)])]:
        row=df.iloc[index][RAW_FEATURES].astype(object).where(df.iloc[index][RAW_FEATURES].notna(),None).to_dict()
        (ROOT/f'examples/{name}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
    report['elapsed_seconds']=time.perf_counter()-started
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('RESULT',json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--force',action='store_true')
    train_models(parser.parse_args().force)
