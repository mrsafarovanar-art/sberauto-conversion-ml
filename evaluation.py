"""Метрики, пороги и единая поправка вероятностей для взвешенного обучения."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss)


def undo_class_weight(probability, positive_weight=1.0):
    """p = q / (w*(1-q)+q). Поправка prior, не обученная калибровка."""
    q = np.asarray(probability, dtype=float)
    if positive_weight <= 0 or not np.isfinite(positive_weight):
        raise ValueError('Вес положительного класса должен быть положительным конечным числом')
    return q / (positive_weight * (1-q) + q)


def choose_threshold(y, probability, policy='f1', minimum_recall=.4):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if policy == 'f1':
        score = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
        i = np.argmax(score)
    elif policy == 'recall_floor':
        eligible = np.flatnonzero(recall[:-1] >= minimum_recall)
        i = eligible[np.argmax(precision[:-1][eligible])]
    else:
        raise ValueError(policy)
    return float(thresholds[i])


def metrics(y, probability, threshold):
    y = np.asarray(y)
    p = np.asarray(probability)
    prediction = (p >= threshold).astype(int)
    return {'rows': len(y), 'positive_rate': float(y.mean()),
            'roc_auc': float(roc_auc_score(y,p)) if np.unique(y).size==2 else None,
            'pr_auc_ap': float(average_precision_score(y,p)),
            'brier_score': float(brier_score_loss(y,p)), 'mean_probability': float(p.mean()),
            'threshold': float(threshold), 'precision': float(precision_score(y,prediction,zero_division=0)),
            'recall': float(recall_score(y,prediction,zero_division=0)),
            'f1': float(f1_score(y,prediction,zero_division=0)),
            'predicted_positive_fraction': float(prediction.mean()),
            'confusion_matrix': confusion_matrix(y,prediction,labels=[0,1]).tolist()}
