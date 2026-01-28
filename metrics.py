import numpy as np
from sklearn.metrics import roc_curve, auc


def calculate_apcer(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    apcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    return apcer


def calculate_bpcer(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    bpcer = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return bpcer


def calculate_ace(y_true, y_pred):
    apcer = calculate_apcer(y_true, y_pred)
    bpcer = calculate_bpcer(y_true, y_pred)
    ace = (apcer + bpcer) / 2.0
    return ace


def calculate_auc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def calculate_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }


def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total if total > 0 else 0.0


def calculate_precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def calculate_recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def calculate_f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def calculate_all_metrics(y_true, y_pred, y_scores=None):
    metrics = {
        'APCER': calculate_apcer(y_true, y_pred),
        'BPCER': calculate_bpcer(y_true, y_pred),
        'ACE': calculate_ace(y_true, y_pred),
        'Accuracy': calculate_accuracy(y_true, y_pred),
        'Precision': calculate_precision(y_true, y_pred),
        'Recall': calculate_recall(y_true, y_pred),
        'F1_Score': calculate_f1_score(y_true, y_pred),
    }
    
    if y_scores is not None:
        metrics['AUC'] = calculate_auc(y_true, y_scores)
    
    cm = calculate_confusion_matrix(y_true, y_pred)
    metrics.update(cm)
    
    return metrics
