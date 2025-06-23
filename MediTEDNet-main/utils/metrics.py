import numpy as np
from sklearn import metrics

def metrics_cal_multiclass(all_labels, all_preds, all_probs):
    """
    Compute evaluation metrics for a multi-class classification task.

    Parameters:
    - all_labels: Ground truth labels (list or np.ndarray, shape: [n_samples])
    - all_preds: Predicted labels (list or np.ndarray, shape: [n_samples])
    - all_probs: Predicted probability distributions (np.ndarray, shape: [n_samples, n_classes])

    Returns:
    - A dictionary containing conf_matrix, precision, recall, f1, specificity, auc, and per_class_metrics
    """
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Compute the confusion matrix
    conf_matrix = metrics.confusion_matrix(all_labels, all_preds)

    # Calculate metrics for each class
    report = metrics.classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    per_class_metrics = {}
    for class_label, metrics_data in report.items():
        if class_label.isdigit():
            per_class_metrics[int(class_label)] = {
                'precision': float(metrics_data['precision']),
                'recall': float(metrics_data['recall']),
                'f1_score': float(metrics_data['f1-score']),
                'support': int(metrics_data['support'])
            }

    # Precision
    precision = metrics.precision_score(all_labels, all_preds, zero_division=0, average='weighted')

    # Recall
    recall = metrics.recall_score(all_labels, all_preds, zero_division=0, average='weighted')

    # F1
    f1 = metrics.f1_score(all_labels, all_preds, zero_division=0, average='weighted')

    # Specificity
    confusion_matrix = metrics.confusion_matrix(all_labels, all_preds)
    n_classes = confusion_matrix.shape[0]
    specificity_list = []
    for i in range(n_classes):
        tn = np.sum(confusion_matrix) - np.sum(confusion_matrix[i, :]) - np.sum(confusion_matrix[:, i]) + confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_list.append(float(specificity))
    specificity = np.mean(specificity_list)

    # AUC
    try:
        auc = metrics.roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError as e:
        print(f"AUC calculation failed. Error: {e}")
        auc = 0.0

    return conf_matrix, precision, recall, f1, specificity, auc, per_class_metrics


