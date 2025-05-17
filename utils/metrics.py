import numpy as np

def accuracy(y_true, y_pred):
    predictions = (y_pred >= 0.5).astype(int)
    return np.mean(predictions == y_true)

# Optional
def precision(y_true, y_pred):
    y_pred_labels = (y_pred >= 0.5).astype(int)
    tp = np.sum((y_pred_labels == 1) & (y_true == 1))
    fp = np.sum((y_pred_labels == 1) & (y_true == 0))
    return tp / (tp + fp + 1e-9)

def recall(y_true, y_pred):
    y_pred_labels = (y_pred >= 0.5).astype(int)
    tp = np.sum((y_pred_labels == 1) & (y_true == 1))
    fn = np.sum((y_pred_labels == 0) & (y_true == 1))
    return tp / (tp + fn + 1e-9)
