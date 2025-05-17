import numpy as np

def binary_cross_entropy(y_true, y_pred):
    # Avoid log(0) by clipping predictions
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

def binary_cross_entropy_derivative(y_true, y_pred):
    # Gradient of the loss w.r.t. the prediction
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
