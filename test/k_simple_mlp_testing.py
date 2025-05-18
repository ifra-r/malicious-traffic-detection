import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mlp.SimpleMLP import SimpleMLP  # Your model class

# Import your metrics or define here
def precision(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp + 1e-9)

def recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn + 1e-9)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + 1e-9)
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[tn, fp],
                     [fn, tp]])


def load_data():
    data = pd.read_csv('../data/cleaned/se.csv')  # same dataset as train, adjust if needed
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)
    return X, y

def load_model(path="../results/simple_mlp.pkl"):
    import pickle
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def evaluate(model, X, y, threshold=0.5):
    y_pred_probs = model.forward(X)
    loss = model.compute_loss(y, y_pred_probs)
    y_pred = (y_pred_probs >= threshold).astype(int)
    accuracy = np.mean(y_pred == y) * 100
    prec = precision(y, y_pred)
    rec = recall(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"Evaluation Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(f"[[TN: {cm[0, 0]}  FP: {cm[0, 1]}]")
    print(f" [FN: {cm[1, 0]}  TP: {cm[1, 1]}]]")

def main():
    print("Loading test data...")
    X, y = load_data()

    print("Loading saved model...")
    model = load_model()

    print("Evaluating model on full dataset:")
    evaluate(model, X, y)

if __name__ == "__main__":
    main()
