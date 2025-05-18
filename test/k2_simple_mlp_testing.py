import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import pickle
import os
import sys

# Add parent directory to path to import the model class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_and_test.k_simple_mlp import SimpleMLP 

def evaluate(model, X, y, threshold=0.5):
    print("\n=========== Evaluation ===========")
    y_pred_probs = model.forward(X)
    loss = model.compute_loss(y, y_pred_probs)
    y_pred = (y_pred_probs >= threshold).astype(int)

    acc = accuracy_score(y, y_pred) * 100
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print("\n📊 Confusion Matrix:")
    print(cm)
    print("\n📄 Classification Report:")
    print(report)

def main():
    print("=========== Testing Trained MLP Model ===========\n")

    # Load test data
    data = pd.read_csv('../data/cleaned/se.csv')  # Same data source used in training
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)

    # Load trained model
    model_path = "../results/k_simple_mlp.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Evaluate on whole dataset or create your own test split
    evaluate(model, X, y)

if __name__ == "__main__":
    main()
