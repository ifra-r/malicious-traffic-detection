import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

# Add parent folder to sys.path
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(path_to_add)
from models.mlp.DeepMLP import MLPModel  # Import the MLPModel from your module

def accuracy(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    return np.mean(y_pred_labels == y_true) * 100

def train(model, X_train, y_train, X_val=None, y_val=None, epochs=50, lr=0.01):
    for epoch in range(epochs):
        y_pred = model.forward(X_train)
        loss = model.compute_loss(y_train, y_pred)
        model.backward(X_train, y_train, learning_rate=lr)
        train_acc = accuracy(y_train, y_pred)

        if X_val is not None and y_val is not None:
            y_val_pred = model.forward(X_val)
            val_loss = model.compute_loss(y_val, y_val_pred)
            val_acc = accuracy(y_val, y_val_pred)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}, Acc: {train_acc:.2f}%")

def evaluate(model, X_test, y_test):
    y_pred = model.forward(X_test)
    test_loss = model.compute_loss(y_test, y_pred)
    test_acc = accuracy(y_test, y_pred)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return test_loss, test_acc

def main():
    # Load CSV data
    data = pd.read_csv('csic2010/data/12_final_dataset.csv')

    # First column = label, rest = features
    X = data.iloc[:, 1:].values.astype(np.float32)
    y = data.iloc[:, 0].values.reshape(-1, 1).astype(np.float32)

    # Split into train (64%), validation (16%), test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    # Initialize model
    model = MLPModel(input_dim=X.shape[1], hidden_dim=64, output_dim=1)

    print("Training...")
    train(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.01)

    print("\nTesting...")
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    print("deep mlp model on csic2010")
    main()
