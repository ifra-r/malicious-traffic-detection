import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import precision, recall

class MLPModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)  # He init
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def leaky_relu(self, z):
        return np.where(z > 0, z, 0.01 * z)

    def leaky_relu_derivative(self, z):
        return np.where(z > 0, 1, 0.01)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.leaky_relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y_true, y_pred):
        eps = 1e-10  # to avoid log(0)
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    def backward(self, X, y_true, learning_rate):
        m = X.shape[0]
        dz2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.W2.T) * self.leaky_relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


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


def accuracy(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    return np.mean(y_pred_labels == y_true) * 100

from sklearn.metrics import confusion_matrix


def evaluate(model, X_test, y_test):
    y_pred = model.forward(X_test)
    test_loss = model.compute_loss(y_test, y_pred)
    test_acc = accuracy(y_test, y_pred)

    # Calculate precision, recall, f1
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

    # Optionally, print confusion matrix as before (you have code for that)
    y_pred_labels = (y_pred >= 0.5).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_labels)
    print("Confusion Matrix:")
    print(cm)

    return test_loss, test_acc



def main():
    print("=========== DEEP mlp ===========")

    # Load data
    data = pd.read_csv('../data/cleaned/4_remove_features.csv')  

    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    # Create and train model
    model = MLPModel(input_dim=X_train.shape[1], hidden_dim=64, output_dim=1)
    train(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.01)

    # Evaluate on test data
    evaluate(model, X_test, y_test)

    # Save model
    os.makedirs("results", exist_ok=True)
    with open("../results/k_deep_mlp.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to results/deep_mlp.pkl")


if __name__ == "__main__":
    main()
