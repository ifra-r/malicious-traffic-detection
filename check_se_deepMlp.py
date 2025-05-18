import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

        # Optional: print mean gradients for debugging
        # print("mean |dW1|", np.mean(np.abs(dW1)), "|dW2|", np.mean(np.abs(dW2)))

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


def train(model, X_train, y_train, X_val=None, y_val=None, epochs=50, lr=0.01):
    for epoch in range(epochs):
        # Forward + Backprop
        y_pred = model.forward(X_train)
        loss = model.compute_loss(y_train, y_pred)
        model.backward(X_train, y_train, learning_rate=lr)
        train_acc = accuracy(y_train, y_pred)

        # Optional: Validation
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

def evaluate(model, X_test, y_test):
    y_pred = model.forward(X_test)
    test_loss = model.compute_loss(y_test, y_pred)
    test_acc = accuracy(y_test, y_pred)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return test_loss, test_acc

print("=========== DEEP mlp ===========")

# Load data
data = pd.read_csv('data/12_final_dataset.csv')

X = data.iloc[:, :-1].values.astype(np.float32)   # all columns except last one as features
y = data.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)  # last column as label

# Split into train (64%), validation (16%), and test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 0.2 of 0.8 = 0.16

# Create model
model = MLPModel(input_dim=X_train.shape[1], hidden_dim=64, output_dim=1)

# Train with validation
train(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.01)

# Final evaluation on test data
evaluate(model, X_test, y_test)