import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class SimpleMLP:
    def __init__(self, input_dim, hidden_dim=10):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-8
        m = y_true.shape[0]
        loss = - (1/m) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss
    
    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]
        dZ2 = y_pred - y_true
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (1 - np.power(self.A1, 2))
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2, lr):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
    
    def predict(self, X, threshold=0.5):
        probs = self.forward(X)
        return (probs >= threshold).astype(int)

# def train(model, X_train, y_train, epochs=50, batch_size=32, lr=0.01):
#     n_samples = X_train.shape[0]
#     for epoch in range(epochs):
#         permutation = np.random.permutation(n_samples)
#         X_train_shuffled = X_train[permutation]
#         y_train_shuffled = y_train[permutation]
#         epoch_loss = 0
#         for i in range(0, n_samples, batch_size):
#             X_batch = X_train_shuffled[i:i+batch_size]
#             y_batch = y_train_shuffled[i:i+batch_size]
#             y_pred = model.forward(X_batch)
#             loss = model.compute_loss(y_batch, y_pred)
#             epoch_loss += loss * len(X_batch)
#             dW1, db1, dW2, db2 = model.backward(X_batch, y_batch, y_pred)
#             model.update_params(dW1, db1, dW2, db2, lr)
#         epoch_loss /= n_samples
#         preds = model.predict(X_train)
#         accuracy = np.mean(preds == y_train) * 100
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

# def evaluate(model, X, y, threshold=0.5):
#     y_pred_probs = model.forward(X)
#     loss = model.compute_loss(y, y_pred_probs)
#     y_pred = (y_pred_probs >= threshold).astype(int)
#     accuracy = np.mean(y_pred == y) * 100
#     print(f"Evaluation Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

# def main():
#     # Load your CSV file here, adjust path as needed
#     data = pd.read_csv('data/12_final_dataset.csv')
    
#     # Assuming first column is label and rest are features
#     X = data.iloc[:, 1:].values.astype(np.float32)
#     y = data.iloc[:, 0].values.reshape(-1, 1).astype(np.float32)
    
#     # Split into train/test (80/20)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     input_dim = X.shape[1]
#     model = SimpleMLP(input_dim=input_dim, hidden_dim=10)
    
#     print("Training...")
#     train(model, X_train, y_train, epochs=50, batch_size=32, lr=0.01)
    
#     print("\nTesting...")
#     evaluate(model, X_test, y_test)

# if __name__ == "__main__":
#     main()
