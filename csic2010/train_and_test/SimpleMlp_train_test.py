import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

# Add parent folder to sys.path
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(path_to_add)
from models.mlp.SimpleMLP import SimpleMLP  # Import your model


def train(model, X_train, y_train, epochs=50, batch_size=32, lr=0.01):
    n_samples = X_train.shape[0]
    for epoch in range(epochs):
        permutation = np.random.permutation(n_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        epoch_loss = 0
        for i in range(0, n_samples, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            y_pred = model.forward(X_batch)
            loss = model.compute_loss(y_batch, y_pred)
            epoch_loss += loss * len(X_batch)
            dW1, db1, dW2, db2 = model.backward(X_batch, y_batch, y_pred)
            model.update_params(dW1, db1, dW2, db2, lr)
        epoch_loss /= n_samples
        preds = model.predict(X_train)
        accuracy = np.mean(preds == y_train) * 100
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

def evaluate(model, X, y, threshold=0.5):
    y_pred_probs = model.forward(X)
    loss = model.compute_loss(y, y_pred_probs)
    y_pred = (y_pred_probs >= threshold).astype(int)
    accuracy = np.mean(y_pred == y) * 100
    print(f"Evaluation Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

def main():
    data = pd.read_csv('csic2010/data/12_final_dataset.csv')  # Adjust path if needed

    # Assuming first column is label and rest are features
    X = data.iloc[:, 1:].values.astype(np.float32)
    y = data.iloc[:, 0].values.reshape(-1, 1).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SimpleMLP(input_dim=X.shape[1], hidden_dim=10)

    print("Training...")
    train(model, X_train, y_train, epochs=50, batch_size=32, lr=0.01)

    print("\nTesting...")
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    print("simple mlp model on csic2010")
    main()
