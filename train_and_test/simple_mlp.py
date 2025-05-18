import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

    print("=========== simple mlp  train n test ===========\n")

    # data = pd.read_csv('data/cleaned/4_remove_features.csv')  # Adjust path if needed
    data = pd.read_csv('data/cleaned/se.csv')  # Adjust path if needed

    X = data.iloc[:, :-1].values.astype(np.float32)   # all columns except last one as features
    y = data.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)  # last column as label


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SimpleMLP(input_dim=X.shape[1], hidden_dim=10)

    print("Training...")
    train(model, X_train, y_train, epochs=50, batch_size=32, lr=0.01)

    print("\nTesting...")
    evaluate(model, X_test, y_test)

    # Create results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)

    import pickle
    # Save model
    with open("results/simple_mlp.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved to results/simple_mlp.pkl")

if __name__ == "__main__":
    main()
