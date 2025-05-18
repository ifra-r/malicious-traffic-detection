import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
import sys

# Add parent folder to sys.path
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(path_to_add)
from models.logistic_regression import LogisticRegressionModel  # ← Use this now 
from utils.data_loader import DataLoader
from utils.loss import weighted_binary_cross_entropy

print("linear regression model testing and training on csic2010\n")

# Load dataset
data = pd.read_csv("csic2010/data/12_final_dataset.csv")

X = data.iloc[:, 1:].values        # all columns except the first
y = data.iloc[:, 0].values.reshape(-1, 1)  # the first column is the label

# Split into train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Check class distribution in training set
unique, counts = np.unique(y_train, return_counts=True)
print("Class distribution in training set:", dict(zip(unique.flatten(), counts)))


# Hyperparameters
batch_size = 32
epochs = 50
# learning_rate = 0.01

learning_rate = 0.01

# DataLoaders
train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

# Initialize model
model = LogisticRegressionModel(input_dim=X.shape[1])

# Make sure results folder exists
# os.makedirs("results", exist_ok=True)

# Train and log
with open("csic2010/results/linear_train_logs.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    total = len(y_train)

    weight_neg = total / (2 * neg)
    weight_pos = total / (2 * pos)


    for epoch in range(epochs):
        train_loss = 0
        batches = 0

        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = model.forward(X_batch)

            # Compute loss (binary cross entropy)
            epsilon = 1e-8

            # loss = -np.mean(y_batch * np.log(y_pred + epsilon) + (1 - y_batch) * np.log(1 - y_pred + epsilon))
            loss = weighted_binary_cross_entropy(y_batch, y_pred, weight_pos, weight_neg) # loss = weighted_binary_cross_entropy(y_batch, y_pred, weight_pos=10.0, weight_neg=1.0)
            
            train_loss += loss
            batches += 1

            # Backward + update
            dw, db = model.backward(X_batch, y_batch, y_pred)
            model.update_params(dw, db, learning_rate)

        train_loss /= batches

        # Validation evaluation
        val_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in val_loader:
            y_pred = model.forward(X_batch)
            # loss = -np.mean(y_batch * np.log(y_pred + epsilon) + (1 - y_batch) * np.log(1 - y_pred + epsilon))
            loss = weighted_binary_cross_entropy(y_batch, y_pred, weight_pos, weight_neg) # loss = weighted_binary_cross_entropy(y_batch, y_pred, weight_pos=10.0, weight_neg=1.0)
            val_loss += loss * len(y_batch)

            # predictions = (y_pred >= 0.5).astype(int)
            predictions = (y_pred >= 0.7).astype(int)
            correct += np.sum(predictions == y_batch)
            total += len(y_batch)



        val_loss /= total
        val_accuracy = correct / total * 100

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2f}%")
        writer.writerow([epoch+1, train_loss, val_loss, val_accuracy])

# Final test accuracy
correct, total = 0, 0
for X_batch, y_batch in test_loader:
    predictions = model.predict(X_batch)
    correct += np.sum(predictions == y_batch)
    total += len(y_batch)

test_accuracy = correct / total * 100
print(f"\n✅ Test Accuracy: {test_accuracy:.2f}%")


# Save trained model
import pickle
with open("csic2010/results/linear_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved to linear_model.pkl")  # noqa: T201

