# test/test_model.py

import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.metrics import confusion_matrix, classification_report

# Add parent to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import DataLoader

# Load the trained model
with open("../train/results/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
data = pd.read_csv("../data/12_final_dataset.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data again (same way as training)
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Test loader
test_loader = DataLoader(X_test, y_test, batch_size=32, shuffle=False)

# Evaluate
test_loss, test_acc = model.evaluate(test_loader)
print(f"\n✅ Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

# Predict
all_preds = []
all_labels = []

for X_batch, y_batch in test_loader:
    preds = model.predict(X_batch)
    all_preds.extend(preds.flatten())
    all_labels.extend(y_batch.flatten())

# Evaluation
cm = confusion_matrix(all_labels, all_preds)
print("\n📊 Confusion Matrix:\n", cm)

report = classification_report(all_labels, all_preds)
print("\n📄 Classification Report:\n", report)
