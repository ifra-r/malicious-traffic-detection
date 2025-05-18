import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Import custom DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import DataLoader

# Load trained model
model_path = "../results/linear_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load dataset
data = pd.read_csv("../data/cleaned/4_remove_features.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Split data (same as training)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check test set class distribution
unique, counts = np.unique(y_test, return_counts=True)
print("Class distribution in test set:", dict(zip(unique.flatten(), counts)))

# Create DataLoader
test_loader = DataLoader(X_test, y_test, batch_size=32, shuffle=False)

# Evaluate model
test_loss, test_acc = model.evaluate(test_loader)
print(f"\n✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_acc:.2f}%")

# Collect predictions and ground truth
all_preds = []
all_labels = []

for X_batch, y_batch in test_loader:
    preds = model.predict(X_batch)
    all_preds.extend(preds.flatten())
    all_labels.extend(y_batch.flatten())

# Confusion Matrix and Classification Report
cm = confusion_matrix(all_labels, all_preds)
print("\n📊 Confusion Matrix:\n", cm)

report = classification_report(all_labels, all_preds)
print("\n📄 Classification Report:\n", report)
