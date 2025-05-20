# Simple MLP Model (final model) Report

This document details the training and evaluation process for the `SimpleMLP` binary classification model used in web attack detection.

---

## 🧠 Model Architecture

The `SimpleMLP` model is a fully connected feedforward neural network with the following structure:

- **Input Layer:** Size = number of features in the dataset
- **Hidden Layer 1:** Dense layer with `hidden_dim` neurons, using **ReLU** activation
- **Output Layer:** Single neuron with **Sigmoid** activation for binary classification

**Activation Functions:**
- **ReLU**: `f(x) = max(0, x)`
- **Sigmoid**: `f(x) = 1 / (1 + exp(-x))`

---

## ⚙️ Training Process

### Script: `../test/k2_simple_mlp_testing.py`

### Steps:

1. **Model Loading:**
   - A trained `SimpleMLP` model is loaded from disk using `pickle`.

2. **Data Loading:**
   - Test dataset is loaded using a `load_dataset()` function (assumed to load features and labels properly preprocessed and normalized).

3. **Forward Pass:**
   - The model computes output probabilities via a forward pass.

4. **Prediction:**
   - Predictions are converted from probabilities to binary values using a threshold of `0.5`.

5. **Evaluation Metrics:**
   - Loss (Binary Cross-Entropy)
   - Accuracy
   - F1 Score
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1 per class)

---

## 📊 Evaluation Results

### ✅ Metrics:

- **Loss:** 0.2020  
- **Accuracy:** 91.54%  
- **F1 Score:** 0.9144  

### 📄 Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 0.91      | 0.93   | 0.92     | 15041   |
| 1.0   | 0.92      | 0.90   | 0.91     | 15041   |

- **Macro Avg F1:** 0.92  
- **Weighted Avg F1:** 0.92  

### 📊 Confusion Matrix:
[[13931 1110]
[ 1436 13605]]


- **True Negatives (TN):** 13,931  
- **False Positives (FP):** 1,110  
- **False Negatives (FN):** 1,436  
- **True Positives (TP):** 13,605

---

## 📌 Interpretation

- The model generalizes well to unseen data with strong precision and recall for both classes.
- Balanced performance: No heavy bias toward any class.
- Low false positive and false negative rates indicate that the model handles both attack and normal traffic effectively.


---

## 📁 File Reference

- **Model:** `SimpleMLP` (saved with `pickle`)
- **Testing Script:** `../test/k2_simple_mlp_testing.py`
- **Output:** Evaluation metrics printed in terminal

------

## 💡 Recommendations

- Use this model as a strong baseline.
- For further improvement:
  - Try regularization or dropout if overfitting appears in future experiments.
  - Experiment with additional hidden layers for more expressive capacity.
  - Tune learning rate and batch size for optimized training speed and convergence.


