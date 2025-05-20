# Model Training and Testing Documentation

This document explains the training and testing steps for the three NumPy-based models implemented in the `models/` directory: **Logistic Regression**, **Simple MLP**, and **Deep MLP**. All models are trained and evaluated using the same data pipeline.

---

## 1. Logistic Regression

### Training

- **File**: `models/LogisticRegression.py`
- **Class**: `LogisticRegression`
- **Function**: `train(X, y, epochs=100, learning_rate=0.01, verbose=False)`

**Steps:**
1. Initializes weights `self.weights` and bias `self.bias` to zeros.
2. For each epoch:
   - Computes the logits: `np.dot(X, self.weights) + self.bias`.
   - Applies the sigmoid activation to get probabilities.
   - Computes the binary cross-entropy loss.
   - Computes gradients of loss w.r.t weights and bias.
   - Updates weights and bias using gradient descent.

### Testing

- **Function**: `predict(X)`
  - Computes the sigmoid of `np.dot(X, self.weights) + self.bias`.
  - Thresholds the output at 0.5 to return binary predictions (0 or 1).

---

## 2. Simple MLP

### Training

- **File**: `mlp/SimpleMLP.py`
- **Class**: `SimpleMLP`
- **Function**: `train(X, y, epochs=100, learning_rate=0.01, verbose=False)`

**Architecture:**
- One hidden layer (ReLU) → Output layer (sigmoid)

**Steps:**
1. Initializes:
   - Hidden layer weights and biases: `self.W1`, `self.b1`.
   - Output layer weights and bias: `self.W2`, `self.b2`.
2. For each epoch:
   - Forward pass:
     - Hidden layer output: `Z1 = X @ W1 + b1`, `A1 = ReLU(Z1)`.
     - Output layer: `Z2 = A1 @ W2 + b2`, `A2 = sigmoid(Z2)`.
   - Loss computation: binary cross-entropy.
   - Backward pass:
     - Gradients are computed using chain rule for both layers.
   - Parameter updates via gradient descent.

### Testing

- **Function**: `predict(X)`
  - Performs forward pass and thresholds the final sigmoid output at 0.5.

---

## 3. Deep MLP

### Training

- **File**: `mlp/DeepMLP.py`
- **Class**: `DeepMLP`
- **Function**: `train(X, y, epochs=100, learning_rate=0.01, verbose=False)`

**Architecture:**
- Three hidden layers (ReLU) → Output layer (sigmoid)

**Steps:**
1. Initializes:
   - Three sets of weights and biases: `W1`, `W2`, `W3`, and `b1`, `b2`, `b3`.
   - Output layer: `W_out`, `b_out`.
2. For each epoch:
   - Forward pass through all layers:
     - Hidden layers use ReLU activation.
     - Output layer uses sigmoid.
   - Computes binary cross-entropy loss.
   - Backward pass:
     - Gradients are computed layer by layer.
   - Parameters are updated using gradient descent.

### Testing

- **Function**: `predict(X)`
  - Performs forward pass through all layers.
  - Outputs binary predictions after applying threshold to sigmoid output.

---

## General Notes

- All models use NumPy only (no external ML libraries).
- Training is done using gradient descent with a manually computed binary cross-entropy loss.
- The `verbose=True` flag prints loss at each epoch.
- All models implement `train()` and `predict()` functions for clarity and consistency.
