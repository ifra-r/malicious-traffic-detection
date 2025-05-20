# Cyber Attack Detection – Model Design and Training Documentation

## Overview

This project implements and trains multiple machine learning models from scratch using NumPy to detect web based on the unsw_nb15 dataset. The classification task is binary (attack vs normal). The models built include:

- **Logistic Regression** (baseline linear model)
- **Simple MLP** (1 hidden layer)
- **Deep MLP** (multi-layer with more expressive power)

I created **multiple models** instead of just one to:

- **Compare performance** between simple linear and more complex neural models.
- **Test model expressiveness** for capturing non-linear patterns in attack behaviors.
- **Ensure robustness**: different models may excel under different data distributions.
- **Understand trade-offs**: between accuracy, complexity, training time, and generalization.

---

## Model Training Pipeline

All models follow a similar training workflow with these major steps:

### 1. **Model Initialization**

- Initialize weights and biases.
- Logistic regression uses zeros.
- MLP models use small random values (He initialization for deep networks) to prevent vanishing gradients.

### 2. **Forward Propagation**

- Compute predictions based on current weights.
- **Activation functions**:
  - Logistic Regression: Sigmoid
  - Simple MLP: Tanh → Sigmoid
  - Deep MLP: Leaky ReLU → Sigmoid

### 3. **Loss Computation**

- Loss function used: **Binary Cross-Entropy**
- Helps measure how well predicted probabilities match true labels.
- Use a small `epsilon` to avoid `log(0)` issues.

### 4. **Backward Propagation**

- Compute gradients of the loss with respect to each weight/bias.
- Use **backpropagation** with chain rule to propagate errors.

### 5. **Parameter Update**

- Use **Stochastic Gradient Descent (SGD)**:
  - Update weights using the gradient and learning rate.
  - Repeat over mini-batches for stable convergence.

### 6. **Training Loop**

- Loop over several epochs:
  - Shuffle data and batch it.
  - Perform forward → loss → backward → update.
- Track accuracy and loss per epoch for monitoring.

---

## Model Summary

| Model               | Type         | Architecture                   | Activation Functions        | Use Case                                                    |
|---------------------|--------------|--------------------------------|-------------------------------------------------------------------------------------------|
| Logistic Regression | Linear       | Input → Sigmoid                | Sigmoid                      | Baseline for linearly separable patterns                   |
| Simple MLP          | Neural Net   | Input → Tanh → Sigmoid         | Tanh (hidden), Sigmoid (out) | Captures simple non-linear relationships                   |
| Deep MLP            | Deep NN      | Input → Leaky ReLU → Sigmoid   | Leaky ReLU, Sigmoid          | Models complex patterns; more expressive but more training |

---

## Code Organization Recommendation

