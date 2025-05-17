# Day 2 - Model Implementation & Training (Person A)

## Overview
This document summarizes the work completed for Day 2 on the Web Attack Detection project using the CSIC-2010 dataset. The focus was on implementing the machine learning model architecture and preparing for training.

---

## Tasks Completed

### 1. Model Implementation
- Designed and implemented a **Logistic Regression model** from scratch using NumPy.
- Included the following components in the model:
  - Initialization of parameters (weights and bias).
  - Forward pass calculation (sigmoid activation).
  - Loss computation using binary cross-entropy.
  - Backpropagation to compute gradients.
  - Parameter update method implementing stochastic gradient descent (SGD).

---

## Pending Tasks
- Develop the **training loop script** to:
  - Load data in batches using a dataloader (to be provided by Person B).
  - Execute forward pass, loss calculation, backpropagation, and parameter updates iteratively.
  - Track and log training metrics such as accuracy and loss per epoch.

- Integrate with Person B’s data pipeline for proper batching, shuffling, and train/validation/test splitting.

---

## Notes
- Coordination with Person B is ongoing to obtain the dataloader.

---
