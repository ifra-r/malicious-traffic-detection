# Cyberattack Detection through Traffic Profiling

This repository presents an end-to-end machine learning pipeline to detect cyberattacks in network traffic using the **UNSW-NB15** dataset. The project includes the implementation of **multiple classification models from scratch** using **NumPy**, including:

- Logistic Regression (Baseline)
- Simple MLP (1 Hidden Layer Neural Network)
- Deep MLP (Multi-layer Neural Network)

The objective is to compare and evaluate these models for binary classification: **attack vs normal traffic**.

---

##  Project Highlights

- **From-scratch implementation**: All models are built using only NumPy.
- **Traffic profiling**: Preprocessed data from UNSW-NB15 used to detect attack patterns.
- **Model comparison**: Analyzed performance across linear and non-linear models.
- **Strong performance**: The final model (Simple MLP) achieved **91.54% accuracy** and **F1 score of 0.9144**.

---

##  Models Implemented

| Model               | Type        | Architecture                  | Activation Functions         | Use Case                                  |
|---------------------|-------------|-------------------------------|------------------------------|--------------------------------------------|
| Logistic Regression | Linear      | Input → Sigmoid               | Sigmoid                      | Baseline for linear patterns               |
| Simple MLP          | Neural Net  | Input → ReLU → Sigmoid        | ReLU (hidden), Sigmoid (out) | Captures non-linear traffic patterns       |
| Deep MLP            | Deep NN     | Input → 3×(ReLU) → Sigmoid    | ReLU (hidden), Sigmoid (out) | More expressive but sensitive to overfit   |

---

##  Evaluation Summary

| Metric         | Logistic Regression | Simple MLP        | Deep MLP          |
|----------------|---------------------|-------------------|-------------------|
| **Loss**       | 3.0766              | 0.2020            | 0.5722            |
| **Accuracy**   | 85.15%              | **91.54%**        | 78.96%            |
| **F1 Score**   | 0.84                | **0.9144**        | 0.7650            |
| **Precision**  | 0.86                | 0.92              | 0.8665            |
| **Recall**     | 0.85                | 0.90              | 0.6847            |

---

##  Simple MLP: Final Model

###  Architecture

- **Input Layer**: Feature size of UNSW-NB15 dataset
- **Hidden Layer**: Fully connected, ReLU activation
- **Output Layer**: Single node with Sigmoid activation

###  Final Results

- **Accuracy**: 91.54%
- **F1 Score**: 0.9144
- **Loss**: 0.2020

####  Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 0.91      | 0.93   | 0.92     | 15041   |
| 1.0   | 0.92      | 0.90   | 0.91     | 15041   |

####  Confusion Matrix
[[13931 1110]
[ 1436 13605]]

- **True Negatives**: 13,931
- **True Positives**: 13,605
- **False Positives**: 1,110
- **False Negatives**: 1,436


---

##  Dataset

- **Source**: 
  - [UNSW-NB15 Official Website](https://research.unsw.edu.au/projects/unsw-nb15-dataset)  
  - [Kaggle Mirror (Parquet Format)](https://www.kaggle.com/datasets/dhoogla/unswnb15?select=UNSW_NB15_testing-set.parquet)

- **Preprocessing**:
  - Missing values handling
  - Feature selection
  - Normalization
  - Label encoding (attack = 1, normal = 0)
  - Feature encoding
  - Train-test split

---

##  Training & Testing

Each model implements the following methods:

- `train(X, y, epochs=..., learning_rate=...)`: Performs forward and backward passes using gradient descent.
- `predict(X)`: Outputs binary predictions based on thresholded sigmoid output.

---

##  Future Improvements

- Support multi-class classification for specific attack types. 
- Move to PyTorch for scaling to larger datasets.

---

##  Author

**Ifra Abdul Rauf**  
**Khadijah Farooqi**    

Feel free to connect or contribute!