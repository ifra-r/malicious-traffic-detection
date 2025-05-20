# Model Comparison Report

This report presents the evaluation results of three models trained for binary classification on web attack detection:

- **Logistic Regression**
- **Simple MLP**
- **Deep MLP**

---

## 🔍 Evaluation Summary

| Metric         | Logistic Regression | Simple MLP        | Deep MLP          |
|----------------|---------------------|-------------------|-------------------|
| **Loss**       | 3.0766              | 0.2020            | 0.5722            |
| **Accuracy**   | 85.15%              | 91.54%            | 78.96%            |
| **F1 Score**   | 0.84                | 0.9144            | 0.7650            |
| **Precision**  | 0.86 (avg)          | 0.92 (avg)        | 0.8665            |
| **Recall**     | 0.85 (avg)          | 0.92 (avg)        | 0.6847            |

---
# model evaluation

## 🧠 Simple MLP

Accuracy: 91.54%
F1 Score: 0.9144
Loss: 0.2020

confusion matrix;
                    [[13931  1110]
                    [ 1436 13605]]


classification reort:
              precision    recall  f1-score   support
         0.0       0.91      0.93      0.92     15041
         1.0       0.92      0.90      0.91     15041
    accuracy                           0.92     30082
   macro avg       0.92      0.92      0.92     30082
weighted avg       0.92      0.92      0.92     30082


##  Logistic Regression

Loss: 3.0766
Accuracy: 85.15%
F1 Score: 0.84

confusion matrix;
                    [[2100  129]
                    [ 541 1743]]


classification reort:
              precision    recall  f1-score   support
           0       0.80      0.94      0.86      2229
           1       0.93      0.76      0.84      2284
    accuracy                           0.85      4513
   macro avg       0.86      0.85      0.85      4513
weighted avg       0.86      0.85      0.85      4513


## Deep MLP

Loss: 0.5722

Accuracy: 78.96%

F1 Score: 0.7650

Precision: 0.8665

Recall: 0.6847

confusion matrix:
                    [[13454  1587]
                    [ 4742 10299]]

