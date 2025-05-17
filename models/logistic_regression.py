# Structure:
# __init__: Initialize weights and bias
# forward: Calculate predictions using sigmoid
# backward: Compute gradients (loss wrt weights and bias)
# update_params: Update weights using gradients and learning rate
# predict: Use learned weights to predict outputs
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils.loss import binary_cross_entropy
from utils.metrics import accuracy
from utils.activations import sigmoid

class LogisticRegressionModel:

    def __init__(self, input_dim):      # input dim: num of features in dataset

        # These are the parameters model will learn during training
        self.weights = np.zeros((input_dim, 1))  # shape: (features, 1)  # weight is col vector init to 0s
        self.bias = 0


    def forward(self, X):
        # X shape: (batch_size, input_dim)

        # computes the weighted sum:
        z = np.dot(X, self.weights) + self.bias             # shape: (batch_size, 1)

        return sigmoid(z) # Returns: predictions shape (batch_size, 1)


    def backward(self, X, y_true, y_pred): 
        # Returns gradients for weights and bias 
        m = X.shape[0]  # batch size
        dz = y_pred - y_true       # computes the prediction error for each sample.       # shape: (batch_size, 1)
        dw = np.dot(X.T, dz) / m   # partial derivatives of loss w.r.t. weights.         # shape: (input_dim, 1)
        db = np.sum(dz) / m        # partial derivatives w.r.t bias
        return dw, db

    def update_params(self, dw, db, lr):
        # Update the weights and bias using Stochastic Gradient Descent (SGD).
        self.weights -= lr * dw     # lr: learning rate
        self.bias -= lr * db

    def predict(self, X):
        # Computes probabilities using forward.
        # Converts probabilities to class labels: if ≥ 0.5 → 1, else → 0.
        probs = self.forward(X)
        return (probs >= 0.5).astype(int)

    
    def evaluate(self, data_loader):
        all_preds = []
        all_labels = []

        for X_batch, y_batch in data_loader:
            preds = self.predict(X_batch)
            all_preds.extend(preds.flatten())
            all_labels.extend(y_batch.flatten())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        loss = binary_cross_entropy(all_labels, all_preds)
        acc = accuracy(all_labels, all_preds) * 100
        return loss, acc

    
# In training, for each batch:
# forward → get predictions
# backward → calculate gradients
# update_params → adjust weights
# After training, use predict to evaluate on test data

 
# X	   :    Input features matrix  ===>	(batch_size, num_features)	
# batch:	Subset of samples from data	 