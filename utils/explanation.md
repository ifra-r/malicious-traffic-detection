🔧 Step 2: Utility Functions — Explained

- for reusability

Activations (used inside the model for non-linear transformations)
Loss (to quantify how wrong the model's prediction is)
Metrics (to evaluate how well the model performs)
 
SUMMARY
File	         |  Purpose
activations.py   |  Functions to introduce non-linearity and calculate their gradients
loss.py	         |  Compute how wrong predictions are + derivatives for training
metrics.py	     |  Evaluate model's actual performance on the task

Once these files are added, they can be imported cleanly into both the model and the training script.



================================================================================
details:

📁 1. utils/activations.py
✅ Why?
Activation functions introduce non-linearity into the model. Without them, no matter how many layers you have, your model would behave like a simple linear model.

🔧 What it includes:
sigmoid(z) → Used in the output layer for binary classification.
sigmoid_derivative(z) → Needed for backpropagation to calculate gradients.
relu(z) → Often used in hidden layers for fast convergence.
relu_derivative(z) → Used to compute gradients during backprop.


📁 2. utils/loss.py
✅ Why?
The loss function measures how far your predictions are from the true labels. This is what you try to minimize during training.

🔧 What it includes:
binary_cross_entropy(y_true, y_pred) → Standard loss for binary classification.
binary_cross_entropy_derivative(y_true, y_pred) → Needed to update weights during training (backprop).


📁 3. utils/metrics.py
✅ Why?
Loss tells you how wrong the model is during training, but metrics tell you how good the model is at actually doing its job (e.g., predicting attacks correctly).

🔧 What it includes:
accuracy → Percent of correct predictions

(Optional) precision and recall → Useful for imbalanced datasets like attack detection
