You already:

Tried dynamic loss weighting

Tried threshold tuning

Have a balanced train/val split

Still get 16% test accuracy due to massive false positives

This means your model is learning to prioritize class 1 because it sees it more confidently separable — but at the cost of precision for class 0.