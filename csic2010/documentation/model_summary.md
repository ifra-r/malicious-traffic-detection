# short summary on models and accuracy improvement

We’ve tried multiple models on the dataset: 

MLP built from scratch: ~75% accuracy

Linear model (from scratch): ~73% accuracy

Random Forest (using scikit-learn): ~75% accuracy

Despite tuning the MLP (e.g., epochs, learning rate), we are consistently hitting a ~75% ceiling and can't seem to improve further. This makes us unsure whether the bottleneck is the data quality/feature signal or limitations in our model architecture.

tried training modle on rf to figure out whether problem is dataset or the model, and we've concluded that the problem is in our data not the model architechture

# notes: in progress