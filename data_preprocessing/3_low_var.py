# Loads the cleaned dataset.

# Computes variance for each numeric feature.

# Checks correlation with the target (classification).

# Logs variance and correlation.

# Drops features that have both:

# Low variance (default threshold = 0.01), and

# Low correlation with target (default abs(corr) < 0.05).

# Logs final shape after cleanup.

import pandas as pd
import numpy as np

def drop_low_variance_features(file_path, variance_threshold=0.01, corr_threshold=0.05):
    print(f"\n=== Loading Dataset ===")
    df = pd.read_csv(file_path)
    print(f"Initial Shape: {df.shape}")

    target_col = 'classification'
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found in dataset.")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\n=== Calculating Variance and Correlation ===")
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

    dropped_features = []

    for col in numeric_cols:
        var = X[col].var()
        corr = df[col].corr(y)  # Pearson correlation

        print(f"[{col}] Variance: {var:.6f}, Correlation with target: {corr:.6f}")

        if var < variance_threshold and abs(corr) < corr_threshold:
            print(f"--> Dropping '{col}' (Low variance + low correlation)")
            dropped_features.append(col)

    # Drop selected features
    df_cleaned = df.drop(columns=dropped_features)

    print(f"\nDropped {len(dropped_features)} low-importance feature(s): {dropped_features}")
    print(f"Final Shape: {df_cleaned.shape}")

    return df_cleaned

# Example usage
if __name__ == "__main__":
    cleaned_file_path = "data/1_removed_cols.csv"
    df_result = drop_low_variance_features(cleaned_file_path)

