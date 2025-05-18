# Drops columns with too many missing values (default threshold: >50%).

# Drops rows with any remaining missing values.

# Logs detailed information before and after.


import pandas as pd

def handle_missing_values(df, col_threshold=0.5):
    print("\n===== Handling Missing Values =====")

    # 1. Log total rows and initial missing values
    total_rows = df.shape[0]
    missing_per_col = df.isna().sum()
    print("\nInitial Missing Values per Column:")
    print(missing_per_col[missing_per_col > 0])

    # 2. Drop columns with > threshold missing values
    high_missing_cols = missing_per_col[missing_per_col / total_rows > col_threshold].index.tolist()
    if high_missing_cols:
        df.drop(columns=high_missing_cols, inplace=True)
        print(f"\nDropped columns with >{col_threshold*100:.0f}% missing values: {high_missing_cols}")
    else:
        print("\nNo columns dropped for high missing values.")

    # 3. Drop rows with any remaining missing values
    remaining_missing = df.isna().sum().sum()
    if remaining_missing > 0:
        before_rows = df.shape[0]
        df.dropna(inplace=True)
        after_rows = df.shape[0]
        print(f"\nDropped {before_rows - after_rows} rows due to remaining missing values.")
    else:
        print("\nNo rows dropped; no remaining missing values.")

    # 4. Final shape
    print(f"\nFinal Dataset Shape: Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    return df

df = pd.read_csv('data/1_removed_cols.csv')
cleaned_df = handle_missing_values(df)
cleaned_df.to_csv('data/2_missing_vals.csv', index=False)
