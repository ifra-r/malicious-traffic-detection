import pandas as pd
import os

def load_and_clean_dataset(file_path):
    print(f"\n=== Loading Dataset from: {file_path} ===")

    if not os.path.exists(file_path):
        print("❌ Error: File does not exist.")
        return

    df = pd.read_csv(file_path)

    print("\n=== Initial Dataset Shape ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Step 1: Drop Unnamed columns
    unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    if unnamed_cols:
        print(f"\nDropping Unnamed Columns: {unnamed_cols}")
        df.drop(columns=unnamed_cols, inplace=True)
    else:
        print("\nNo unnamed columns to drop.")

    # Step 2: Drop columns with only 1 unique value
    single_unique_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if single_unique_cols:
        print(f"\nDropping Columns with Only 1 Unique Value: {single_unique_cols}")
        df.drop(columns=single_unique_cols, inplace=True)
    else:
        print("\nNo constant columns to drop.")

    # Step 3: Drop 'cookie' column if it's almost all unique
    if 'cookie' in df.columns:
        unique_cookies = df['cookie'].nunique(dropna=False)
        total_rows = len(df)
        if unique_cookies >= total_rows * 0.99:  # 99%+ uniqueness
            print(f"\nDropping 'cookie' column due to high uniqueness:")
            print(f"  - Total rows: {total_rows}")
            print(f"  - Unique 'cookie' values: {unique_cookies}")
            df.drop(columns=['cookie'], inplace=True)
        else:
            print(f"\nKeeping 'cookie' column (not highly unique).")

    # Final shape
    print("\n=== Cleaned Dataset Shape ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    return df


if __name__ == "__main__":
    csv_path = "data/csic_raw.csv"  
    cleaned_df = load_and_clean_dataset(csv_path)
    cleaned_df.to_csv('data/1_removed_cols.csv', index=False)

