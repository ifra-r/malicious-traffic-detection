import pandas as pd

def explore_dataset(file_path, max_unique_display=10):
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    print("\n===== Dataset Shape =====")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n===== Column Names =====")
    print(df.columns.tolist())
    
    print("\n===== Data Types =====")
    print(df.dtypes)
    
    print("\n===== Missing Values per Column =====")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
    
    print("\n===== Unique Values Count per Column =====")
    for col in df.columns:
        unique_count = df[col].nunique(dropna=False)
        print(f"{col}: {unique_count} unique values")
        
        # If categorical or low unique count, show some unique values
        if unique_count <= max_unique_display or df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()[:max_unique_display]
            print(f"  Sample unique values: {unique_vals}")
            
    print("\n===== Sample Data =====")
    print(df.head(5))

if __name__ == "__main__":
    # Change the path to your CSV file
    csv_path = "/home/kay/Documents/Workspace-S25/AI/Web Attacks/Web-attack-detection/data/csic_raw.csv"
    explore_dataset(csv_path)
