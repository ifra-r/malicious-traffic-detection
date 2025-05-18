import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df, columns):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/6_reprocess_url.csv")
    
    # Clean URL first (if not done earlier)
    df['clean_url'] = df['URL'].str.replace(r'\sHTTP/1\.1$', '', regex=True)
    
    categorical_cols = ['Method', 'host', 'connection']  # correct column names
    
    df = encode_categorical(df, categorical_cols)
    
    df.to_csv("../data/7_encoded.csv", index=False)
    print(df[categorical_cols].head())
