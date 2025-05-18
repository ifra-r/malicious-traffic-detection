import pandas as pd
from urllib.parse import urlparse

def preprocess_url(df):
    df['url_path'] = df['URL'].apply(lambda x: urlparse(x).path if pd.notna(x) else '')
    df['url_query'] = df['URL'].apply(lambda x: urlparse(x).query if pd.notna(x) else '')
    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/5_clean_content.csv")  # Load from step 2
    df = preprocess_url(df)
    df.to_csv("../data/6_reprocess_url.csv", index=False)
    print(df[['URL', 'url_path', 'url_query']].head())
