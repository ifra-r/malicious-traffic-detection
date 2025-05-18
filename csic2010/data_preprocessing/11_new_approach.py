import pandas as pd
from urllib.parse import urlparse

def clean_http_from_url(s):
    return s.replace(' HTTP/1.1', '').strip() if isinstance(s, str) else s

def preprocess_url(df):
    df['URL'] = df['URL'].apply(clean_http_from_url)
    df['url_path'] = df['URL'].apply(lambda x: urlparse(x).path if pd.notna(x) else '')
    df['url_query'] = df['URL'].apply(lambda x: urlparse(x).query if pd.notna(x) else '')

    # Extract URL extension (e.g., jsp, do, etc.)
    df['url_ext'] = df['url_path'].apply(lambda x: x.strip('/').split('/')[-1].split('.')[-1] if '.' in x else 'none')

    # Number of query parameters
    df['num_params'] = df['url_query'].apply(lambda x: len(x.split('&')) if pd.notna(x) and x else 0)

    return df

def clean_and_encode(df):
    df.drop(['URL', 'Accept', 'content_tokens', 'url_path', 'url_query'], axis=1, inplace=True)

    if df['host'].nunique() == 1:
        df.drop('host', axis=1, inplace=True)

    df = pd.get_dummies(df, columns=['Method', 'connection', 'url_ext'])

    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/6_reprocess_url.csv")
    df = preprocess_url(df)
    df = clean_and_encode(df)
    df.to_csv("../data/11_model_ready.csv", index=False)
    print(df.shape)
    print(df.head())
