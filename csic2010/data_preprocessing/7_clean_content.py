from urllib.parse import urlparse, parse_qs
import pandas as pd

def tokenize_params_from_url(url):
    if pd.isna(url) or url == '':
        return []
    parsed = urlparse(url)
    query = parsed.query
    if not query:
        return []
    return [param.split('=')[0] for param in query.split('&')]

def clean_content_from_url(df):
    df['content_tokens'] = df['URL'].apply(tokenize_params_from_url)
    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/4_fixed_target.csv")
    df = clean_content_from_url(df)
    df.to_csv("../data/5_clean_content.csv", index=False)
    print(df[['URL', 'content_tokens']].head())
