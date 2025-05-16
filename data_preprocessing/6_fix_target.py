import pandas as pd

def fix_target(df):
    df['classification'] = df['classification'].astype(int)
    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/3_balanced_data.csv")  # raw original file
    df = fix_target(df)
    df.to_csv("../data/4_fixed_target.csv", index=False)  # save processed
    print(df[['classification']].head())
