import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_col='classification'):
    train_val, test = train_test_split(df, test_size=0.2, stratify=df[target_col], random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val[target_col], random_state=42)  # 0.25 * 0.8 = 0.2

    train.to_csv("../data/8_train.csv", index=False)
    val.to_csv("../data/9_val.csv", index=False)
    test.to_csv("../data/10_test.csv", index=False)
    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

if __name__ == "__main__":
    df = pd.read_csv("../data/7_encoded.csv")  # Load from step 5
    split_data(df)



# on/data_preprocessing]
# └──╼ $python3 10_split_data.py 
# Train size: 29601, Val size: 9867, Test size: 9868
# (myenv) ┌─[kay@parrot]─[~/Documents/Workspace-S25/AI/Web Attacks/Web-attack-detection/data_preproc