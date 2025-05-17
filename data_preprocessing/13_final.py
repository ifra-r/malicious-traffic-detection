import pandas as pd

# Load your data
df = pd.read_csv('../data/11_model_ready.csv')  # Replace with your actual file path

# 1. Drop boolean columns with too much imbalance (e.g., > 99% same value)
def drop_imbalanced_bool_cols(df, threshold=0.99):
    bool_cols = df.select_dtypes(include='bool').columns
    cols_to_drop = []
    for col in bool_cols:
        top_freq = df[col].value_counts(normalize=True).max()
        if top_freq >= threshold:
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop)

# Apply drop
df = drop_imbalanced_bool_cols(df, threshold=0.99)

# 2. Convert remaining boolean columns to integers (True → 1, False → 0)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# 3. Optionally, save the cleaned dataset
df.to_csv('../data/12_final_dataset.csv', index=False)

print(f"Final shape: {df.shape}")
print("Boolean columns encoded and imbalanced ones removed.")



# $python3 12_wierd.py 
# (49336, 11)
# int64    11
# Name: count, dtype: int64
#    classification  ...  url_ext_none
# 0               1  ...             0
# 1               1  ...             0
# 2               1  ...             0
# 3               0  ...             0
# 4               0  ...             0

# [5 rows x 11 columns]
# Columns with only one unique value: []
# Data shape: (49336, 11)

# Value counts for column 'classification':
# classification
# 1    24668
# 0    24668
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'num_params':
# num_params
# 0     34190
# 1      4851
# 5      4143
# 13     4118
# 3      2034
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'Method_GET':
# Method_GET
# 1    34269
# 0    15067
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'Method_POST':
# Method_POST
# 0    34269
# 1    15067
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'connection_Connection: close':
# connection_Connection: close
# 0    34269
# 1    15067
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'connection_close':
# connection_close
# 1    34269
# 0    15067
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_css':
# url_ext_css
# 0    48659
# 1      677
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_gif':
# url_ext_gif
# 0    45447
# 1     3889
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_jpg':
# url_ext_jpg
# 0    46589
# 1     2747
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_jsp':
# url_ext_jsp
# 1    37450
# 0    11886
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_none':
# url_ext_none
# 0    48456
# 1      880
# Name: count, dtype: int64
# ----------------------------------------
# (myenv)