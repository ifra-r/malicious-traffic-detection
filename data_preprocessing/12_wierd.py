import pandas as pd

df = pd.read_csv("../data/11_model_ready.csv")
print(df.shape)          # Should print (rows, 46)
print(df.dtypes.value_counts())  # Should show how many bool, int, etc.
print(df.head())

# Check for any weird or empty columns
empty_cols = [col for col in df.columns if df[col].nunique() == 1]
print("Columns with only one unique value:", empty_cols)
print(f"Data shape: {df.shape}\n")

for col in df.columns:
    print(f"Value counts for column '{col}':")
    print(df[col].value_counts(dropna=False))  # dropna=False to count NaNs if any
    print("-" * 40)


#     (49336, 46)
# bool     44
# int64     2
# Name: count, dtype: int64
#    classification  num_params  Method_GET  Method_POST  ...  url_ext_properties  url_ext_txt  url_ext_xml  url_ext_~
# 0               1           0       False         True  ...               False        False        False      False
# 1               1           0       False         True  ...               False        False        False      False
# 2               1           0        True        False  ...               False        False        False      False
# 3               0           0        True        False  ...               False        False        False      False
# 4               0           0        True        False  ...               False        False        False      False

# [5 rows x 46 columns]
# Columns with only one unique value: []
# Data shape: (49336, 46)

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
# True     34269
# False    15067
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'Method_POST':
# Method_POST
# False    34269
# True     15067
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'connection_Connection: close':
# connection_Connection: close
# False    34269
# True     15067
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'connection_close':
# connection_close
# True     34269
# False    15067
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_4861362529278789730':
# url_ext_4861362529278789730
# False    49210
# True       126
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_4861362529278789730~':
# url_ext_4861362529278789730~
# False    49320
# True        16
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_BAK':
# url_ext_BAK
# False    49119
# True       217
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_Bak':
# url_ext_Bak
# False    49117
# True       219
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_INC':
# url_ext_INC
# False    49140
# True       196
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_Inc':
# url_ext_Inc
# False    49110
# True       226
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_OLD':
# url_ext_OLD
# False    49074
# True       262
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_Old':
# url_ext_Old
# False    49068
# True       268
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_asf-logo-wide':
# url_ext_asf-logo-wide
# False    49219
# True       117
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_asf-logo-wide~':
# url_ext_asf-logo-wide~
# False    49327
# True         9
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_asp':
# url_ext_asp
# False    49161
# True       175
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_bak':
# url_ext_bak
# False    49119
# True       217
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_box':
# url_ext_box
# False    49330
# True         6
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_btr':
# url_ext_btr
# False    49319
# True        17
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_cfm':
# url_ext_cfm
# False    49200
# True       136
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_cnf':
# url_ext_cnf
# False    49282
# True        54
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_css':
# url_ext_css
# False    48659
# True       677
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_css~':
# url_ext_css~
# False    49332
# True         4
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_dll':
# url_ext_dll
# False    49302
# True        34
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_do':
# url_ext_do
# False    49306
# True        30
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_exe':
# url_ext_exe
# False    49296
# True        40
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_gif':
# url_ext_gif
# False    45447
# True      3889
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_gif~':
# url_ext_gif~
# False    49309
# True        27
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_htm':
# url_ext_htm
# False    49195
# True       141
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_html':
# url_ext_html
# False    49251
# True        85
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_htr':
# url_ext_htr
# False    49277
# True        59
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_inc':
# url_ext_inc
# False    49071
# True       265
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_java':
# url_ext_java
# False    49096
# True       240
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_jpg':
# url_ext_jpg
# False    46589
# True      2747
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_jpg~':
# url_ext_jpg~
# False    49329
# True         7
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_jsp':
# url_ext_jsp
# True     37450
# False    11886
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_jsp~':
# url_ext_jsp~
# False    49273
# True        63
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_none':
# url_ext_none
# False    48456
# True       880
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_nsf':
# url_ext_nsf
# False    49208
# True       128
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_old':
# url_ext_old
# False    49074
# True       262
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_portal':
# url_ext_portal
# False    49332
# True         4
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_properties':
# url_ext_properties
# False    49333
# True         3
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_txt':
# url_ext_txt
# False    49334
# True         2
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_xml':
# url_ext_xml
# False    49332
# True         4
# Name: count, dtype: int64
# ----------------------------------------
# Value counts for column 'url_ext_~':
# url_ext_~
# False    49302
# True        34
# Name: count, dtype: int64
# ----------------------------------------