# Cleaning Process Documentation

**Dataset**: UNSW-NB15  
This document summarizes the step-by-step cleaning and preprocessing pipeline applied to the UNSW-NB15 dataset to prepare it for machine learning tasks. The process involves handling missing values, encoding categorical variables, analyzing features, normalizing data, and removing skewed or low-utility features.

---

## Step 1: Handling Missing Values

**Script**: `1_missingVals.py`

- **Total number of rows in original dataset**: 135,000  
- **Rows containing '-' in the `service` column**: 72,590  
- **Rows remaining after dropping**: 62,410  

The dataset was filtered by removing all rows where the `service` column contained `'-'`, a placeholder for missing or invalid data.

**Output**: Cleaned dataset saved as  
`data/cleaned/1_missingVals.csv`

---

## Step 2: Encoding Categorical Features

**Script**: `2_encoding.py`

Identified categorical columns and converted them into numeric codes for model compatibility:

- `label`: Binary classification target with values `[0, 1]`
- `is_sm_ips_ports`: Constant column with a single unique value `[0]`
- `service`: 12 unique values (e.g., `ftp`, `smtp`, `http`) encoded as integers
- `state`: 5 unique values encoded as integers
- `attack_cat`: 9 unique attack categories encoded as integers
- `proto`: 2 unique protocol types (`tcp`, `udp`) encoded as integers

**Output**: Encoded dataset saved as  
`data/cleaned/2_encoded_data.csv`

---

## Step 3: Feature Analysis

**Script**: `3_analyseFeatures.py`

- **Total columns analyzed**: 36  
- **Data types**: Mostly `int64` and `float64`  
- Reported **unique values per column** to identify categorical vs continuous features  
- Identified likely **continuous features** (those with more than 20 unique values), such as `dur`, `sbytes`, `dbytes`, `rate`, and network packet metrics  
- Confirmed some columns have **very low variability** (e.g., `is_sm_ips_ports` with only 1 unique value)  

This analysis helps decide which features to normalize or remove.

**Output**: No file output at this stage (exploratory step)

---

## Step 4: Data Normalization

**Script**: `4_normalise_Data.py`

- Continuous features identified in Step 3 were **scaled/normalized** to standard ranges
- Improves **model training stability and performance**

**Output**: Normalized dataset saved as  
`data/cleaned/3_normalised_data.csv`

---

## Step 5: Feature Removal and Dataset Balancing

**Script**: `5_remove_features.py`

- Checked **low-cardinality features** for skewness and model impact  
- Dropped the following **skewed or low-utility columns**:
  - `proto`
  - `swin`
  - `dwin`
  - `trans_depth`
  - `is_ftp_login`
  - `ct_ftp_cmd`
  - `is_sm_ips_ports`

- Also dropped the `attack_cat` column (likely to avoid multi-class complexity for binary classification)

Balanced the dataset by equalizing the number of samples per label class:

- **Final counts**:  
  - `15,041` samples for label `1` (attack)  
  - `15,041` samples for label `0` (normal)

**Output**: Cleaned, balanced dataset saved as  
`data/cleaned/4_remove_features.csv`

---

## Summary

This pipeline cleans the UNSW-NB15 dataset by:

- Removing rows with missing or invalid data  
- Encoding categorical variables numerically  
- Analyzing feature types and distributions  
- Normalizing continuous variables  
- Removing skewed or non-informative features  
- Balancing the dataset for binary classification tasks

Each step produces an intermediate cleaned dataset saved under `data/cleaned/` for traceability and reproducibility.
