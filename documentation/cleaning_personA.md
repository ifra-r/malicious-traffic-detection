# CSIC 2010 Dataset Preprocessing Summary

## ✅ 1. Initial Exploration
- Loaded `csic2010.csv`.
- Inspected dataset shape: **61,065 rows × 17 columns**.
- Logged:
  - Column names and data types.
  - Count of missing values per column.
  - Unique value counts and samples.

## ✅ 2. Column Cleanup
- Dropped:
  - Columns with only 1 unique value (e.g., `User-Agent`, `Pragma`, `language`).
  - Columns with unnecessary identifiers (e.g., `Unnamed: 0`).
  - `cookie` column: contained unique values for nearly all rows (61,065), thus not informative for modeling.

## ✅ 3. Low Variance Feature Detection
- Identified columns where a single value dominates (e.g., 99%+ same value).
- Dropped those columns as they add little predictive power.

## ✅ 4. Missing Value Handling
- Dropped columns with >50% missing data: `content-type`, `lenght`, `content`.
- Dropped rows with any remaining missing values (e.g., from `Accept` column).
- Logged before/after shape and reasons for each removal.

## ✅ 5. Class Column Balance
- Checked the balance of the `classification` column.
- If the class distribution fell outside the 45–55% range, downsampled the larger class to restore balance.
- Saved the balanced dataset as a new file.

---

## 🔢 Final Dataset State
- **Remaining Columns**: `Method`, `Accept`, `host`, `connection`, `classification`, `URL`
- **No missing values**
- **Balanced classification column (50-50)**
- **Useful categorical and textual features**
