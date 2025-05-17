
# CSIC 2010 Dataset Preprocessing – Khadijah’s Step Summary

## ✅ 6. Target Column Fix 

* **Loaded** `3_balanced_data.csv` (pre-balanced dataset).
* **Converted** `classification` column from string to integer.
* **Saved** the fixed dataset as `4_fixed_target.csv`.
* ✅ Output check:

  ```
  classification
  0               1
  1               1
  2               1
  3               0
  4               0
  ```

---

## ✅ 7. Content Tokenization from URL

* **Loaded** `4_fixed_target.csv`.
* **Parsed** query parameters from the `URL` column.
* **Tokenized** content names (parameter keys only, not values).
* **Saved** result as `5_clean_content.csv`.
* ✅ Output sample:

  ```
  content_tokens
  []
  []
  []
  []
  []
  ```

---

## ✅ 8. URL Structure Decomposition

* **Loaded** `5_clean_content.csv`.
* **Extracted**:

  * `url_path`: Path portion of the URL.
  * `url_query`: Raw query string.
* **Saved** updated file as `6_reprocess_url.csv`.
* ✅ Output sample:

  ```
  URL                                              | url_path                       | url_query
  ---------------------------------------------------------------------------------------------
  http://localhost:8080/tienda1/publico/anadir.jsp | /tienda1/publico/anadir.jsp   | 
  ```

---

## ✅ 9. Categorical Encoding

* **Loaded** `6_reprocess_url.csv`.
* **Removed** HTTP protocol suffix from `URL` if present.
* **Encoded** categorical features using `LabelEncoder`:

  * `Method`, `host`, `connection`
* **Saved** encoded dataset as `7_encoded.csv`.
* ✅ Output sample:

  ```
  Method  host  connection
  0       1     0          0
  1       1     0          0
  2       0     0          1
  ```

---

## ✅ 10. Train-Validation-Test Split

* **Loaded** `7_encoded.csv`.
* **Performed** stratified split on `classification`:

  * 60% training → `8_train.csv`
  * 20% validation → `9_val.csv`
  * 20% test → `10_test.csv`
* ✅ Output stats:

  ```
  Train size: 29601, Val size: 9867, Test size: 9868
  ```

---

## 📦 Final Dataset State (Khadijah's Version)

* `classification` fixed as integer type
* `content_tokens`, `url_path`, `url_query` extracted from `URL`
* Categorical columns encoded: `Method`, `host`, `connection`
* Clean URLs (no trailing HTTP suffixes)
* Stratified split done: train, val, test sets created and saved

