
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

## Final Dataset Summary:

    Shape: (49336, 11)

    Data types: All columns are int64

### Sample Data (First 5 Rows):
classification	num_params	Method_GET	Method_POST	connection_Connection: close	connection_close	url_ext_css	url_ext_gif	url_ext_jpg	url_ext_jsp	url_ext_none
1	0	0	1	1	0	0	0	0	1	0
1	0	0	1	1	0	0	0	0	1	0
1	0	1	0	0	1	0	0	0	0	0
0	0	1	0	0	1	1	0	0	0	0
0	0	1	0	0	1	0	0	0	1	0
🔍 Column-Wise Value Distributions:

Below are the frequency counts for each feature:

classification
1    24668
0    24668
------------------------
num_params
0     34190
1      4851
5      4143
13     4118
3      2034
------------------------
Method_GET
1    34269
0    15067
------------------------
Method_POST
0    34269
1    15067
------------------------
connection_Connection: close
0    34269
1    15067
------------------------
connection_close
1    34269
0    15067
------------------------
url_ext_css
0    48659
1      677
------------------------
url_ext_gif
0    45447
1     3889
------------------------
url_ext_jpg
0    46589
1     2747
------------------------
url_ext_jsp
1    37450
0    11886
------------------------
url_ext_none
0    48456
1      880