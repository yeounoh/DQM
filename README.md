# Data Quality Metric (DQM)
Estimating the number of remaining errors (aka Data Quality Metric or `DQM`) in a dataset is an important problem. Previously (http://www.vldb.org/pvldb/vol10/p1094-chung.pdf), we have shown that some heuristic estimators can provide useful estimates to guide the data cleaning process (e.g., know when to stop cleaning). In this work, we hope to continue developing a new estimator with some correctness guarantees (e.g., upper/error bounds).

## DQM Test Cases
* Compare `DQM` estimators
```python
python dqm_test.py DQMTest.test_estimators
```
