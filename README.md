# Data Quality Metric (DQM) -- obsolete
Estimating the number of remaining errors (aka Data Quality Metric/`DQM`) in a dataset is an important problem. Previously (http://www.vldb.org/pvldb/vol10/p1094-chung.pdf), we have shown that some heuristic estimators can provide useful estimates to guide the data cleaning process (e.g., know when to stop cleaning). In this work, we hope to continue developing a new estimator with some correctness guarantees (e.g., upper/error bounds).

This is an old repository. I have moved this project to DQMflask (https://github.com/yeounoh/DQMflask).

## DQM Test Cases
* Compare `DQM` estimators
```python
python dqm_test.py DQMTest.test_estimators
```

## Amazon Mechanical Turks Expeirments
`amt/` folder contains codes for AMT experiments. The experiments (data cleaning or error detection) were crowdsourced on AMT.
```python
python amt/exp_restaurant_dataset.py
```
