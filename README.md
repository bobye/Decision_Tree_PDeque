## Leaf-wise Induction of Decision Tree with Presorted Deque



This is the proof-of-concept demo code for reproducing experiments in the arXiv note "A Faster Drop-in Implementation for Leaf-wise Exact Greedy Induction of Decision Tree Using Pre-sorted Deque" (https://arxiv.org/abs/1712.06989).


### Prepare sample data

- download data from [HIGGS](https://archive.ics.uci.edu/ml/datasets/HIGGS) and uncompress gz file.
- create training data `head -1000000 HIGGS.csv > higgs-train-1m.csv`
- create testing data `tail -50000 HIGGS.csv > higgs-test.csv`


### Compile and test

```
$ make
$ OMP_NUM_THREADS=28 ./build/test_dt higgs-train-1m.csv higgs-test.csv
tree induction time: 1.475672 seconds
training time: 2.048821 seconds
nleafs: 1845 
test metric: FP 0.276, FN 0.317, Sensitivity 0.720, Specificity 0.687, Accuracy 0.705
```

### Other tests on synthetic data
```
$ OMP_NUM_THREADS=12 ./build/test_dt 
tree induction time: 1.425927 seconds
training time: 2.048105 seconds
nleafs: 24 
test metric: FP 0.801, FN 0.000, Sensitivity 1.000, Specificity 0.985, Accuracy 0.985
```
----
All rights reserved (2017-2023). Jianbo Ye
