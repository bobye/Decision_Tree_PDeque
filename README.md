## Leaf-wise Induction of Decision Tree with Presorted Deque



This is the proof-of-concept demo code for reproducing experiments in the arXiv note "A Faster Drop-in Implementation for Leaf-wise Exact Greedy Induction of Decision Tree Using Pre-sorted Deque" (https://arxiv.org/abs/1712.06989).


### Prepare sample data

- download data from [HIGGS](https://archive.ics.uci.edu/ml/datasets/HIGGS) and uncompress gz file.
- create training data `head -1000000 HIGGS.csv > higgs-train-1m.csv`
- create testing data `tail -50000 HIGGS.csv > higgs-test.csv`


### Compile and test

```
$ make
$ OMP_NUM_THREADS=28 ./test_dt higgs-train-1m.csv higgs-test.csv
tree induction time: 1.217055 seconds
training time: 2.237200 seconds
nleafs: 2216 
test accuracy: 0.710
```

TODO

- implement tree pruning
- implement random forest
- model serialization and communication


----
All rights reserved. Jianbo Ye
