## Depth-wise Induction of Decision Tree with Presorted Deque

This is the proof-of-concept demo code for reproducing experiments in the arXiv note "A Faster Drop-in Implementation for Depth-wise Exact Greedy Induction of Decision Tree Using Pre-sorted Deque".




Compile and test

```
$ g++ -fopenmp -std=c++11 -O3 -msse2 -funroll-loops test_dt.cpp -o test_dt -lrt
  -D _D2_DOUBLE \
  -D N=1000000 \
  -D D=28 \
  -D MD=32 \
  -D MW=100 \
  -D M=50000
$ OMP_NUM_THREADS=28 ./test_dt higgs-train-1m.csv higgs-test.csv
tree induction time: 1.617345 seconds
time: 3.018330 seconds
nleafs: 2216
test accuracy: 0.710
```

TODO

- implement tree pruning
- implement random forest
- model serialization and communication


----
All rights reserved.