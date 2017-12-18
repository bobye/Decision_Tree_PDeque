## Depth-wise Induction of Decision Tree With Presorted Dequeue


Compile and test

```
$ g++ -fopenmp -std=c++11 -O3 -msse2 -funroll-loops test_dt.cpp -o test_dt -D _D2_DOUBLE -D N=1000000 -D D=28 -D MD=32 -D MW=100
$ OMP_NUM_THREADS=28 ./test_dt higgs-train-1m.csv
tree induction time: 1.617345 seconds
time: 3.018330 seconds
nleafs: 2216
```

