## Depth-wise Induction of Decision Tree Using Presorted Linked Lists


Compile and test

```
$ g++ -fopenmp -std=c++11 -O3 -msse2 -funroll-loops test_dt.cpp -o test_dt -D _D2_DOUBLE -D N=1000000 -D D=28 -D MD=32 -D MW=100
$ OMP_NUM_THREADS=28 ./test_dt higgs-train-1m.csv
tree induction time: 5.667219 seconds
time: 6.793412 seconds
nleafs: 2216 
```

