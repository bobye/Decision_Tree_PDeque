CC=g++
CFLAGS=
LIB=
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CFLAGS += -fopenmp
	LIB += -lrt
endif

all: test/*.cpp
	$(CC) $(CFLAGS) -std=c++11 -O3 -msse2 -funroll-loops -I include test/test_day_sharpe.cpp -o build/test_day_sharpe -D _D2_SINGLE -D N=1000000 -D D=28 -D MD=8 -D MW=100 -D M=50000 $(LIB)
	$(CC) $(CFLAGS) -std=c++11 -O3 -msse2 -funroll-loops -I include test/eval_day_sharpe.cpp -o build/eval_day_sharpe -D _D2_SINGLE -D N=1000000 -D D=28 -D MD=8 -D MW=100 -D M=50000 $(LIB)
	$(CC) $(CFLAGS) -std=c++11 -O3 -msse2 -funroll-loops -I include test/test_dt.cpp -o build/test_dt -D _D2_SINGLE -D N=1000000 -D D=28 -D MD=24 -D MW=1000 -D M=50000 -D USE_D2_CLTYPE $(LIB)
	$(CC) $(CFLAGS) -std=c++11 -O3 -msse2 -funroll-loops -I include tools/sharpe_finder.cpp -o build/sharpe_finder -D _D2_SINGLE -D DIMENSION=28 -D DAYS=100 -D DAYS_TEST=100 -D MD=8

