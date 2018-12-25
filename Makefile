ifeq ($(detected_OS), Linux)
	CFLAGS += -fopenmp
	LIB += -lrt
endif

all: *.cpp *.hpp
	g++ -std=c++11 -O3 -msse2 -funroll-loops test_day_sharpe.cpp -o test_day_sharpe -D _D2_SINGLE -D N=1000000 -D D=6 -D MD=8 -D MW=100 -D M=50000 $(LIB)
	g++ -std=c++11 -O3 -msse2 -funroll-loops test_dt.cpp -o test_dt -D _D2_SINGLE -D N=1000000 -D D=28 -D MD=32 -D MW=100 -D M=50000 -D USE_D2_CLTYPE $(LIB)
