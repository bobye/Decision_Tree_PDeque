#include "decision_tree.hpp"

#include <random>
#include <fstream>
#include <sstream>
using namespace d2;
using namespace std;


/* training sample size */
#ifndef N
#define N 1000000
#endif

/* testing sample size */
#ifndef M
#define M 0
#endif

/* dimension of features */
#ifndef D
#define D 28
#endif

/* number of classes */
#ifndef NC
#define NC 2
#endif

/* maximal depth */
#ifndef MD
#define MD 8
#endif

/* minimal sample weight (size) */
#ifndef MW
#define MW .0
#endif

#ifndef DAYS
#define DAYS 100
#endif


using namespace d2::def;

typedef reward_date_pair d2_label_t;

template <typename LabelType>
void sample_naive_data(real_t *X, LabelType *y, real_t *w, size_t n);

template <typename LabelType>
real_t metric(LabelType *y_pred, LabelType *y_true, size_t n);


template <>
void sample_naive_data<reward_date_pair>(real_t *X, reward_date_pair *y, real_t *w, size_t n) {
  for (size_t i=0; i<n; ++i) {
    y[i].reward = 2* (rand() % 2) - 1;
    y[i].date   = rand() % DAYS;
    if (y[i].reward) {
      for (size_t j=0; j<D; ++j)
	X[i*D+j]=(real_t) rand() / (real_t) RAND_MAX;
    } else {
      for (size_t j=0; j<D; ++j)
	X[i*D+j]=(real_t) rand() / (real_t) RAND_MAX - .1;
    }
    if (w) w[i] = 1.; // (real_t) rand() / (real_t) RAND_MAX;
  }  
}

template <>
real_t metric<reward_date_pair>(reward_date_pair *y_pred, reward_date_pair *y_true, size_t n) {
  std::array<real_t, DAYS> k = {};
  for (size_t i=0; i<n; ++i)
    if (y_pred[i].reward != 0) k[y_true[i].date] += y_true[i].reward;

  return _sharpe_helper(k);
}


int main(int argc, char* argv[]) {
  assert(N >= M);
  real_t *X, *w=NULL;
  d2_label_t *y, *y_pred;

  // prepare naive training data
  X = new real_t[D*N];
  y = new d2_label_t[N];
  //w = new real_t[N];
  y_pred = new d2_label_t[M];

  if (argc == 1) {
    sample_naive_data(X, y, w, N);
  } else {
    ifstream train_fs;
    train_fs.open(argv[1]);
    for (auto i=0; i<N; ++i) {
      string line;
      getline(train_fs, line);
      istringstream ss(line);
      string number;
      getline(ss, number, ',');
      y[i].reward = (real_t) stof(number);
      getline(ss, number, ',');
      y[i].date = (size_t) stoi(number);
      for (auto j=0; j<D; ++j) {
	getline(ss, number, ',');
	X[i*D+j] = stof(number);
      }
    }
    train_fs.close();
    std::cout << "finished data load!" << std::endl;
  }

  auto classifier = new Decision_Tree<D, DaySharpeStats<DAYS>, sharpe>();


  classifier->init();
  classifier->set_max_depth(MD);
  classifier->set_min_leaf_weight(MW);
  // training
  double start=getRealTime();
  classifier->fit(X, y, w, N);
  printf("training time: %lf seconds\n", getRealTime() - start);
  printf("nleafs: %zu \n", classifier->root->get_leaf_count());

  std::ostringstream oss;
  classifier->dotgraph(oss);
  std::cout << oss.str() << std::endl;

  if (argc == 1) {
    sample_naive_data(X, y, w, M);
    classifier->predict(X, M, y_pred);
  } else {
    // use training set to compute sharpe
    classifier->predict(X, M, y_pred);
  }

  // output result
  printf("test sharpe: %.3f\n", -metric(y_pred, y, M) );  

  delete [] X;
  delete [] y;
  delete [] y_pred;
  
  return 0;
}
