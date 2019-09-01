#include "core/decision_tree.hpp"

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

#ifdef USE_D2_CLTYPE
typedef _D2_CLTYPE d2_label_t;
#elif defined USE_D2_RGTYPE
typedef _D2_RGTYPE d2_label_t;
#endif

template <typename LabelType>
void sample_naive_data(real_t *X, LabelType *y, real_t *w, size_t n) {
  for (size_t i=0; i<n; ++i) {
    y[i] = ((real_t) rand() / (real_t) RAND_MAX) < 0.02;
    if (y[i]) {
      for (size_t j=0; j<D; ++j)
	X[i*D+j]=(real_t) rand() / (real_t) RAND_MAX;
    } else {
      for (size_t j=0; j<D; ++j)
	X[i*D+j]=(real_t) rand() / (real_t) RAND_MAX - 0.01;
    }
    if (w) w[i] = 1.; // (real_t) rand() / (real_t) RAND_MAX;
  }  
}

template <typename LabelType>
std::pair<real_t, real_t> metric(LabelType *y_pred, LabelType *y_true, size_t n) {
  size_t fn=0, fp=0, tn=0, tp=0;
  for (size_t i=0; i<n; ++i) {
    if (y_true[i] == 0) {
      if (y_pred[i] < 0.5) tn ++;
      else fn ++;
    } else {
      if (y_pred[i] > 0.5) tp ++;
      else fp ++;
    }
  }
  return std::make_pair((real_t) fp/ (real_t) (fp + tp), (real_t) fn/ (real_t) (fn + tn));
}

int main(int argc, char* argv[]) {

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
      y[i] = (d2_label_t) stof(number);
      for (auto j=0; j<D; ++j) {
	getline(ss, number, ',');
	X[i*D+j] = stof(number);
      }
    }
    train_fs.close();
  }


  // create classifier
#if USE_D2_CLTYPE
  auto classifier = new Decision_Tree<D, def::ClassificationStats<NC>, def::gini>();
  //auto classifier = new Decision_Tree<D, def::BinaryClassificationStats, def::fntn>();
#elif USE_D2_RGTYPE
  auto classifier = new Decision_Tree<D, def::RegressionStats, def::mse>();
#endif
  
  classifier->init();
  classifier->set_max_depth(MD);
  classifier->set_min_leaf_weight(MW);
  // training
  double start=getRealTime();
  classifier->fit(X, y, w, N);
  printf("training time: %lf seconds\n", getRealTime() - start);
  printf("nleafs: %zu \n", classifier->root->get_leaf_count());

  if (argc == 1) {
    // prepare naive testing data
    sample_naive_data(X, y, w, M);
    classifier->predict(X, M, y_pred);

    // output result
    auto result = metric(y_pred, y, M);
    printf("test metric: FP %.3f, FN %.3f\n", std::get<0>(result), std::get<1>(result));
  } else if (argc == 3) {
    assert(M < N);
    ifstream test_fs;
    test_fs.open(argv[2]);
    for (auto i=0; i<M; ++i) {
      string line;
      getline(test_fs, line);
      istringstream ss(line);
      string number;
      getline(ss, number, ',');
      y[i] = stof(number);
      for (auto j=0; j<D; ++j) {
	getline(ss, number, ',');
	X[i*D+j] = stof(number);
      }
    }
    test_fs.close();
    classifier->predict(X, M, y_pred);
    auto result = metric(y_pred, y, M);
    printf("test metric: FP %.3f, FN %.3f\n", std::get<0>(result), std::get<1>(result));
  }

  delete [] X;
  delete [] y;
  delete [] y_pred;

  return 0;
}
