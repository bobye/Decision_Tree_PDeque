#include "core/decision_tree.hpp"
#include "utility/CLI11.hpp"

#include <random>
#include <fstream>
#include <sstream>
using namespace d2;
using namespace d2::def;
using namespace std;

typedef reward_date_pair d2_label_t;

void metric_time(reward_date_pair *y_pred, reward_date_pair *y_true, size_t n, unsigned long long int *orderid) {
  std::array<real_t, DAYS_TEST> k = {};
  std::set<unsigned long long int> orderid_set;
  int current_day = -1;
  for (size_t i=0; i<n; ++i) {
    if (y_true[i].date != current_day) {
      current_day = y_true[i].date;
      orderid_set.clear();
    }
    if (orderid_set.find(orderid[i]) != orderid_set.end()) continue;

    if (y_pred[i].reward != 0) {
      k[y_true[i].date] += y_true[i].reward;
      orderid_set.insert(orderid[i]);
    }
  }

  auto stats = _sharpe_helper(k);
  std::cout << "mean:   " << stats.mean << std::endl;
  std::cout << "std:    " << stats.std  << std::endl;
  std::cout << "sharpe: " << stats.sharpe << std::endl;
}

int read_data (const string filename, vector<real_t> &X, vector<d2_label_t> &y,
	       vector<unsigned long long int> &order_id, bool has_order_id = false) {
  ifstream train_fs;
  if (filename.empty()) {
    return -1;
  }

  train_fs.open(filename);
  if (!train_fs.good()) {
    cerr <<"Error: cannot open " << filename << endl;
    return -1;
  }
  string line;

  while (getline(train_fs, line)) {
    istringstream ss(line);
    string number;
    if (has_order_id) {
      getline(ss, number, ',');
      order_id.push_back(stoll(number));
    }
    y.push_back(d2_label_t());
    getline(ss, number, ',');
    y.back().reward = (real_t) stof(number);
    getline(ss, number, ',');
    y.back().date = (size_t) stoi(number);
    for (auto j=0; j<DIMENSION; ++j) {
      getline(ss, number, ',');
      X.push_back(stof(number));
    }
  }
  train_fs.close();
  cout << "finished data (" << y.size() << ") load from "<< filename << endl;
  return 0;
}

int main(int argc, char* argv[]) {


  CLI::App app{"Find high sharpe node using a cascade of decision trees"};

  string filenames_buffer;
  string test_filename;

  app.add_option("-f,--files", filenames_buffer, "a comma seperated filename strings");
  app.add_option("-t,--test", test_filename, "filename of test data");
  CLI11_PARSE(app, argc, argv);

  auto tokenize = [](std::string const &str, const char delim,
		     std::vector<std::string> &out) {
    // construct a stream from the string
    std::stringstream ss(str);
    
    std::string s;
    while (std::getline(ss, s, delim)) {
      out.push_back(s);
    }
  };

  std::vector<string> filenames;
  tokenize(filenames_buffer, ',', filenames);

  vector< Decision_Tree<DIMENSION, DaySharpeStats<DAYS>, sharpe> > classifiers (filenames.size());

  for (auto i=0; i < filenames.size(); ++i) {
    string filename = filenames[i];
    vector<real_t> X;
    vector<d2_label_t> y;
    vector<unsigned long long int> orders;
    read_data(filename, X, y, orders);
    
    vector<real_t> X_reduced;
    vector<d2_label_t> y_reduced;

    size_t n = y.size();
    for (size_t nn = 0; nn < n; ++ nn) {
      bool skip = false;
      for (int j=0; j<i; ++j) {
	d2_label_t y_pred;
	classifiers[j].predict(&X[DIMENSION*nn], 1, &y_pred);
	if (y_pred.reward == 0) {
	  skip = true;
	  break;
	}
      }

      if (!skip) {
	real_t *xx = &X[DIMENSION*nn];	
	y_reduced.push_back(y[nn]);
	for (auto d=0; d<DIMENSION; ++d) 
	  X_reduced.push_back(xx[d]);
      }
    }
    X.clear();
    y.clear();
    cout << "tree construct from reduced data (" << y_reduced.size() << ")" << endl;

    auto &classifier = classifiers[i];

    classifier.init();
    classifier.set_max_depth(8);
    classifier.set_min_leaf_weight(100);
    // training
    double start=getRealTime();
    classifier.fit(X_reduced.data(), y_reduced.data(), NULL, y_reduced.size());
    printf("training time: %lf seconds\n", getRealTime() - start);
    printf("nleafs: %zu \n", classifier.root->get_leaf_count());
    X_reduced.clear();
    y_reduced.clear();

    std::ostringstream oss;
    classifier.dotgraph(oss);
    std::cout << oss.str() << std::endl;
    
    std::fstream f;
    string tree_file = "tree.bin." + to_string(i);
    f.open(tree_file.c_str(), std::fstream::out);
    classifier.save(&f);
    f.close();

    // read test data
    if (read_data(test_filename, X, y, orders, true) == 0) {
      vector<d2_label_t> y_test(y.size());
      classifier.predict(X.data(), y.size(), y_test.data());
      // output result
      metric_time(y_test.data(), y.data(), y.size(), orders.data());
    }
  }



  
  return 0;
}
