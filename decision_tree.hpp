#ifndef _D2_DECISION_TREE_H_
#define _D2_DECISION_TREE_H_

#include "common.hpp"
#include "timer.h"
#include "traits.hpp"

// stl headers
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>

#ifdef RABIT_RABIT_H
#include <dmlc/io.h>
#endif

namespace d2 {
  namespace internal {

    struct _DT {
      constexpr static real_t prior_weight = 0.00;
    };
    
    /*! \brief base class for decision tree nodes
     * which includes shared functions and data members of both leaf and branch
    */
    template <size_t dim, class YStats>
    class _DTNode {
    public:
      YStats y_stats;

      typedef _DTLeaf<dim, YStats> Leaf;
      typedef _DTBranch<dim, YStats> Branch;

      _DTNode(){}
      _DTNode(const YStats &ys): y_stats(ys) {
      }
      
      /*! \brief get pointer to the leaf node by a given sample */
      virtual Leaf* get_leafnode(const real_t *X) = 0;

      virtual size_t get_leaf_count() = 0;
      
      /*! \brief write data into a stream buffer */
      virtual void write(std::ostream *fo) const = 0;
      /*! \brief read data from a stream buffer */
      virtual void read(std::istream *fi) = 0;

      std::string hashCode() const {
	std::stringstream ss;
	ss << (void const *) this;
	return ss.str();
      }

      virtual void dotgraph(std::ostream &f) const = 0;

      int parent;
    };    

    /*! \brief lead node in decision tree
     */
    template <size_t dim, class YStats>
    class _DTLeaf : public _DTNode<dim, YStats> {
    public:
      using typename _DTNode<dim, YStats>::Leaf;
      using typename _DTNode<dim, YStats>::Branch;
      using _DTNode<dim, YStats>::hashCode;

      _DTLeaf(){}
      _DTLeaf(const YStats &ys): _DTNode<dim, YStats>(ys) {
	label = ys.get_label();
      }
      
      /*! \brief construct a new leaf node from a branch node */
      _DTLeaf(const Branch &that) {
	this->y_stats = that.y_stats;
	//this->score = that.score;
	//this->weight = that.weight;
	//this->r = that.r;
	this->parent = that.parent;
	this->label = that.y_stats.get_label();
      }
      Leaf* get_leafnode(const real_t *X) {
	return this;
      }
      size_t get_leaf_count() {return 1.;}
      
      void dotgraph(std::ostream &f) const {
	f << "node" << hashCode() << " [label=\"" << label << "\", shape=box, style=filled ]\n";
      }
      
      void write(std::ostream *fo) const {
	
	fo->write((const char *) &this->label, sizeof(typename YStats::LabelType));
	fo->write((const char *) &this->parent, sizeof(int));
      }
      void read(std::istream *fi) {
	fi->read((char *) &this->label, sizeof(typename YStats::LabelType));
	fi->read((char *) &this->parent, sizeof(int));
      }

      typename YStats::LabelType label;
    };

    /*! \brief branch node in decision tree
     */
    template <size_t dim, class YStats>
    class _DTBranch : public _DTNode<dim, YStats> {
    public:
      using typename _DTNode<dim, YStats>::Leaf;
      using typename _DTNode<dim, YStats>::Branch;
      using _DTNode<dim, YStats>::hashCode;


      _DTBranch () {};
      _DTBranch (size_t i, real_t cto): index(i), cutoff(cto) {
      }

      Leaf* get_leafnode(const real_t *X) {
	assert(left && right);
	if (X[index]<cutoff) {
	  return left->get_leafnode(X);
	} else {
	  return right->get_leafnode(X);
	}
      }

      size_t get_leaf_count() {
	n_leafs = left->get_leaf_count() + right->get_leaf_count();
	return n_leafs;
      }

      void dotgraph(std::ostream &f) const {
	assert(left && right);
	left->dotgraph(f);
	right->dotgraph(f);

	f << "node" << hashCode() << " [label=\"x" << index << " < " << cutoff << "?\", style=filled]\n";
	f << "node" << hashCode() << " -> node" <<  left->hashCode() << " [label=\"yes\"]\n node" <<  hashCode() << " -> node" << right->hashCode() << "[label=\"no\"]\n";
      }


      void write(std::ostream *fo) const {
	fo->write((const char *) &this->nleft, sizeof(int));
	fo->write((const char *) &this->nright, sizeof(int));
	fo->write((const char *) &this->index, sizeof(size_t));
	fo->write((const char *) &this->cutoff, sizeof(real_t));
	fo->write((const char *) &this->parent, sizeof(int));
	fo->write((const char *) &this->n_leafs, sizeof(size_t));
      }
      void read(std::istream *fi) {
	fi->read((char *) &this->nleft, sizeof(int));
	fi->read((char *) &this->nright, sizeof(int));
	fi->read((char *) &this->index, sizeof(size_t));
	fi->read((char *) &this->cutoff, sizeof(real_t));
	fi->read((char *) &this->parent, sizeof(int));
	fi->read((char *) &this->n_leafs, sizeof(size_t));
      }

      _DTNode<dim, YStats> *left=nullptr, *right=nullptr;
      int nleft = -1, nright = -1;
      size_t index;
      real_t cutoff;
      size_t n_leafs;
    };

    
    /*! \brief node assignment data structure stores
     * the indexes of sample data
     */
    template <class YStats>
    struct ss_deque_t: public std::deque<sorted_sample<YStats> > {
      ss_deque_t(): std::deque<sorted_sample<YStats> >() {}
      ss_deque_t(const size_t n): std::deque<sorted_sample<YStats> >(n) {}
    };

    template <class YStats>
    struct node_assignment {
      size_t * ptr; ///< index array
      std::vector<ss_deque_t<YStats> *> sorted_samples;
      size_t size; ///< size of index array      
      size_t cache_offset; ///< offset to the cache array head, aka (ptr - cache_offset) should be constant
      int idx_cache_index;
      int depth;
      YStats y_stats;
    };

    struct index_cache {
      size_t index;
      int nleft;
      int nright;
    };

    /*! \brief the whole data structure used in building the decision trees
     */
    template <size_t dim, class YStats>    
    struct buf_tree_constructor {      
      //std::vector<std::vector<real_t> > X; ///< store data in coordinate-order
      std::vector<typename YStats::LabelType> y;
      std::vector<real_t> sample_weight;
      size_t max_depth;
      real_t min_leaf_weight;
      bool warm_start = false;
      //std::vector<sample> sample_cache;
      std::stack<std::tuple<node_assignment<YStats>, int> > tree_stack;

      // decision tree with presort
      //std::vector<std::vector<sorted_sample> > sorted_samples;
      std::vector<std::vector<size_t> > inv_ind_sorted;
      std::vector<char> sample_mask_cache;
    };
    

    template <size_t dim, class YStats, typename criterion>
    real_t best_split_ptr(std::deque<sorted_sample<YStats> > &sample_deque,
			  size_t n,
			  real_t &cutoff,
			  size_t &left_count,
			  const bool presort,
			  const YStats &y_stats) {
      assert(presort);
      
      real_t best_goodness = 0;

      YStats y_stats_left  = def::prepare<YStats, criterion>::left_op(y_stats);      
      YStats y_stats_right = def::prepare<YStats, criterion>::right_op(y_stats);
      
      const real_t no_split_score = criterion::op(y_stats);

      size_t i=0;
      typename YStats::LabelType label;
      for (auto sample = sample_deque.begin(); sample != sample_deque.end();) {	
	const real_t current_x = sample->x;
	typename YStats::LabelType yy = label = sample->y;
	while (i<n && (sample->x == current_x || yy == label)) {
	  y_stats_left.update_left(yy);
	  y_stats_right.update_right(yy);
	  i++; sample ++;
	  if (sample != sample_deque.end()) {
	    yy = sample->y;
	  }
	};
	if (i<n) {
	  label = yy;
	  const real_t goodness = no_split_score - YStats::template goodness_score<criterion>(y_stats_left, y_stats_right);
	  if (goodness > best_goodness) {
	    best_goodness = goodness;
	    cutoff = sample->x;
	    left_count = i;
	  }
	}
      }
      return best_goodness;
    }


    template <class YStats>
    void inplace_split_ptr(std::deque<sorted_sample<YStats> > &sample_deque,
			   node_assignment<YStats> &assignment) {
#pragma omp parallel for
      for (size_t i=0; i<assignment.size; ++i) {
	assignment.ptr[i] = sample_deque[i].index;
      }
    }    
    template <size_t dim, class YStats, typename criterion>
    _DTNode<dim, YStats>  *build_dtnode(node_assignment<YStats> &assignment,
					node_assignment<YStats> &aleft,
					node_assignment<YStats> &aright,
					buf_tree_constructor<dim, YStats> &buf,
					const bool presort,
					const int dim_index = -1) {
      // default: return leaf node
      aleft.ptr = NULL;
      aright.ptr= NULL;

      // make sure there is at least one sample
      assert(assignment.size > 0);

      // compute Y stats on the sample
      YStats y_stats = def::prepare<YStats, criterion>::left_op(assignment.y_stats);
      size_t *index=assignment.ptr;
#pragma omp for
      for (size_t ii = 0; ii < assignment.size; ++ii) {
	y_stats.update_left(buf.y[index[ii]]);
      }
      def::finalize<YStats, criterion>::op(y_stats);

      // build node
      if (assignment.size == 1 || 
	  assignment.depth == buf.max_depth || 
	  //r < 2 / all_class_w || 
	  //all_class_w < buf.min_leaf_weight
	  y_stats.stop()
	  ) {
	// if the condtion to create a leaf node is satisfied
	_DTLeaf<dim, YStats> *leaf = new _DTLeaf<dim, YStats>(y_stats);
	return leaf;
      } else {
	// if it is possible to create a branch node
	std::array<real_t, dim> goodness = {};
	std::array<real_t, dim> cutoff   = {};
	std::array<size_t, dim> left_count = {};

	// compute goodness split score across different dimensions
	//	if (dim_index >= 0) printf("cached index: %d\n", dim_index);
#pragma omp parallel for
	for (size_t ii = 0; ii < dim; ++ii)
	{
	  if (dim_index < 0 || ii == dim_index) {
	    auto &sorted_samples = assignment.sorted_samples[ii];	    
	    goodness[ii] = best_split_ptr<dim, YStats, criterion>
	      (*sorted_samples, assignment.size,
	       cutoff[ii], left_count[ii], presort, y_stats);
	  }
	}
	// pick the best goodness 
	real_t* best_goodness = std::max_element(goodness.begin(), goodness.end());
	size_t ii = best_goodness - goodness.begin();
	
	if (dim_index >= 0) assert(best_goodness - goodness.begin() == dim_index || *best_goodness == 0);
	if (*best_goodness == 0 ||
	    left_count[ii] < buf.min_leaf_weight ||
	    left_count[ii] > assignment.size - buf.min_leaf_weight) {
	  // if the best goodness is not good enough, a leaf node is still created
	  _DTLeaf<dim, YStats> *leaf = new _DTLeaf<dim, YStats>(y_stats);

	  return leaf;
	} else {
	  // otherwise, create a branch node subject to the picked dimension/goodness
	  _DTBranch<dim, YStats> *branch = new _DTBranch<dim, YStats>(ii, cutoff[ii]);

	  inplace_split_ptr(*assignment.sorted_samples[ii], assignment);

	  // create branched assignment
	  aleft.sorted_samples.resize(dim);
	  aleft.ptr = assignment.ptr;
	  aleft.size = left_count[ii];
	  aleft.cache_offset = assignment.cache_offset;
	  aleft.y_stats = y_stats;

	  aright.sorted_samples.resize(dim);
	  aright.ptr = assignment.ptr + left_count[ii];
	  aright.size = assignment.size - left_count[ii];
	  aright.cache_offset = assignment.cache_offset + left_count[ii];
	  aright.y_stats = y_stats;

	  if (presort) {
#pragma omp parallel for
	    for (size_t i=0; i<aleft.size; ++i) {
	      buf.sample_mask_cache[aleft.ptr[i]] = 'l';
	    }
#pragma omp parallel for
	    for (size_t i=0; i<aright.size; ++i){
	      buf.sample_mask_cache[aright.ptr[i]]= 'r';
	    }

#pragma omp parallel for
	    for (size_t ii=0; ii<dim; ++ii) {
	      auto &ass = assignment.sorted_samples[ii];
	      auto &left = aleft.sorted_samples[ii];
	      auto &right = aright.sorted_samples[ii];
	      left = new ss_deque_t<YStats>();
	      right = new ss_deque_t<YStats>();
	      for (size_t i=0; i<assignment.size; ++i) {
		auto &sorted_sample = ass->front();
		char mask = buf.sample_mask_cache[sorted_sample.index];
		if (mask == 'l') {
		  left->push_back(sorted_sample);
		} else if (mask == 'r') {
		  right->push_back(sorted_sample);
		}
		ass->pop_front();
	      }
	      delete ass;
	    }	    
	  }	  
	  return branch;
	}	
      }
    }


#define BIT_HIGH_POS 30
    template <size_t dim, class YStats>
    _DTNode<dim, YStats>*
    post_process_node_arr(std::vector<internal::_DTLeaf<dim, YStats> > &leaf_arr,
			  std::vector<internal::_DTBranch<dim, YStats> > &branch_arr) {
      for (auto iter = branch_arr.begin(); iter < branch_arr.end(); ++iter) {
	if (iter->nleft & 1<<BIT_HIGH_POS) {
	  iter->left = &branch_arr[iter->nleft & ~(1<<BIT_HIGH_POS)];
	} else {
	  iter->left = &leaf_arr[iter->nleft];
	}


	if (iter->nright & 1<<BIT_HIGH_POS) {
	  iter->right = &branch_arr[iter->nright & ~(1<<BIT_HIGH_POS)];
	} else {
	  iter->right = &leaf_arr[iter->nright];
	}
      }
      _DTNode<dim, YStats>* r;
      if (!branch_arr.empty()) {
	r = &branch_arr[0];
	//	printf("%zd\n", static_cast<_DTBranch<dim, n_class> *>(r)->nleft);
      } else {
	r = &leaf_arr[0];
      }
      return r;
    }
    template <size_t dim, class YStats, typename criterion>
    _DTNode<dim, YStats>* build_tree(size_t sample_size,
				     buf_tree_constructor<dim, YStats> &_buf,
				     node_assignment<YStats> &assign,
				     std::vector<internal::_DTLeaf<dim, YStats> > &leaf_arr,
				     std::vector<internal::_DTBranch<dim, YStats> > &branch_arr,
				     const bool presort) {
      std::vector<index_cache> index_arr;
      if (_buf.warm_start && branch_arr.size() > 0) {
	for (size_t ii = 0; ii < branch_arr.size(); ++ii) {
	  size_t index = branch_arr[ii].index;
	  int nleft, nright;
	  if (branch_arr[ii].nleft & (1<<BIT_HIGH_POS))
	    nleft = branch_arr[ii].nleft & ~(1<<BIT_HIGH_POS);
	  else
	    nleft = -1;

	  if (branch_arr[ii].nright& (1<<BIT_HIGH_POS))
	    nright = branch_arr[ii].nright & ~(1<<BIT_HIGH_POS);
	  else
	    nright = -1;

	  index_cache idc = {index, nleft, nright};
	  index_arr.push_back(idc);
	}
      } else {
	_buf.warm_start = false;
      }
      leaf_arr.clear();
      branch_arr.clear();

      auto &tree_stack = _buf.tree_stack;

      // create index array at root node
      std::vector<size_t> root_index(sample_size);
      for (size_t i=0; i<sample_size; ++i) root_index[i] = i;
      // create the node_assignment at root node and push into stack
      node_assignment<YStats> &root_assignment = assign;
      root_assignment.ptr = &root_index[0];
      root_assignment.size = sample_size;
      root_assignment.cache_offset = 0;
      root_assignment.idx_cache_index = 0;
      root_assignment.depth = 1;
      
      tree_stack.push(std::make_tuple(root_assignment, -1));

      // allocate cache memory
      // _buf.sample_cache.resize(sample_size);
      // to be returned
      _DTNode<dim, YStats> *root = nullptr;
      //printf("finish tree induction initialization!\n");

      
      // start to travel a tree construction using a stack
      auto current_sample_size_not_in_leaf = sample_size;
      
      while (!tree_stack.empty()) { 
	std::cout << "fetch data from the top node of stack ... " << std::flush;
	auto cur_tree = tree_stack.top(); 
	auto cur_assignment = std::get<0>(cur_tree);	
	int cur_parent = std::get<1>(cur_tree);

	node_assignment<YStats> assignment_left, assignment_right;
	_DTNode<dim, YStats> *node;
	if (_buf.warm_start && cur_assignment.idx_cache_index >= 0)
	  node = build_dtnode<dim, YStats, criterion> (cur_assignment,
						       assignment_left,
						       assignment_right,
						       _buf,
						       presort,
						       index_arr[cur_assignment.idx_cache_index].index);
	else
	  node = build_dtnode<dim, YStats, criterion> (cur_assignment,
						       assignment_left,
						       assignment_right,
						       _buf,
						       presort);
	node->parent = cur_parent; // set parent index
	tree_stack.pop();
	bool is_branch;
	if (assignment_left.ptr && assignment_right.ptr) {// spanning the tree	  
	  std::cout << "branching" << std::endl;
	  assignment_left.depth = cur_assignment.depth + 1;
	  assignment_right.depth = cur_assignment.depth + 1;
	  if (_buf.warm_start && cur_assignment.idx_cache_index >= 0) {
	    assignment_left.idx_cache_index   = index_arr[cur_assignment.idx_cache_index].nleft;
	    assignment_right.idx_cache_index= index_arr[cur_assignment.idx_cache_index].nright;
	  } else {
	    assignment_left.idx_cache_index = -1;
	    assignment_right.idx_cache_index = -1;
	  }	    
	  
	  is_branch = true;
	  tree_stack.push(std::make_tuple(assignment_left, branch_arr.size()));
	  tree_stack.push(std::make_tuple(assignment_right,branch_arr.size()));	  
	  branch_arr.push_back(std::move(*static_cast<_DTBranch<dim, YStats>* > (node)));	  
	} else {
	  current_sample_size_not_in_leaf -= cur_assignment.size;
	  std::cout << "reaching a leaf (" << current_sample_size_not_in_leaf << ")" << std::endl;
	  is_branch = false;
	  leaf_arr.push_back(std::move(*static_cast<_DTLeaf<dim, YStats>* > (node))); 
	}
	if (cur_parent >= 0) {
	  // set child node index
	  auto &parent = branch_arr[cur_parent];
	  size_t ind = (is_branch)? ((branch_arr.size()-1) | 1<<BIT_HIGH_POS) : (leaf_arr.size()-1);
	  if (parent.nright < 0)
	    parent.nright = ind;
	  else
	    parent.nleft = ind;	  
	}
      }

#ifdef COMPILE_PRUNING
      // start to pruning the constructed tree
      bool pruning = true;
      if (false) {	  
	root = post_process_node_arr(leaf_arr, branch_arr);
	real_t error_before_pruning = root->get_R();
	real_t weight = root->weight;	
	size_t n_leafs = root->get_leaf_count();
	real_t min_alpha = 0;
	std::cerr << getLogHeader() << "initial terminal nodes: "<<  n_leafs << std::endl;
	while (n_leafs > 512) {
	  // find the min(r-R)
	  std::stack<int> branch_ind_stack;
	  branch_ind_stack.push(0);
	  min_alpha = branch_arr[0].weight;
	  int min_ind;
	  while (!branch_ind_stack.empty()) {
	    int ind = branch_ind_stack.top();
	    real_t alpha = (branch_arr[ind].r - branch_arr[ind].R) / (branch_arr[ind].n_leafs - 1);
	    if (alpha < min_alpha) {
	      min_alpha = alpha;
	      min_ind = ind;
	    }
	    branch_ind_stack.pop();
	    if (branch_arr[ind].nleft & (1<<BIT_HIGH_POS))
	      branch_ind_stack.push(branch_arr[ind].nleft & ~(1<<BIT_HIGH_POS));
	    if (branch_arr[ind].nright& (1<<BIT_HIGH_POS))
	      branch_ind_stack.push(branch_arr[ind].nright &~(1<<BIT_HIGH_POS));
	  }
	  if (branch_arr[min_ind].parent < 0 || (weight - branch_arr[0].R) < (weight - error_before_pruning) * 0.99999) {
	    break; // terminate search
	  }
	  //	printf("%lf %d\n", min_alpha, min_ind);
	  //record pruning branch candidate
	  _DTLeaf<dim, YStats>* leaf = new _DTLeaf<dim, YStats>(branch_arr[min_ind]);
	  _DTBranch<dim, YStats> &parent = branch_arr[branch_arr[min_ind].parent];
	  if (parent.nleft == (min_ind | (1<<BIT_HIGH_POS))) {
	    parent.nleft = leaf_arr.size();
	    leaf_arr.push_back(std::move(*leaf));
	  } else if (parent.nright == (min_ind | (1<<BIT_HIGH_POS))) {
	    parent.nright = leaf_arr.size();
	    leaf_arr.push_back(std::move(*leaf));
	  } else {
	    assert(false);
	  }
	  // update R of ancestor of pruned branch
	  int ind = min_ind;
	  n_leafs -= (branch_arr[min_ind].n_leafs - 1);
	  while (branch_arr[ind].parent >= 0) {
	    ind = branch_arr[ind].parent;
	    branch_arr[ind].R += (branch_arr[min_ind].r - branch_arr[min_ind].R);
	    branch_arr[ind].n_leafs -= (branch_arr[min_ind].n_leafs - 1);
	  }
	}
	std::cerr << getLogHeader() << "remaining terminal nodes: "<<  n_leafs << std::endl;
      }
#endif      
      return post_process_node_arr(leaf_arr, branch_arr);
    }    
  }
  
  /*! \brief the decision tree class that is currently used in marriage learning framework 
   */
  template <size_t dim, class YStats, typename criterion>
  class Decision_Tree {
  public:
    typedef typename YStats::LabelType label_t;

    void init() {
      leaf_arr.clear();
      branch_arr.clear();
    }
    void predict(const real_t *X, const size_t n, label_t *y) const {
      const real_t* x = X;
      assert(root);
      for (size_t i=0; i<n; ++i, x+=dim) {
	auto leaf = root->get_leafnode(x);
	y[i] = leaf->label;
      }
    };

    void dotgraph(std::ostream &f) {
      assert(root);
      f << "digraph G {\n";
      root->dotgraph(f);
      f << "}\n";
    }
    
    int fit(const real_t *X, const label_t *y, 
	    const real_t *sample_weight, const size_t n,
	    bool sparse = false) {
      assert(X && y && !(sparse && !sample_weight));
      using namespace internal;
      // convert sparse data to dense
      const real_t *XX, *ss;
      const label_t *yy;

      size_t sample_size;
      if (sparse) {
	size_t nz = 0;
	for (size_t i = 0; i<n; ++i) nz += sample_weight[i] > 0;
	real_t* XX_ = new real_t [nz * dim];
	label_t* yy_ = new label_t [nz];
	real_t* ss_ = new real_t [nz];
	size_t count = 0;
	for (size_t i = 0; i<n; ++i)
	  if (sample_weight[i] > 0) {	  
	    for (size_t j = 0; j<dim; ++j) XX_[count*dim + j] = X[i*dim + j];
	    yy_[count] = y[i];
	    ss_[count] = sample_weight[i];
	    count ++;
	  }
	XX = XX_;
	yy = yy_;
	ss = ss_;
	sample_size = nz;
      } else {
	XX = X;
	yy = y;
	ss = sample_weight;
	sample_size = n;
      }
      
      buf.max_depth = max_depth;
      buf.min_leaf_weight = min_leaf_weight;
      buf.warm_start = true;

      if (!presorted) {
	prepare_presort(XX, yy, ss, sample_size, buf, assign);
	presorted = true;
      } else {
      }
      //printf("finish presorting!\n");
      
      double start=getRealTime();
      root = build_tree<dim, YStats, criterion>(sample_size, buf, assign, leaf_arr, branch_arr, true);
      printf("tree induction time: %lf seconds\n", getRealTime() - start);

      if (sparse) {
	delete [] XX;
	delete [] yy;
	delete [] ss;
      }
      return 0;
    }

    inline void set_communicate(bool bval) { communicate = bval; }
    inline void set_max_depth(size_t depth) { max_depth = depth; }
    inline void set_min_leaf_weight(real_t weight) { min_leaf_weight = weight; }

    typedef internal::_DTNode<dim, YStats > Node;
    typedef internal::_DTLeaf<dim, YStats > LeafNode;
    typedef internal::_DTBranch<dim, YStats > BranchNode;

#ifdef RABIT_RABIT_H_
    typedef rabit::utils::MemoryBufferStream MemoryBufferStream;
    /*! \brief synchronize between multiple processors */
    void sync(size_t rank) {
      bool no_model = false;
      if (rabit::GetRank() == rank) { // check if model exists
	if (leaf_arr.empty()) no_model = true;
      }
      rabit::Broadcast(&no_model, sizeof(bool), rank);
      if (no_model) return;
      
      std::string s_model;
      MemoryBufferStream fs(&s_model);
      size_t n_leaf = leaf_arr.size();
      size_t n_branch = branch_arr.size();

      rabit::Broadcast(&n_leaf, sizeof(size_t), rank);
      rabit::Broadcast(&n_branch, sizeof(size_t), rank);
      if (rabit::GetRank() != rank) {
        leaf_arr.resize(n_leaf);
	branch_arr.resize(n_branch);
      } else if (rabit::GetRank() == rank) {	
	save(&fs);
      }
      fs.Seek(0);
      rabit::Broadcast(&s_model, rank);
      //      if (rabit::GetRank() == rank) printf("%zd: %zd\t %zd\n", rank, n_leaf, n_branch);
      if (rabit::GetRank() != rank) {
	load(&fs);
	//	printf("%zd: load data from %zd\n", rabit::GetRank(), rank);
	this->root = internal::post_process_node_arr(leaf_arr, branch_arr);
	assert(root && !leaf_arr.empty());
      }
      rabit::Barrier();	      
    }
#endif

    /*! \brief helper function that caches data to stream */
    inline void save(std::ostream * fo) {
      size_t n_leaf = leaf_arr.size();
      size_t n_branch = branch_arr.size();
      fo->write((const char *) &n_leaf, sizeof(size_t));
      fo->write((const char *) &n_branch, sizeof(size_t));

      for (const LeafNode &leaf : leaf_arr) {
	leaf.write(fo);
      }
      for (const BranchNode &branch : branch_arr) {
	branch.write(fo);
      }
    }
    /*! \brief helper function that restores data from stream */
    inline void load(std::istream * fi) {
      size_t n_leaf;
      size_t n_branch;
      fi->read((char*) &n_leaf, sizeof(size_t));
      fi->read((char*) &n_branch, sizeof(size_t));

      leaf_arr.resize(n_leaf);
      branch_arr.resize(n_branch);

      for (LeafNode &leaf : leaf_arr) {
	leaf.read(fi);
      }
      for (BranchNode &branch : branch_arr) {
	branch.read(fi);
      }
      this->root = internal::post_process_node_arr(leaf_arr, branch_arr);
      assert(root && !leaf_arr.empty());
    }
    
    Node *root = nullptr;
  private:
    internal::buf_tree_constructor<dim, YStats> buf;
    internal::node_assignment<YStats> assign;
    std::vector<LeafNode> leaf_arr; 
    std::vector<BranchNode> branch_arr;
    size_t max_depth = 10;
    real_t min_leaf_weight = .0;
    bool presorted = false;
    bool communicate = true;
    
    void prepare_presort(const real_t *XX, const label_t *yy, const real_t* ss,
			 const size_t sample_size,
			 internal::buf_tree_constructor<dim, YStats> &buf,
			 internal::node_assignment<YStats> &assign) {

      /*
      buf.X.resize(dim);
      for (size_t k=0; k<dim; ++k) buf.X[k].resize(sample_size);
      */
      buf.y.resize(sample_size);
      buf.sample_weight.resize(sample_size, 1.);
      for (size_t i=0; i<sample_size; ++i) {
	/*
	for (size_t k=0; k<dim; ++k, ++j) {
	  buf.X[k][i]=XX[j];
	}
	*/
	buf.y[i] = yy[i];
      }
      if (ss)
	for (size_t i=0; i<sample_size; ++i) buf.sample_weight[i]=ss[i];
	

      assign.sorted_samples.resize(dim);
      buf.inv_ind_sorted.resize(dim);
      buf.sample_mask_cache.resize(sample_size);
#pragma omp parallel for
      for (size_t k=0; k<dim; ++k) {
	auto &sorted_samples = assign.sorted_samples[k];
	//auto &inv_ind_sorted = buf.inv_ind_sorted[k];
	sorted_samples = new internal::ss_deque_t<YStats>(sample_size);
	//inv_ind_sorted.resize(sample_size);
	//const real_t * XX = &buf.X[k][0];
	const real_t *XXX = XX + k;
	for (size_t i=0; i<sample_size; ++i, XXX+=dim) {
	  auto &sample = (*sorted_samples)[i];
	  sample = {*XXX, yy[i], i};
	  /*
	  if (ss)
	    sample.weight = ss[i];
	  else
	    sample.weight = 1.;
	  */
	}
	// presort
	std::sort(sorted_samples->begin(), sorted_samples->end(), internal::sorted_sample<YStats>::cmp);
	
      }      
    }

  };    
}

#include "classification.hpp"
#include "regression.hpp"
#include "day_sharpe.hpp"

#endif /* _D2_DECISION_TREE_H_ */
