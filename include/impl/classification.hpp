#ifndef _CLASSIFICATION_H_
#define _CLASSIFICATION_H_

#include "core/traits.hpp"

#include <array>
#include <cassert>
#include <cmath>


#define _D2_CLTYPE unsigned short int

namespace d2 {
  namespace def {
    //! \brief the Stats class for classification problems
    template <size_t n_class>
    struct ClassificationStats : public Stats<_D2_CLTYPE> {
      // member variables
      std::array<real_t, n_class+1> histogram;

      ClassificationStats (): histogram({}) {
      }
      
      ClassificationStats (const ClassificationStats<n_class> & that): histogram (that.histogram) {
      }
      
      using LabelType = Stats<unsigned short int>::LabelType;
      
      inline LabelType get_label() const override {
	return std::max_element(histogram.begin(), histogram.end()-1) - histogram.begin();
      }

      inline void update_left(LabelType y) override {
	histogram[y] ++;
	histogram.back() ++;
      }

      inline void update_right(LabelType y) override {
	histogram[y] --;
	histogram.back() --;
      }

      inline bool stop() const override {
	// todo
	return false;
      }

      template <class criterion>
      inline static real_t goodness_score(const ClassificationStats<n_class> left, const ClassificationStats<n_class> right) {
	return
	  (criterion::op(left)  * left.histogram.back() + criterion::op(right) * right.histogram.back())
	  / (left.histogram.back() + right.histogram.back());
      }

    };

    /*! \brief gini function used in make splits
     */    
    struct gini {
      template <size_t n_class>
      static inline real_t op(const ClassificationStats<n_class> &y_stats) {
	auto &proportion = y_stats.histogram;
	real_t total_weight_sq;
	total_weight_sq = proportion.back() * proportion.back();
	//if (total_weight_sq <= 0) return 1.;

	real_t gini = total_weight_sq;
	for (size_t i = 0; i<n_class; ++i)
	  gini -= proportion[i] * proportion[i];
	gini /= total_weight_sq;
	return gini;
      }
      static inline real_t loss(const real_t &x) {return 1-x;}
    };

    /*! \brief entropy function used in make splits
     */
    struct entropy {
      template<size_t n_class>
      static inline real_t op(const ClassificationStats<n_class> &y_stats) {
	auto &proportion = y_stats.histogram;
	real_t total_weight;
	total_weight = proportion.back();
	assert(total_weight > 0);

	real_t entropy = 0.;
	for (size_t i = 0; i<n_class; ++i) {
	  if (proportion[i] > 0) {
	    real_t p = proportion[i] / total_weight;
	    entropy -= log(p) * p ;
	  }
	}
      
	return entropy;
      }
      static inline real_t loss(const real_t &x) {return -log(x);}
    };


  }

  namespace internal {

    template <size_t n_class>
    struct sorted_sample<def::ClassificationStats<n_class> > {
      real_t x;
      unsigned short int y;
      //      real_t weight;
      size_t index;
      //sorted_sample *next;
      inline static bool cmp(const sorted_sample<def::ClassificationStats<n_class> > &a, 
			     const sorted_sample<def::ClassificationStats<n_class> > &b) {
	return a.x < b.x;
      }
    };
  }
}

#endif /* _CLASSIFICATION_H_ */
