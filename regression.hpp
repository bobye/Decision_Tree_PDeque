#ifndef _REGRESSION_H_
#define _REGRESSION_H_

#include "traits.hpp"

namespace d2 {
  //! \brief customizable template classes that extend the scope of decision tree implementations
  namespace def {
    template <typename Type>
    struct RegressionStats : Stats<real_t> {
      size_t count;
      Type sum;
      Type sum_sq;

      using LabelType = Stats<real_t>::LabelType;

      RegressionStats(): count(0), sum(0), sum_sq(0) {}

      RegressionStats(const RegressionStats & that): 
	count(that.count), sum(that.sum), sum_sq(that.sum_sq) {
      }

      RegressionStats & operator=(const RegressionStats & that) {
	count = that.count;
	sum = that.sum;
	sum_sq = that.sum_sq;
	return *this;
      }

      inline LabelType get_label() const override {
	return (LabelType) sum / (LabelType) count;
      }

      inline void update_left(LabelType y) override {
	count ++;
	sum += (Type) y;
	sum_sq += (Type) y * (Type) y;
      }

      inline void update_right(LabelType y) override {
	count --;
	sum -= (Type) y;
	sum_sq -= (Type) y * (Type) y;
      }

      inline bool stop() const override {
	// todo
	return false;
      }

      template <class criterion>
      inline static real_t goodness_score(const RegressionStats<Type> left, const RegressionStats<Type> right) {
	return
	  (criterion::op(left)  * left.count + criterion::op(right) * right.count)
	  / (left.count + right.count);
      }
    };

    /*! \brief mean square error function
     */
    struct mse {
      template <typename Type>
      static inline real_t op(const RegressionStats<Type> &y_stats) {
	real_t mean = (real_t) y_stats.sum / (real_t) y_stats.count;
	return (real_t) y_stats.sum_sq / (real_t) y_stats.count - mean * mean;
      }
    };
  }

  namespace internal {
    template <class YStats>
    struct sorted_sample;

    template <typename Type>
    struct sorted_sample<def::RegressionStats<Type> > {
      real_t x;
      real_t y;
      //      real_t weight;
      size_t index;
      //sorted_sample *next;
      inline static bool cmp(const sorted_sample<def::RegressionStats<Type> > &a, 
			     const sorted_sample<def::RegressionStats<Type> > &b) {
	return a.x < b.x;
      }
    };
  }
}

#endif /* _REGRESSION_H_ */
