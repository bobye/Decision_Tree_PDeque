#ifndef _REGRESSION_H_
#define _REGRESSION_H_

#include "traits.hpp"

#define _D2_RGTYPE real_t

namespace d2 {
  //! \brief customizable template classes that extend the scope of decision tree implementations
  namespace def {
    struct RegressionStats : Stats<_D2_RGTYPE> {
      using LabelType = Stats<_D2_RGTYPE>::LabelType;
      size_t count;
      LabelType sum;
      LabelType sum_sq;


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
	sum += y;
	sum_sq += y * y;
      }

      inline void update_right(LabelType y) override {
	count --;
	sum -= y;
	sum_sq -= y * y;
      }

      inline bool stop() const override {
	// todo
	return false;
      }

      template <class criterion>
      inline static real_t goodness_score(const RegressionStats left, const RegressionStats right) {
	return
	  (criterion::op(left)  * left.count + criterion::op(right) * right.count)
	  / (left.count + right.count);
      }
    };

    /*! \brief mean square error function
     */
    struct mse {
      static inline real_t op(const RegressionStats &y_stats) {
	real_t mean = (real_t) y_stats.sum / (real_t) y_stats.count;
	return (real_t) y_stats.sum_sq / (real_t) y_stats.count - mean * mean;
      }
    };
  }

  namespace internal {
    template <>
    struct sorted_sample<def::RegressionStats > {
      real_t x;
      real_t y;
      //      real_t weight;
      size_t index;
      //sorted_sample *next;
      inline static bool cmp(const sorted_sample<def::RegressionStats > &a, 
			     const sorted_sample<def::RegressionStats > &b) {
	return a.x < b.x;
      }
    };
  }
}

#endif /* _REGRESSION_H_ */
