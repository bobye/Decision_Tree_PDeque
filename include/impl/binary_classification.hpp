#ifndef _BINARY_CLASSIFICATION_H_
#define _BINARY_CLASSIFICATION_H_

#include "core/common.hpp"
#include "core/traits.hpp"

#include <array>
#include <cassert>
#include <cmath>

namespace d2 {
  namespace def {
    //! \brief the Stats class for binary classification problem
    struct BinaryClassificationStats : public Stats<_D2_CLTYPE> {
      // member variables
      std::array<real_t, 2> histogram;
      real_t alpha;

      BinaryClassificationStats (): histogram ({}), alpha (0.5) {
      }

      BinaryClassificationStats (real_t alpha): histogram ({}), alpha (alpha) {
      }

      BinaryClassificationStats (const BinaryClassificationStats &that) : histogram (that.histogram), alpha (that.alpha) {
      }

      using LabelType = Stats<_D2_CLTYPE>::LabelType;

      inline LabelType get_label() const override {
	if (alpha * histogram[0] - (1-alpha) * histogram[1] >= 0) {
	  return 0;
	} else {
	  return 1;
	}
      }

      inline void update_left(LabelType y) override {
	histogram[y] ++;
      }

      inline void update_right(LabelType y) override {
	histogram[y] --;
      }

      inline bool stop() const override {
	// todo
	return false;
      }

      template <class criterion>
      inline static real_t goodness_score(const BinaryClassificationStats left, const BinaryClassificationStats right) {
	return std::min(criterion::op(left), criterion::op(right));
      }
      
    };

    /*! \brief FN - TN */
    struct fntn {
      static inline real_t op(const BinaryClassificationStats &y_stats) {
	const auto &alpha = y_stats.alpha;
	return std::min(alpha * y_stats.histogram[0] - (1-alpha) * y_stats.histogram[1], (real_t) 0.);
      }
    };
  }

  namespace internal {

    template <>
    struct sorted_sample<def::BinaryClassificationStats> {
      real_t x;
      _D2_CLTYPE y;
      size_t index;
      inline static bool cmp(const sorted_sample<def::BinaryClassificationStats> &a, 
			     const sorted_sample<def::BinaryClassificationStats> &b) {
	return a.x < b.x;
      }
    };
  }

}

#endif /* _BINARY_CLASSIFICATION_H_ */
