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

            BinaryClassificationStats() : histogram({}), alpha(0.5) {
            }

            BinaryClassificationStats(real_t alpha) : histogram({}), alpha(alpha) {
            }

            BinaryClassificationStats(const BinaryClassificationStats &that) : histogram(that.histogram),
                                                                               alpha(that.alpha) {
            }

            using LabelType = Stats<_D2_CLTYPE>::LabelType;

            inline LabelType getLabel() const override {
                if (alpha * histogram[1] - (1 - alpha) * histogram[0] <= 0) {
                    return 0;
                } else {
                    return 1;
                }
            }

            inline void updateLeft(LabelType y) override {
                histogram[y]++;
            }

            inline void updateRight(LabelType y) override {
                histogram[y]--;
            }

            inline bool stop() const override {
                // todo
                return false;
            }

            template<class criterion>
            inline static real_t
            goodness_score(const BinaryClassificationStats left, const BinaryClassificationStats right) {
                return std::min(criterion::op(left), criterion::op(right));
            }

        };

        /*! \brief FN - TN */
        struct fntn {
            static inline real_t op(const BinaryClassificationStats &y_stats) {
                const auto &alpha = y_stats.alpha;
                return std::min(alpha * y_stats.histogram[1] - (1 - alpha) * y_stats.histogram[0], (real_t) 0.);
            }

            static inline real_t unnormalized_op(const BinaryClassificationStats &y_stats) {
                return op(y_stats);
            }
            
        };
    }

    namespace internal {

        template<>
        struct SortedSample<def::BinaryClassificationStats> {
            real_t x;
            _D2_CLTYPE y;
            size_t index;

            inline static bool cmp(const SortedSample<def::BinaryClassificationStats> &a,
                                   const SortedSample<def::BinaryClassificationStats> &b) {
                return a.x < b.x;
            }
        };
    }

}

#endif /* _BINARY_CLASSIFICATION_H_ */
