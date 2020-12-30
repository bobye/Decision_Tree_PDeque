#ifndef _REGRESSION_H_
#define _REGRESSION_H_

#include "core/common.hpp"
#include "core/traits.hpp"

namespace d2 {
    //! \brief customizable template classes that extend the scope of decision tree implementations
    namespace def {
        struct RegressionStats : Stats<_D2_RGTYPE> {
            using LabelType = Stats<_D2_RGTYPE>::LabelType;
            size_t count;
            double sum;
            double sum_sq;


            RegressionStats() : count(0), sum(0), sum_sq(0) {}

            RegressionStats(const RegressionStats &that) :
                    count(that.count), sum(that.sum), sum_sq(that.sum_sq) {
            }

            inline LabelType getLabel() const override {
                return (LabelType) sum / (LabelType) count;
            }

            inline void updateLeft(LabelType y) override {
                count++;
                sum += y;
                sum_sq += y * y;
            }

            inline void updateRight(LabelType y) override {
                count--;
                sum -= y;
                sum_sq -= y * y;
            }

            inline bool stop() const override {
                // todo
                return false;
            }

            template<class criterion>
            inline static real_t goodness_score(const RegressionStats left, const RegressionStats right) {
                return
                        (criterion::op(left) * left.count + criterion::op(right) * right.count)
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

            static inline real_t unnormalized_op(const RegressionStats &y_stats) {
                return op(y_stats) * y_stats.count;
            }
        };
    }

    namespace internal {
        template<>
        struct SortedSample<def::RegressionStats> {
            real_t x;
            real_t y;
            //      real_t weight;
            size_t index;

            //SortedSample *next;
            inline static bool cmp(const SortedSample<def::RegressionStats> &a,
                                   const SortedSample<def::RegressionStats> &b) {
                return a.x < b.x;
            }
        };
    }
}

#endif /* _REGRESSION_H_ */
