#ifndef _CLASSIFICATION_H_
#define _CLASSIFICATION_H_

#include "core/common.hpp"
#include "core/traits.hpp"

#include <array>
#include <cassert>
#include <cmath>

namespace d2 {
    namespace def {
        //! \brief the Stats class for classification problems
        template<size_t n_class>
        struct ClassificationStats : public Stats<_D2_CLTYPE> {
            // member variables
            std::array<double, n_class + 1> histogram;

            ClassificationStats() : histogram({}) {
            }

            ClassificationStats(const ClassificationStats<n_class> &that) : histogram(that.histogram) {
            }

            using LabelType = Stats<_D2_CLTYPE>::LabelType;

            inline LabelType getLabel() const override {
                return std::max_element(histogram.begin(), histogram.end() - 1) - histogram.begin();
            }

            inline void updateLeft(LabelType y) override {
                histogram[y]++;
                histogram.back()++;
            }

            inline void updateRight(LabelType y) override {
                histogram[y]--;
                histogram.back()--;
            }

            inline bool stop() const override {
                // todo
                return false;
            }

            template<class criterion>
            inline static real_t
            goodness_score(const ClassificationStats<n_class> left, const ClassificationStats<n_class> right) {
                return
                        (criterion::op(left) * left.histogram.back() + criterion::op(right) * right.histogram.back())
                        / (left.histogram.back() + right.histogram.back());
            }

        };

        /*! \brief gini function used in make splits
         */
        struct gini {
            template<size_t n_class>
            static inline real_t op(const ClassificationStats<n_class> &y_stats) {
                auto &proportion = y_stats.histogram;
                real_t total_weight_sq;
                total_weight_sq = proportion.back() * proportion.back();
                //if (total_weight_sq <= 0) return 1.;

                real_t gini = total_weight_sq;
                for (size_t i = 0; i < n_class; ++i)
                    gini -= proportion[i] * proportion[i];
                gini /= total_weight_sq;
                return gini;
            }

            template<size_t n_class>
            static inline real_t unnormalized_op(const ClassificationStats<n_class> &y_stats) {
                return op(y_stats) * y_stats.histogram.back();
            }
            
            static inline real_t loss(const real_t &x) { return 1 - x; }
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
                for (size_t i = 0; i < n_class; ++i) {
                    if (proportion[i] > 0) {
                        real_t p = proportion[i] / total_weight;
                        entropy -= log(p) * p;
                    }
                }

                return entropy;
            }

            template<size_t n_class>
            static inline real_t unnormalized_op(const ClassificationStats<n_class> &y_stats) {
                return op(y_stats) * y_stats.histogram.back();
            }
            
            static inline real_t loss(const real_t &x) { return -log(x); }
        };


    }

    namespace internal {

        template<size_t n_class>
        struct SortedSample<def::ClassificationStats<n_class> > {
            real_t x;
            _D2_CLTYPE y;
            //      real_t weight;
            size_t index;

            //SortedSample *next;
            inline static bool cmp(const SortedSample<def::ClassificationStats<n_class> > &a,
                                   const SortedSample<def::ClassificationStats<n_class> > &b) {
                return a.x < b.x;
            }
        };
    }
}

#endif /* _CLASSIFICATION_H_ */
