#ifndef _TRAITS_H_
#define _TRAITS_H_

namespace d2 {

    namespace def {

        //! \brief the generic trait class for node statistics
        //! a Stats class should contain:
        //    (1) the sufficient statistics for supporting the calculation of the split criteria at each node
        //    (2) the update rules of statistics, O(1) time, when a cutoff threshold at each node splitting is moving upwards
        //    (3) the stop criterion of node splitting specific to the sufficient statistics
        template<class LT>
        struct Stats {
            typedef LT LabelType;

            virtual inline void updateLeft(LabelType y) = 0;

            virtual inline void updateRight(LabelType y) = 0;

            virtual inline LabelType getLabel() const = 0;

            virtual inline bool stop() const = 0;
        };

        template<class YStats, typename criterion>
        struct finalize {
            static void op(YStats &y_stats) {}
        };

        // inherit statistics from parent
        template<class YStats, typename criterion>
        struct prepare {
            static YStats left_op(const YStats &y_stats) { return YStats(); }

            static YStats right_op(const YStats &y_stats) { return y_stats; }
        };
    }

    namespace internal {
        /*! \brief data structure for additional linked list on presort samples
         */
        template<class YStats>
        struct SortedSample;

        template<size_t dim, class YStats>
        class DTLeaf;

        template<size_t dim, class YStats>
        class DTBranch;

    }
}
#endif /* _TRAITS_H_ */
