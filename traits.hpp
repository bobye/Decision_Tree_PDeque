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
      virtual inline void update_left(LabelType y) = 0;
      virtual inline void update_right(LabelType y) = 0;
      virtual inline LabelType get_label() const = 0;
      virtual inline bool stop() const = 0;
    };

  }

  namespace internal {
    /*! \brief data structure for additional linked list on presort samples
     */
    template <class YStats> struct sorted_sample;

    template <size_t dim, class YStats> class _DTLeaf;
    template <size_t dim, class YStats> class _DTBranch;

  }
}
#endif /* _TRAITS_H_ */
