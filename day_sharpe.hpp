#ifndef _DAY_SHARPE_H_
#define _DAY_SHARPE_H_


#include "traits.hpp"

#include <cmath>
#include <vector>

namespace d2 {
  namespace def {

    template <size_t days>
    real_t _sharpe_helper (const std::array<real_t, days> &fr) {
      real_t m1 = 0, m2 = 1E-10;
      for (size_t i = 0; i< days; ++i) {
	m1 += fr[i];
	m2 += fr[i] * fr[i];
      }

      m1 = m1 / days;
      m2 = m2 / days;

      return m1 / sqrt(m2 - m1*m1);
    }
    
    typedef std::pair<real_t, size_t> reward_date_pair;
    
    template <size_t days>
    struct DaySharpeStats: Stats<reward_date_pair> {
      size_t count;
      std::array<real_t, days> forward_return;

      DaySharpeStats() {}
      
      DaySharpeStats(const DaySharpeStats<days> &that): count(that.count), forward_return(forward_return) {}

      DaySharpeStats<days> & operator=(const DaySharpeStats<days> &that) {
	count = that.count;
	forward_return = that.forward_return;
	return *this;
      }

      using Stats<reward_date_pair>::LabelType;
      
      inline LabelType get_label() const override {
	return std::make_pair(std::max(_sharpe_helper(this->forward_return), 0), 
			      std::numeric_limits<std::size_t>::max());
      }

      inline void update_left(LabelType y) override {
	forward_return[y.second] += y.first;
	count ++;
      }

      inline void update_right(LabelType y) override {
	forward_return[y.second] -= y.first;
	count --;
      }
      
      inline bool stop() const override {
	// todo
	return false;
      }

      template <class criterion>
      inline static real_t goodness_score(const DaySharpeStats<days> left, const DaySharpeStats<days> right) {
	return std::max(criterion::op(left), criterion::op(right));
      }
    };

    struct sharpe {
      template <size_t days>
      static inline real_t op(const DaySharpeStats<days> &y_stats) {
	return _sharpe_helper(y_stats->forward_return);
      }
    };
  }

  namespace internal {
    template <class YStats>
    struct sorted_sample;

    template <size_t days>
    struct sorted_sample<def::DaySharpeStats<days> > {
      real_t x;
      def::reward_date_pair y;
      //      real_t weight;
      size_t index;
      //sorted_sample *next;
      inline static bool cmp(const sorted_sample<def::DaySharpeStats<days> > &a, 
			     const sorted_sample<def::DaySharpeStats<days> > &b) {
	return a.x < b.x;
      }
    };
  }
}


#endif /* _DAY_SHARPE_H_ */
