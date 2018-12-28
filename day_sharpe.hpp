#ifndef _DAY_SHARPE_H_
#define _DAY_SHARPE_H_


#include "traits.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace d2 {
  namespace def {

    struct sharpe_stats {
      real_t mean;
      real_t std;
      real_t sharpe;
    };

    template <size_t days>
    sharpe_stats _sharpe_helper (const std::array<real_t, days> &fr) {
      real_t m1 = 0, m2 = 0;
      for (size_t i = 0; i< days; ++i) {
	m1 += fr[i];
	m2 += fr[i] * fr[i];
      }

      m1 = m1 / days;
      m2 = m2 / days;
      
      return  {m1, m2, static_cast<real_t>(- m1 / (sqrt(m2 - m1*m1) + 1E-10))};
    }
    
    struct reward_date_pair {
      real_t reward;
      size_t date;
      bool operator==(const reward_date_pair &that) {
	return reward == that.reward;
      }
    };

    std::ostream & operator << (std::ostream &out, const reward_date_pair &p) {
      out << p.reward;
      return out;
    }
    
    template <size_t days>
    struct DaySharpeStats: Stats<reward_date_pair> {
      size_t count;
      std::array<real_t, days> forward_return;

      DaySharpeStats(): count(0), forward_return({}) {}
      
      DaySharpeStats(const DaySharpeStats<days> &that): count(that.count), forward_return(that.forward_return) {}

      DaySharpeStats<days> & operator=(const DaySharpeStats<days> &that) {
	count = that.count;
	forward_return = that.forward_return;
	return *this;
      }

      using Stats<reward_date_pair>::LabelType;
      
      inline LabelType get_label() const override {
	return {std::min(_sharpe_helper(this->forward_return).sharpe, (real_t) 0.), std::numeric_limits<std::size_t>::max()};
      }

      inline void update_left(LabelType y) override {
	forward_return[y.date] += y.reward;
	count ++;
      }

      inline void update_right(LabelType y) override {
	forward_return[y.date] -= y.reward;
	count --;
      }
      
      inline bool stop() const override {
	// todo
	return false;
      }

      template <class criterion>
      inline static real_t goodness_score(const DaySharpeStats<days> left, const DaySharpeStats<days> right) {
	return std::min(criterion::op(left), criterion::op(right));
      }
    };

    struct sharpe {
      template <size_t days>
      static inline real_t op(const DaySharpeStats<days> &y_stats) {
	return _sharpe_helper(y_stats.forward_return).sharpe;
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
