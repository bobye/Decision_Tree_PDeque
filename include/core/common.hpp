#ifndef _COMMON_H_
#define _COMMON_H_

#include <string>
#include <cstring>

namespace d2 {

#define __IN__
#define __OUT__ 
#define __IN_OUT__


  /*! \brief the float point that will be used to 
   * store real numbers. */
#ifdef _D2_DOUBLE
  typedef double real_t;
#elif defined _D2_SINGLE
  typedef float real_t;
#endif


#ifdef  _D2_DOUBLE
#define _D2_SCALAR          double
#define _D2_FUNC(x)         _d ## x
#define _D2_CBLAS_FUNC(x)   cblas_d ## x
  //#define _D2_LAPACKE_FUNC(x) d ## x
#elif defined  _D2_SINGLE
#define _D2_SCALAR          float
#define _D2_FUNC(x)         _s ## x
#define _D2_CBLAS_FUNC(x)   cblas_s ## x
  //#define _D2_LAPACKE_FUNC(x) s ## x
#endif

  /*! \brief the unsigned integer type that will
   * be used to store index 
   */
  typedef unsigned index_t;

  /*! \brief the header string of log printing subject to each processer
   */
#ifdef RABIT_RABIT_H_
  inline const std::string getLogHeader() 
  {return std::string("@d2suite(") + std::to_string(rabit::GetRank()) + ")";}
#else
  inline const std::string getLogHeader() 
  {return std::string("@d2suite");}
#endif


}

/*! \brief the ad-hoc solution to barrier() api */
#ifdef RABIT_RABIT_H_
namespace rabit {
  inline void Barrier(){ static float a = 1; rabit::Broadcast(&a, sizeof(float), 0); }
}
#endif

#endif /* _COMMON_H_ */
