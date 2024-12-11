#ifndef TF_UCP_HPP_
#define TF_UCP_HPP_

#include "ntf/auntf.hpp"

namespace planc {
template <class T>
class TFUCP : public AUNTF <T> {
 protected:
  MAT update(const int mode) {
    MAT L = arma::chol(this->gram_without_one, "lower");
    MAT H = arma::solve(arma::trimatl(L), this->ncp_mttkrp_t[mode]);
    return H;
  }

 public:
  TFUCP(const T &i_tensor, const int i_k, algotype i_algo)
      : AUNTF<T>(i_tensor, i_k, i_algo) {}
};  // class TFUCP
}  // namespace planc

#endif