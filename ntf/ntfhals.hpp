/* Copyright Ramakrishnan Kannan 2018 */

#ifndef NTF_NTFHALS_HPP_
#define NTF_NTFHALS_HPP_

#include "ntf/auntf.hpp"
#include "common/utils.hpp"

namespace planc {

template <class T>
class NTFHALS : public AUNTF <T> {
 protected:
  MAT update(const int mode) {
    MAT H(this->m_ncp_factors.factor(mode));
    // iterate over all columns of H
    for (int i = 0; i < this->m_ncp_factors.rank(); i++) {
      VEC updHi =
          H.col(i) + ((this->ncp_mttkrp_t[mode].row(i)).t() -
                      H * this->gram_without_one.col(i));
      fixNumericalError<VEC>(&updHi, EPSILON_1EMINUS16, EPSILON_1EMINUS16);
      double normHi = arma::norm(updHi, 2);
      normHi *= normHi;
      double globalnormHi = normHi;
      if (globalnormHi > 0) {
        H.col(i) = updHi;
      }
    }
    return H.t();
  }

 public:
  NTFHALS(const T &i_tensor, const int i_k, algotype i_algo)
      : AUNTF<T>(i_tensor, i_k, i_algo) {}
};  // class NTFHALS

}  // namespace planc

#endif  // NTF_NTFHALS_HPP_
