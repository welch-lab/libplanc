/* Copyright Ramakrishnan Kannan 2018 */

#ifndef NTF_NTFMU_HPP_
#define NTF_NTFMU_HPP_

#include "ntf/auntf.hpp"

namespace planc {
template <class T>
class NTFMU : public AUNTF <T> {
 protected:
  MAT update(const int mode) {
    MAT H(this->m_ncp_factors.factor(mode));
    MAT temp = H * this->gram_without_one + EPSILON;
    H = (H % this->ncp_mttkrp_t[mode].t()) / temp;
    return H.t();
  }

 public:
  NTFMU(const T &i_tensor, const int i_k, algotype i_algo)
      : AUNTF<T>(i_tensor, i_k, i_algo) {}
};  // class NTFMU
}  // namespace planc

#endif  // NTF_NTFMU_HPP_
