#ifndef NTF_NTFMU_GPU_HPP_
#define NTF_NTFMU_GPU_HPP_

#include "sparse_ntf/auntf_gpu.hpp"
#include "mu.h"

namespace planc {
template <class T>
class NTFMU_GPU : public AUNTF_GPU <T> {
  protected:
    void update_gpu(const int mode, const MAT_GPU * gram, MAT_GPU ** factors, MAT_GPU * o_mttkrp) {
      // mu_update(factors[mode], o_mttkrp, gram);
      mu_update_opt(factors[mode], o_mttkrp, gram);
    }

  public:
    NTFMU_GPU(const T &i_tensor, const int i_k, algotype i_algo)
      : AUNTF_GPU<T>(i_tensor, i_k, i_algo) {}
};  // class NTFMU_GPU
}  // namespace planc

#endif  // NTF_NTFMU_GPU_HPP_
