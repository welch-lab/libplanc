#ifndef NTF_NTFHALS_GPU_HPP_
#define NTF_NTFHALS_GPU_HPP_

#include "sparse_ntf/auntf_gpu.hpp"
#include "hals.h"

namespace planc {
template <class T>
class NTFHALS_GPU : public AUNTF_GPU <T> {
  protected:
    void update_gpu(const int mode, const MAT_GPU * gram, MAT_GPU ** factors, MAT_GPU * o_mttkrp) {
      hals_update(factors[mode], o_mttkrp, gram);
    }

  public:
    NTFHALS_GPU(const T &i_tensor, const int i_k, algotype i_algo)
      : AUNTF_GPU<T>(i_tensor, i_k, i_algo) {}
};  // class NTFHALS_GPU
}  // namespace planc

#endif  // NTF_NTFHALS_GPU_HPP_
