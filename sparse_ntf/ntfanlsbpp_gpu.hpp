#ifndef NTF_NTFANLSBPP_GPU_HPP_
#define NTF_NTFANLSBPP_GPU_HPP_

#include "sparse_ntf/auntf_gpu.hpp"
#include "cuda_utils.h"
#include "anlsbpp.h"

#define CHUNK_SIZE 2000

namespace planc {
template <class T>
class NTFANLSBPP_GPU : public AUNTF_GPU <T> {
  private:
    // MAT_GPU** aux_factors_gpu;
    int admm_iter;
    double tolerance;

  protected:
    void update_gpu(const int mode, const MAT_GPU * gram, MAT_GPU ** factors, MAT_GPU * o_mttkrp) {
      int nhrs = o_mttkrp->n_cols;
      // printf("fm size: %d %d\n", factors[mode]->n_rows, factors[mode]->n_cols);

      mat_transpose_inplace(factors[mode]);
      // printf("fm size: %d %d\n", factors[mode]->n_rows, factors[mode]->n_cols);
  
      int num_chunks = nhrs / CHUNK_SIZE;
      if (num_chunks * CHUNK_SIZE < nhrs) num_chunks++;

      // later might need to move it so that it can use cudaStreams
      for (int i = 0; i < num_chunks; ++i) {
        // printf("i: %d\n", i);
        int col_start_idx = i * CHUNK_SIZE;
        int col_end_idx = (i + 1) * CHUNK_SIZE - 1;
        if (col_end_idx > nhrs - 1) col_end_idx = nhrs -1;

        anlsbpp_update(factors[mode]->cols(col_start_idx, col_end_idx), o_mttkrp->cols(col_start_idx, col_end_idx), gram);
      }

      mat_transpose_inplace(factors[mode]);
      
      // exit(0);
    }


    /**
     * @brief Create ANLSBPP subproblems on host given Gram matrix and RHS (submatrix of mttkrp output)
     * 
     * This methods takes in operands from GPU and moves it back to host and creates subProblem
     * 
     * @param a
     * @param b
     * @return The subproblem
    */
    void generate_subproblem_host() {

    }

  public:
    NTFANLSBPP_GPU(const T &i_tensor, const int i_k, algotype i_algo)
      : AUNTF_GPU<T>(i_tensor, i_k, i_algo) {
        // set up dual variable that only exists in GPU
        // aux_factors_gpu = new MAT_GPU*[i_k];

        for (int m = 0; m < i_tensor.modes(); ++m) {
          int n_rows = i_tensor.dimensions()[m];
          int n_cols = i_k;
          // printf("aux_factors_gpu[%d]: (%dx%d)\n", m, n_rows, n_cols);
          // aux_factors_gpu[m] = init_mat_gpu(n_rows, n_cols);
        }
        admm_iter = 5;
        tolerance = 0.01;
      }
};  // class NTFANLSBPP_GPU
}  // namespace planc

#endif  // NTF_NTFANLSBPP_GPU_HPP_
