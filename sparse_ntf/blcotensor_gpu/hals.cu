#include "hals.h"

void hals_update(MAT_GPU * fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram) {
  _FType *fm_times_gram_col;

  int m = fm->n_rows;
  int n = fm->n_cols;

  check_cuda(
    cudaMalloc((void**)&fm_times_gram_col, m * sizeof(_FType)), "cudaMalloc fm_times_gram_col"
  );

  int num_elements = m * n;
  int num_blocks = (num_elements + BLOCK_SIZE - 1 )/ BLOCK_SIZE;

  int rank = n;
  for (int r = 0; r < rank; ++r) {
    // get column from gram_without_one
    int col_idx = m * r;
    mat_vec_mul(fm->vals, &gram->vals[rank * r], fm_times_gram_col, m, n, 1.0, 0.0);

    num_elements = m;
    num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dvec_sub(fm->vals+col_idx, fm_times_gram_col, fm->vals+col_idx, m);
    dvec_add(fm->vals+col_idx, o_mttkrp_gpu->vals+col_idx, fm->vals+col_idx, m);
    __apply_threshold<<<num_blocks, BLOCK_SIZE>>>(fm->vals+col_idx, m, 1e-16, 1e-16);
  }

  cudaFree(fm_times_gram_col);
}