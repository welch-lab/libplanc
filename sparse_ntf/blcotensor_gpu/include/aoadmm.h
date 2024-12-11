#ifndef AOADMM_GPU_H_
#define AOADMM_GPU_H_

#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "vector_ops.h"
#include "cuda_utils.h"
#include "matrix_ops.h"
#include "blco.h"
#include <iostream>

void admm_gpu_kernel(
  double * gram_vals, 
  double rho, 
  double * primal_vals, 
  double * mttkrp_vals, // necessay to compute 
  double * prev_fac_vals, // necessary to store previous vals
  double * dual_vals, // dual variables -- same size as primals vals
  double * tilde_h_vals, // H tilde
  int start_row, int end_row, int rank, int admm_iter, double tolerance, 
  double * admm_condition_vals, 
  int stream_id,
  cudaStream_t stream);
void aoadmm_update(MAT_GPU * fm, MAT_GPU * aux_fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram, int admm_iter, double tolerance);
void aoadmm_blocked_update(MAT_GPU * fm, MAT_GPU * aux_fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram, int block_size, int admm_iter, double tolerance, int num_streams);

void compute_t_H(cudaStream_t stream, double * t_fm_vals, double * mttkrp_vals, double * fm_vals, double * dual_vals, double rho, size_t n);
void compute_U(cudaStream_t stream, double * dual_vals, double * fm_vals, double * t_fm_vals, size_t n);
void compute_diff_squared(cudaStream_t stream, double * fm_vals, double * prev_vals, double * t_fm_vals, size_t n);
void compute_fm_with_projection(cudaStream_t stream, double * fm_vals, double * t_fm_vals, double * dual_vals, size_t n);

#endif // AOADMM_GPU_H_