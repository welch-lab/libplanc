#include "mu.h"
#include <cmath>

__global__ void update_fm(double * h, double * o_mttkrp_vals, double * temp, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    h[idx] = h[idx] * o_mttkrp_vals[idx] / temp[idx];
  }
}

__global__ void update_fm_opt(double * h, double * o_mttkrp_vals, double * temp, double epsilon, size_t size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    h[idx] = h[idx] * o_mttkrp_vals[idx] / (temp[idx] + epsilon);
  }
}

/**
 * Optimized implementation of MU update
 * Simply reduce redundant kernel calls and memory operations
 * Non-blocked version.. might need blocked version if STF breaks for R > 128? 
*/
void mu_update_opt(MAT_GPU * fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram) {
  int m = fm->n_rows;
  int n = gram->n_cols;
  int k = fm->n_cols;

  double * d_temp;
  cudaStream_t stream1;

  cudaStreamCreate(&stream1); // mat mul 

  check_cuda(
    cudaMallocAsync((void**)&d_temp, m * n * sizeof(double), stream1), "malloc result matrix");

    // d_temp = H * gram + EPSILON
    cublasHandle_t handle1;
    cublasCreate(&handle1);

    cublasSetStream(handle1, stream1);

    double alpha = 1.0;
    double beta = 0.0;

    // double * dummy
    // temp = H * gram
    // temp = H * gram
    check_cublas(cublasDgemm(handle1, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, fm->vals, m, gram->vals, n, &beta, d_temp, m), "H * this->gram_without_one");

    int num_elements = m * k;
    int num_blocks = (num_elements + TILE_SIZE - 1) / TILE_SIZE;

    update_fm_opt<<<num_blocks, TILE_SIZE, 0, stream1>>>(fm->vals, o_mttkrp_gpu->vals, d_temp, EPSILON, m * k);
    cudaStreamSynchronize(stream1);
    cudaFreeAsync(d_temp, stream1);
    cublasDestroy(handle1);
    cudaStreamDestroy(stream1);
}

void mu_update(MAT_GPU * fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram) {
  // need matrix multiplicatin -- fm * gram
  int m = fm->n_rows; // rows in A and C
  int n = gram->n_cols; // cols in B and C
  int k = fm->n_cols; // cols in A and C

  // create temp matrix
  double * d_temp, * d_mttkrp_t;
  check_cuda(
    cudaMalloc((void**)&d_temp, m * n * sizeof(double)),
    "malloc result matrix"
  );
  check_cuda(
    cudaMalloc((void**)&d_mttkrp_t, m * n * sizeof(double)),
    "malloc mttkrp_t matrix"
  );

  value_dfill(d_temp, m * n, EPSILON);
  check_cuda(cudaDeviceSynchronize(), "value fill -- epsilon");

  mat_mat_mul(fm->vals, gram->vals, d_temp, m, n, k, 1.0, 1.0); // H * gram + EPSILON

  // Update factor matrix
  int num_elements = m * k;
  int num_blocks = (num_elements + TILE_SIZE - 1) / TILE_SIZE;

  // REFACTOR -> can use cublas geam
  // __mat_transpose<<<num_blocks, TILE_SIZE>>>(d_mttkrp_t, o_mttkrp_gpu->vals, k, m); // 
  
  check_cuda(cudaDeviceSynchronize(), "sync after normaliztation");
  // check_cuda(cudaMemcpy(o_mttkrp_gpu->vals, d_mttkrp_t, m * k * sizeof(double), cudaMemcpyDeviceToDevice), "copy o_mttkrp_t to o_mttkrp");

  update_fm<<<num_blocks, TILE_SIZE>>>(fm->vals, o_mttkrp_gpu->vals, d_temp, num_elements);
  cudaFree(d_temp);
  cudaFree(d_mttkrp_t);
}
