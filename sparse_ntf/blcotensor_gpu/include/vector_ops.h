#pragma once

#include "config.h"
#include "cuda_runtime.h"


// template <typename T>
// __global__ void value_fill_kernel(T * x, unsigned int n, T val) {
//   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < n) x[index] = val;
// }

// template <typename T>
// void value_fill(T * x, unsigned int n, T val) {
//   value_fill_kernel<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(x, n, val);
// //   check_cuda(cudaGetLastError(), "value_fill_kernel launch");
// }

// Basic elemeent-wise vector operations
void ivec_add(const int * v1, const int * v2, int * v3, size_t size);
void ivec_sub(const int * v1, const int  * v2, int * v3, size_t size);
void ivec_mult(const int * v1, const int * v2, int * v3, size_t size);
void ivec_scale(const int * v1, const int * v2, int * v3, size_t size);

void dvec_add(const double * v1, const double * v2, double * v3, size_t size);
void dvec_add_async(cudaStream_t stream, const double * v1, const double * v2, double * v3, size_t size);

void dvec_sub(const double * v1, const double  * v2, double * v3, size_t size);
void dvec_sub_async(const double * v1, const double * v2, double * v3, size_t size, cudaStream_t stream);
void dvec_mult(const double * v1, const double * v2, double * v3, size_t size);
void dvec_scale(const double * v1, const double * v2, double * v3, size_t size);

void uvec_add(const unsigned int * v1, const unsigned int * v2, unsigned int * v3, size_t size);
void uvec_sub(const unsigned int * v1, const unsigned int  * v2, unsigned int * v3, size_t size);
void uvec_mult(const unsigned int * v1, const unsigned int * v2, unsigned int * v3, size_t size);
void uvec_scale(const unsigned int * v1, unsigned int * v2, unsigned int factor, size_t size);

void uvec_copy(const unsigned int * vi, unsigned int * vo, size_t size);
void uvec_copy_idx_based(const unsigned int * vi, unsigned int * vo, const unsigned int * ind_vec, size_t size);
void uvec_sub_idx_based(unsigned int * vi, const unsigned int * ind_vec, size_t size, unsigned int val);

void dvec_apply_mask(const double * vi, double * vo, const unsigned int * mask_vec, size_t size);
void dvec_update_only_mask(const double * vi, double * vo, const unsigned int * mask_vec, size_t size);

void fixAbsNumericalError_gpu(double * vals, double threshold, double value, size_t size);