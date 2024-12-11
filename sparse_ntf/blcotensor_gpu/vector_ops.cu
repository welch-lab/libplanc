#include <vector_ops.h>

__global__ void ivec_add_kernel(const int * v1, const int * v2, int * v3, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) v3[i] = v1[i] + v2[i];
}

__global__ void ivec_sub_kernel(const int * v1, const int * v2, int * v3, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) v3[i] = v1[i] - v2[i];
}

__global__ void ivec_mult_kernel(const int * v1, const int * v2, int * v3, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) v3[i] = v1[i] * v2[i];
}

__global__ void ivec_scale_kernel(const int * vi, int * vo, int factor, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) vo[i] = factor * vi[i];
}

void ivec_add(const int * v1, const int * v2, int * v3, size_t size) {
  ivec_add_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, v3, size);
}
void ivec_sub(const int * v1, const int * v2, int * v3, size_t size) {
  ivec_sub_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, v3, size);
}
void ivec_mult(const int * v1, const int * v2, int * v3, size_t size) {
  ivec_mult_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, v3, size);
}
void ivec_scale(const int * v1, int * v2, int factor, size_t size) {
  ivec_scale_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, factor, size);
}

__global__ void dvec_add_kernel(const double * v1, const double * v2, double * v3, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) v3[i] = v1[i] + v2[i];
}

__global__ void dvec_sub_kernel(const double * v1, const double * v2, double * v3, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) v3[i] = v1[i] - v2[i];
}

__global__ void dvec_mult_kernel(const double * v1, const double * v2, double * v3, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) v3[i] = v1[i] * v2[i];
}

__global__ void dvec_scale_kernel(const double * vi, double * vo, double factor, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) vo[i] = factor * vi[i];
}

void dvec_add(const double * v1, const double * v2, double * v3, size_t size) {
  dvec_add_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, v3, size);
}

void dvec_add_async(cudaStream_t stream, const double * v1, const double * v2, double * v3, size_t size) {
  dvec_add_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(v1, v2, v3, size);

}
void dvec_sub(const double * v1, const double * v2, double * v3, size_t size) {
  dvec_sub_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, v3, size);
}

void dvec_sub_async(const double * v1, const double * v2, double * v3, size_t size, cudaStream_t stream) {
  dvec_sub_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(v1, v2, v3, size);
}

void dvec_mult(const double * v1, const double * v2, double * v3, size_t size) {
  dvec_mult_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, v3, size);
}
void dvec_scale(const double * v1, double * v2, double factor, size_t size) {
  dvec_scale_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, factor, size);
}

__global__ void uvec_add_kernel(const unsigned int * v1, const unsigned int * v2, unsigned int * v3, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) v3[i] = v1[i] + v2[i];
}

__global__ void uvec_sub_kernel(const unsigned int * v1, const unsigned int * v2, unsigned int * v3, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) v3[i] = (v1[i] > v2[i]) * (v1[i] - v2[i]);
}

__global__ void uvec_mult_kernel(const unsigned int * v1, const unsigned int * v2, unsigned int * v3, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) v3[i] = v1[i] * v2[i];
}

__global__ void uvec_scale_kernel(const unsigned int * vi, unsigned int * vo, unsigned int factor, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) vo[i] = factor * vi[i];
}

__global__ void uvec_copy_kernel(const unsigned int * vi, unsigned int * vo, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) vo[i] = vi[i];
}

__global__ void uvec_copy_idx_based_kernel(const unsigned int * vi, unsigned int * vo, const unsigned int * ind_vec, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) vo[i] = (ind_vec[i] == 1) * vi[i] + (ind_vec[i] != 1) * vo[i];
}

__global__ void uvec_sub_idx_based_kernel(unsigned int * vi, const unsigned int * ind_vec, size_t size, unsigned int val) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) vi[i] = (ind_vec[i] == 1) * (vi[i] - val) + (ind_vec[i] != 1) * vi[i];
}

__global__ void dvec_apply_mask_kernel(const double * vi, double * vo, const unsigned int * mask_vec, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) vo[i] = (mask_vec[i] == 1) * vi[i];
}

/**
 * @brief updates values where mask_vec indicates 1, otherwise set as 0
*/
__global__ void dvec_update_only_mask_kernel(const double * vi, double * vo, const unsigned int * mask_vec, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) vo[i] = (mask_vec[i] == 1) * vi[i] + (mask_vec[i] != 1) * 0.0f;
}

/**
 * If vals[i] < threshold ? value : vals[i]; 
*/
__global__
void fixAbsNumericalError_gpu_kernel(double * vals, double threshold, double value, size_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) vals[i] = (fabs(vals[i]) < threshold) * value + (fabs(vals[i]) >= threshold) * vals[i];
}


void uvec_add(const unsigned int * v1, const unsigned int * v2, unsigned int * v3, size_t size) {
  uvec_add_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, v3, size);
}
void uvec_sub(const unsigned int * v1, const unsigned int * v2, unsigned int * v3, size_t size) {
  uvec_sub_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, v3, size);
}
void uvec_mult(const unsigned int * v1, const unsigned int * v2, unsigned int * v3, size_t size) {
  uvec_mult_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, v3, size);
}
void uvec_scale(const unsigned int * v1, unsigned int * v2, unsigned int factor, size_t size) {
  uvec_scale_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(v1, v2, factor, size);
}

void uvec_copy(const unsigned int * vi, unsigned int * vo, size_t size) {
  uvec_copy_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(vi, vo, size);
}

void uvec_copy_idx_based(const unsigned int * vi, unsigned int * vo, const unsigned int * ind_vec, size_t size) {
  uvec_copy_idx_based_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(vi, vo, ind_vec, size);
}

void uvec_sub_idx_based(unsigned int * vi, const unsigned int * ind_vec, size_t size, unsigned int val) {
  uvec_sub_idx_based_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(vi, ind_vec, size, val);
}

void dvec_apply_mask(const double * vi, double * vo, const unsigned int * mask_vec, size_t size) {
  dvec_apply_mask_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(vi, vo, mask_vec, size);
}

void dvec_update_only_mask(const double * vi, double * vo, const unsigned int * mask_vec, size_t size) {
  dvec_update_only_mask_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(vi, vo, mask_vec, size);
}


void fixAbsNumericalError_gpu(double * vals, double threshold, double value, size_t size) {
  fixAbsNumericalError_gpu_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE>>>(vals, threshold, value, size);
}