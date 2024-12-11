#include "matrix_ops.h"

static const double one = 1.0;
static const double zero = 0.0;


// Takes the reciprocal of each vector entry, sets to zero if smaller than tol
__global__ void reciprocal_vector_kernel(double * v, unsigned int n, double tol) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    v[index] = (fabs(v[index]) > tol) ? 1.0 / v[index] : 0;
  }
}

// Use cusolverDnDgesvdj_bufferSize to calculate needed buffer size
void pseudoinverse_gpu(cusolverDnHandle_t cusolverHandle, cublasHandle_t cublasHandle,
    cudaStream_t stream, double* A, unsigned int n, double* work, unsigned int lwork, int* info, gesvdjInfo_t gesvd_info) {

    // I tried Cholesky / QR / LU factorization
    // They scale poorly to larger matrices compared to svd + gemm

    double* U = work;
    double* V = U + n * n;
    double* S = V + n * n;
    work = S + n;
    lwork -= (2 * n * n + n);

    // Gen SVD
    check_cublas(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
    check_cusolver(cusolverDnDgesvdj(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 
        0, n, n, A, n, S, U, n, V, n, work, lwork, info, gesvd_info), "cusolverDnDgesvdj");

    check_cuda(cudaStreamSynchronize(stream), "cusolverDngesvdj execute");

    // Multiply U by S^-1 (scale rows of U by reciprocal of S);
    unsigned int blocks = n / BLOCK_SIZE + 1;
    double s = 0; // Get largest singular value
    check_cuda(cudaMemcpyAsync(&s, S, sizeof(double), cudaMemcpyDeviceToHost, stream), "memcpy");
    s = n * (nextafter(s, s + 1) - s);
    reciprocal_vector_kernel <<<blocks, BLOCK_SIZE, 0, stream>>>(S, n, s);
    //check_cuda(cudaGetLastError(), "reciprocal_vector launch");
    check_cuda(cudaStreamSynchronize(stream), "reciprocal_vector execute");

    // Multiply U * S
    for (unsigned int i = 0; i < n; i++) {
        cublasSetStream(cublasHandle, stream);
        #ifdef USE_32BIT_TYPE
            check_cublas(cublasSscal(cublasHandle, n, S + i, U + i * n, 1), "cublasSscal");
        #else 
            check_cublas(cublasDscal(cublasHandle, n, S + i, U + i * n, 1), "cublasDscal");
        #endif
    }
    check_cuda(cudaStreamSynchronize(stream), "cublasDscal execute");

    // Multiply U by V (we multiply V by U^T to convert col to row major)
    check_cublas(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    cublasSetStream(cublasHandle, stream);

    check_cublas(cublasDgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        n, n, n,
        &one, 
        V, n,
        U, n,
        &zero, 
        A, n), "cublasDgemm"
    );
    check_cuda(cudaStreamSynchronize(stream), "cublasDgemm execute");
}

__global__ void apply_mask_to_gram_matrix_kernel(double * vals, unsigned int n_rows, unsigned int n_cols, unsigned int * idx_to_mask) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n_rows * n_cols) {
    int row_idx = index % n_rows;
    int col_idx = index / n_rows;
    vals[index] = (idx_to_mask[row_idx] == 1 && idx_to_mask[col_idx] == 1) * vals[index];
  }
}

void apply_mask_to_gram_matrix(cudaStream_t stream, double * masked_gram_vals, int m, unsigned int * idx_to_mask) {
  apply_mask_to_gram_matrix_kernel<<<m * m / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(masked_gram_vals, m, m, idx_to_mask);
}

__global__ void apply_mask_to_matrix_kernel(double * vo, unsigned int * vi, size_t size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    vo[index] = (vi[index] == 1) * vo[index];
  }
}

void apply_mask_to_matrix(cudaStream_t stream, double * vo, unsigned int * vi, size_t size) {
  apply_mask_to_matrix_kernel<<<size / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(vo, vi, size);
}

/**
 * Computes the inverse of a submatrix using the reverse rank-1 update (downdate)
 * @param[in] cublasHandle cublasHandle context
 * @param[in] d_A // device pointer to input matrix A (submatrix)
 * @param[in] rows
 * @param[in] cols
 * @param[in] idx_to_drop // idx to drop
 * @param[out] d_A_inv // device pointer to the inverse of submatrix of A
*/
void sub_matrix_inverse(cublasHandle_t cublasHandle, double * d_A, int rows, int cols, int idx_to_drop, double * d_A_inv) {
  // check_cuda(cudaMemcpy(), "memcpy to A_inv");
  // for (int r = 0; r < num_idx_to_drop; ++r) {
  //   downdate_rank_matrix(cublasHandle, )
  // }

}

void downdate_rank_matrix(cudaStream_t stream, cublasHandle_t cublasHandle, double * d_A, int rows, int cols, int idx_to_drop, double * h_x_diag) {
    // define x, which is the same as y

  // vector that points to vector that will drop
  double * x = d_A + idx_to_drop * rows;
  double * y;

  // check_cuda(cudaMallocAsync(&x, rows * sizeof(double), stream), "cudaMallocAsync x");
  // check_cuda(cudaMemcpyAsync(x , d_A + idx_to_drop * rows, rows * sizeof(double), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy");
  // check_cuda(cudaMemcpyAsync(h_x_diag, &x[idx_to_drop], sizeof(double), cudaMemcpyDeviceToHost, stream), "cudaMemcpy scalar");
  y = x;
  // double alpha = -1.0 / *h_x_diag;
  double alpha = -1.0;

  check_cublas(cublasDger(cublasHandle, rows, cols, &alpha, x, 1, y, 1, d_A, rows), "submatrix inverse op");
}

void solveDownDate(cudaStream_t stream, cublasHandle_t cublasHandle, 
    double * masked_gram_vals, size_t R, const unsigned int * mask, int num_masked_vars, double * masked_b_vals, double * h_x_diag) {

      cublasSetStream(cublasHandle, stream);
      for (int r = 0; r < R; ++r) {
        if ((mask[r] & 1) == 0) {
          // if entry is 0
          // printf("dropping idx: %d\n", r);
          downdate_rank_matrix(stream, cublasHandle, masked_gram_vals, R, R, r, h_x_diag);
        }
      }

      // do dgemm mat_inv * masked_b_vals
      double one = 1.0;
      double zero = 0.0;
      check_cublas(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, R, 
        1, R, &one, masked_gram_vals, R, masked_b_vals, R, &zero, masked_b_vals, R), "cublasDgemm");

}

void solveSvdGemm(cudaStream_t stream, cublasHandle_t cublasHandle, double * a_vals, double * b_vals, double * x_vals, int m, int n, int k) {
  // printf("solveSvdGemm\n");
  
  cusolverDnHandle_t cusolverHandle;
  check_cusolver(cusolverDnCreate(&cusolverHandle), "cusolverDnCreate");
  cusolverDnSetStream(cusolverHandle, stream);

  double *work;
  int work_int = 0;
  // assert(A.n_rows == A.n_cols); // A should be square matrix

  // int m = A.n_rows;
  // int k = A.n_cols;
  // int n = B.n_cols;

  check_cublas(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
  gesvdjInfo_t gesvd_info;

  check_cusolver(cusolverDnCreateGesvdjInfo(&gesvd_info), "cusolverDnCreateGesvdjInfo");
  cusolverDnXgesvdjSetMaxSweeps(gesvd_info, 15); // As recommended by cuSOLVER docs

  check_cusolver(
    cusolverDnDgesvdj_bufferSize(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 1, m, m, a_vals, m, NULL, NULL, m, NULL, m, &work_int, gesvd_info), "cusolverDnDgesvdj_bufferSize");
  unsigned int work_length = 2 * m * m + m + work_int;
  check_cuda(cudaMallocAsync(&work, sizeof(double) * work_length, stream), "cudaMallocAsync work");
  int * info;
  check_cuda(cudaMallocAsync(&info, sizeof(int), stream), "cudaMallocAsync info");
  // double * A
  // unsigned int n
  // double * work
  // unsigned int lwork
  // int * info
  // gesvdjInfo_t

  // Take pseudoinverse
  pseudoinverse_gpu(cusolverHandle, cublasHandle, stream, 
    a_vals, m, work, work_length, info, gesvd_info);

  // Multiply V^-1 by b
  check_cublas(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, 
    n, k, &one, a_vals, m, b_vals, k, &zero, x_vals, m), "cublasDgemm");

  // Clean up
  // cublasDestroy(cublasHandle);
  cusolverDnDestroy(cusolverHandle);
  cusolverDnDestroyGesvdjInfo(gesvd_info);
  // cudaStreamDestroy(v_stream);
  // cudaFree(work);
  cudaFreeAsync(work, stream);
  cudaFreeAsync(info, stream);
  
  // cudaFree(info);

}

void solveSvdGemm(const MAT_GPU &A, const MAT_GPU &B, MAT_GPU &X) {

  cudaStream_t v_stream;
  cublasHandle_t cublasHandle;
  cusolverDnHandle_t cusolverHandle;
  check_cuda(cudaStreamCreate(&v_stream), "cudaStreamCreate");
  check_cublas(cublasCreate(&cublasHandle), "cublasCreate");
  check_cusolver(cusolverDnCreate(&cusolverHandle), "cusolverDnCreate");
  
  // Set stream
  cublasSetStream(cublasHandle, v_stream);
  cusolverDnSetStream(cusolverHandle, v_stream);

  // Allocate pseudoinverse array + work
  double *work;

  int work_int = 0;
  assert(A.n_rows == A.n_cols); // A should be square matrix

  int m = A.n_rows;
  int k = A.n_cols;
  int n = B.n_cols;

  check_cublas(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
  gesvdjInfo_t gesvd_info;

  check_cusolver(cusolverDnCreateGesvdjInfo(&gesvd_info), "cusolverDnCreateGesvdjInfo");
  cusolverDnXgesvdjSetMaxSweeps(gesvd_info, 15); // As recommended by cuSOLVER docs

  check_cusolver(
    cusolverDnDgesvdj_bufferSize(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 1, m, m, A.vals, m, NULL, NULL, m, NULL, m, &work_int, gesvd_info), "cusolverDnDgesvdj_bufferSize");
  unsigned int work_length = 2 * m * m + m + work_int;
  check_cuda(cudaMalloc(&work, sizeof(double) * work_length), "cudaMalloc work");
  int * info;
  check_cuda(cudaMalloc(&info, sizeof(int)), "cudaMalloc info");
  // double * A
  // unsigned int n
  // double * work
  // unsigned int lwork
  // int * info
  // gesvdjInfo_t

  // Take pseudoinverse
  pseudoinverse_gpu(cusolverHandle, cublasHandle, v_stream, 
    A.vals, m, work, work_length, info, gesvd_info);

  // Multiply V^-1 by b
  check_cublas(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, 
    n, k, &one, A.vals, m, B.vals, k, &zero, X.vals, m), "cublasDgemm");

  // Clean up
  cublasDestroy(cublasHandle);
  cusolverDnDestroy(cusolverHandle);
  cusolverDnDestroyGesvdjInfo(gesvd_info);
  cudaStreamDestroy(v_stream);
  cudaFree(work);
  cudaFree(info);
}

// This can be done better -- directly setting diagonal entries to 1.0
__global__ void __fill_diagonal_kernel(double * vals, int r, int c, double val) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < r * c) {
    int r_idx = index % r;
    int c_idx = index / r;
    if (r_idx == c_idx) {
      vals[index] = val;
    }
  }
}

/**
 * Assumes mat_vals are already set as 0 through cudaMemset
 * mat_vals are stored in column major -- not sure if that matters though
*/
void fill_diagonal_matrix(cudaStream_t stream, double * mat_vals, int r, int c, double val) {
  __fill_diagonal_kernel<<<r * c / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(mat_vals, r, c, val);
}

__global__ void __compute_trace_kernel(double * gram_vals, double * trace, int R) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double shared[BLOCK_SIZE];

  if (index < R) {
    shared[threadIdx.x] = gram_vals[index * R + index];
  } else {
    shared[threadIdx.x] = 0.0;
  }
  __syncthreads();
  // Reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared[threadIdx.x] += shared[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(trace, shared[0]);
  } 
}

__global__ void __add_rho_to_gram_mat_kernel(double * gram_vals, double * trace, int R) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < R) {
    gram_vals[index * R + index] += *trace / R;
  }
}

void add_rho_to_gram_mat(cudaStream_t stream, double * gram_vals, double * trace, int R)  {
  // compute trace

  __compute_trace_kernel<<<R / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(gram_vals, trace, R);
  __add_rho_to_gram_mat_kernel<<<R/BLOCK_SIZE+1, BLOCK_SIZE, 0, stream>>>(gram_vals, trace, R);
}