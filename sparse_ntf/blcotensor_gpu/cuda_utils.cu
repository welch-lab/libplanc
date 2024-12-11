#include <cassert>
#include <cmath>
#include "cuda_utils.h"
 
template _IType* make_device_copy(_IType* vector, _IType n, std::string name);
template _FType* make_device_copy(_FType* vector, _IType n, std::string name);
template _FType** make_device_copy(_FType** vector, _IType n, std::string name);
template _FType*** make_device_copy(_FType*** vector, _IType n, std::string name);
template unsigned int* make_device_copy(unsigned int* vector, _IType n, std::string name);
template unsigned long* make_device_copy(unsigned long* vector, _IType n, std::string name);
template int* make_device_copy(int* vector, _IType n, std::string name);

void check_cublas(cublasStatus_t status, std::string message) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Error: " << cublasGetStatusString(status);
    std::cerr << ". " << message << std::endl;
    exit(EXIT_FAILURE);
  }
}

void check_cuda(cudaError_t status, std::string message) {
  if (status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(status);
    std::cerr << ". " << message << std::endl;
    exit(EXIT_FAILURE);
  }
}

std::string cusolverGetErrorString(cusolverStatus_t status) {
    switch(status) {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }
    return "unknown error";
}

void check_cusolver(cusolverStatus_t status, std::string message) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Error: " << cusolverGetErrorString(status);
    std::cerr << ". " << message << std::endl;
    exit(EXIT_FAILURE);
  }
}

__global__ void value_dfill_kernel(_FType* x, _IType n, _FType val) {
  _IType index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) x[index] = val;
}

void value_dfill(_FType* x, _IType n, _FType val) {
    value_dfill_kernel<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(x, n, val);
    check_cuda(cudaGetLastError(), "value_fill_kernel launch");
}

__global__ void value_ufill_kernel(unsigned int* x, _IType n, unsigned int val) {
  _IType index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) x[index] = val;
}

/**
 * value fill vector x based on ind vector > 0
 * suboptimal to use branch conditions within kernel but is good for now
*/
__global__ void value_ufill_idx_based_kernel(unsigned int* x, unsigned int* ind, _IType n, unsigned int val) {
  _IType index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    if (ind[index] == 1) x[index] = val; // else keep x[index]
  }
}

void value_ufill(unsigned int* x, _IType n, unsigned int val) {
    value_ufill_kernel<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(x, n, val);
    check_cuda(cudaGetLastError(), "value_fill_kernel launch");
}

void value_ufill_idx_based(unsigned int* x, unsigned int* ind, _IType n, unsigned int val) {
    value_ufill_idx_based_kernel<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(x, ind, n, val);
    check_cuda(cudaGetLastError(), "value_fill_idx_based kernel launch");
}

template <typename v1_type, typename v2_type>
__global__ void __compare_elements_greater(const v1_type * vi, v2_type * vo, int size, double th) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    vo[idx] = (vi[idx] > th) ? 1 : 0;
  }
}

template <typename v1_type, typename v2_type>
__global__ void __compare_elements_greater_or_equal(const v1_type * vi, v2_type * vo, int size, double th) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    vo[idx] = (vi[idx] >= th) ? 1 : 0;
  }
}

template <typename v1_type, typename v2_type>
__global__ void __compare_elements_less(const v1_type * vi, v2_type * vo, int size, double th) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    vo[idx] = (vi[idx] < th) ? 1 : 0;
  }
}

template <typename v1_type, typename v2_type>
__global__ void __compare_elements_less_or_equal(const v1_type * vi, v2_type * vo, int size, double th) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    vo[idx] = (vi[idx] <= th) ? 1 : 0;
  }
}

template <typename v1_type, typename v2_type>
__global__ void __compare_elements_equal(const v1_type * vi, v2_type * vo, int size, double th) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    vo[idx] = (vi[idx] == th) ? 1 : 0;
  }
}

template <typename v1_type, typename v2_type>
__global__ void __compare_elements_not_equal(const v1_type * vi, v2_type * vo, int size, double th) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    vo[idx] = (vi[idx] != th) ? 1 : 0;
  }
}

template <COMPARE_OP OP, typename v1_type, typename v2_type>
void __compare_elements(const v1_type * vi, v2_type * vo, int size, double th) {
  int num_tblocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  switch (OP) {
    case GREATER:
      __compare_elements_greater<v1_type, v2_type><<<num_tblocks, BLOCK_SIZE>>>(vi, vo, size, th);
      break;
    case GREATER_OR_EQ:
      __compare_elements_greater_or_equal<v1_type, v2_type><<<num_tblocks, BLOCK_SIZE>>>(vi, vo, size, th);
      break;
    case LESS:
      __compare_elements_less<v1_type, v2_type><<<num_tblocks, BLOCK_SIZE>>>(vi, vo, size, th);
      break;
    case LESS_OR_EQ:
      __compare_elements_less_or_equal<v1_type, v2_type><<<num_tblocks, BLOCK_SIZE>>>(vi, vo, size, th);
      break;
    case EQ:
      __compare_elements_equal<v1_type, v2_type><<<num_tblocks, BLOCK_SIZE>>>(vi, vo, size, th);
      break;
    case NEQ:
      __compare_elements_not_equal<v1_type, v2_type><<<num_tblocks, BLOCK_SIZE>>>(vi, vo, size, th);
      break;

    default:
      break;
  }
}

template void __compare_elements<GREATER>(const double * vi, unsigned int * vo, int size, double th);
template void __compare_elements<GREATER_OR_EQ>(const double * vi, unsigned int * vo, int size, double th);
template void __compare_elements<LESS>(const double * vi, unsigned int * vo, int size, double th);
template void __compare_elements<LESS_OR_EQ>(const double * vi, unsigned int * vo, int size, double th);
template void __compare_elements<EQ>(const double * vi, unsigned int * vo, int size, double th);

// Populate templated code
template void __compare_elements<EQ>(const unsigned int * vi, unsigned int * vo, int size, double th);
template void __compare_elements<NEQ>(const unsigned int * vi, unsigned int * vo, int size, double th);
template void __compare_elements<GREATER>(const unsigned int * vi, unsigned int * vo, int size, double th);
template void __compare_elements<GREATER_OR_EQ>(const unsigned int * vi, unsigned int * vo, int size, double th);
template void __compare_elements<LESS>(const unsigned int * vi, unsigned int * vo, int size, double th);

__global__ void __mat_transpose(double * _mat_t, double * _mat, int m, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < m * n) {
    int orig_x = idx / m;
    int orig_y = idx % m;
    int dest_idx = orig_y * n + orig_x;
    _mat_t[dest_idx] = _mat[idx];
  }
}

void mat_transpose_inplace(MAT_GPU * mat) {
  int m = mat->n_rows;
  int n = mat->n_cols;
  // printf("%d %d\n", m, n); // 4 183
  double * temp_vals;
  check_cuda(cudaMalloc((void**)&temp_vals, sizeof(double) * m * n), "cuda malloc temp mat for .T");

  cublasHandle_t handle; double alpha = 1.0; double beta = 0.0;
  check_cublas(cublasCreate(&handle), "create cublas handle");
  check_cublas(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, 
      &alpha, mat->vals, m,
      &beta, temp_vals, n, 
      temp_vals, n), "mat transpose");
  cublasDestroy(handle);
  cudaFree(mat->vals);
  // C = B, ldc = ldb and transb = CUBLAS_OP_N
  mat->vals = temp_vals;
  mat->n_rows = n;
  mat->n_cols = m;
}

// Hadamard update, i.e. x <-- x .* y
__global__ void hadamard_kernel(_FType* x, _FType* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) x[index] *= y[index];
}
void vector_hadamard(_FType* x, _FType* y, int n) {
    hadamard_kernel <<<n / BLOCK_SIZE + 1, BLOCK_SIZE, 0>>>(x, y, n);
    check_cuda(cudaGetLastError(), "hadamard_kernel launch");
}

__global__ void normalizeColumn(double* matrix, double* lambda, int numRows, int numCols) {
    // chooses column
    int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank < numCols) {
        double * col = matrix + rank * numRows;
        // Calculate the Frobenius norm of the column
        double norm = 0.0;
        for (int i = 0; i < numRows; ++i) {
            norm += col[i] * col[i];
        }
        norm = sqrtf(norm);

        if (norm > 0.0) {
          for (int i = 0; i < numRows; ++i) {
              col[i] /= norm;
          }
        }
        // Normalize the element in the column
        // matrix[index] /= norm;
        lambda[rank] = norm;
    }
}

void normalize_fm(MAT_GPU * fm, double * lambda) {
  int rank = fm->n_cols;
  int dim = fm->n_rows;

  int blockSize = BLOCK_SIZE;
  int gridSize = (rank + blockSize - 1) / blockSize;

  normalizeColumn<<<gridSize, blockSize>>>(fm->vals, lambda, dim, rank);
  check_cuda(cudaDeviceSynchronize(), "sync after normaliztation");
}

void normalize_mat_cublas(int m, int n, double * mat_vals, double * lambda) {
  double * d_norms = (double*) calloc(n, sizeof(double));
  // 1/d_norms
  double * rec_d_norms = (double*) malloc(n * sizeof(double));
  cudaStream_t streams[n];
  cublasHandle_t handles[n];

  for (int r = 0; r < n; ++r) {
    check_cuda(cudaStreamCreate(&streams[r]), "cudaStreamCreate");
    check_cublas(cublasCreate(&handles[r]), "cublasCreate");
    check_cublas(cublasSetStream(handles[r], streams[r]), "cublasSetStream");
  }
  for (int r = 0; r < n; ++r) {
    // double lambda = 0.0;

    check_cublas(cublasDnrm2(handles[r], m, mat_vals + r*m, 1, &d_norms[r]), "cublas Dnrm2 -- compute norm");
    // cudaMemcpyAsync(&rec_d_norms[r], &d_norms[r], sizeof(double), cudaMemcpyDeviceToHost, streams[r]);
    rec_d_norms[r] = fabs(d_norms[r]) > 1e-12 ? 1 / d_norms[r] : 0;
    check_cublas(cublasDscal(handles[r], m, &rec_d_norms[r], mat_vals+r*m, 1), "cublas Dscal -- divide entries by lambda"); 
  }
  for (int r = 0; r < n; ++r) {
    check_cuda(cudaStreamSynchronize(streams[r]), "cudaStream sync");
    check_cublas(cublasDestroy(handles[r]), "cublas handle destroy");
    check_cuda(cudaStreamDestroy(streams[r]), "cudaStream destroy");
  }

  // check_cuda(cudaMemcpy(lambda, d_norms, sizeof(double) * n, cudaMemcpyDeviceToDevice), "cudaMemcpy -- lambda to host");
  check_cuda(cudaDeviceSynchronize(), "normalize fm complete");
  check_cuda(cudaMemcpy(lambda, d_norms, sizeof(double) * n, cudaMemcpyHostToDevice), "memcpy");

  free(rec_d_norms);
  free(d_norms);
}

void normalize_fm_cublas(MAT_GPU * fm, double * lambda) {
  int rank = fm->n_cols;
  int dim = fm->n_rows;
  // Array to store norms for each columns
  
  // double * d_norms;
  // check_cuda(cudaMalloc((void**)&d_norms, rank * sizeof(double)), "cudaMalloc d_norm");
  // check_cuda(cudaMemset(d_norms, 0, rank * sizeof(double)), "cudaMemset d_norm to zero");

  double * d_norms = (double*) calloc(rank, sizeof(double));
  // 1/d_norms
  double * rec_d_norms = (double*) malloc(rank * sizeof(double));

  cudaStream_t streams[rank];
  cublasHandle_t handles[rank];

  for (int r = 0; r < rank; ++r) {
    check_cuda(cudaStreamCreate(&streams[r]), "cudaStreamCreate");
    check_cublas(cublasCreate(&handles[r]), "cublasCreate");
    check_cublas(cublasSetStream(handles[r], streams[r]), "cublasSetStream");
  }
  for (int r = 0; r < rank; ++r) {
    // double lambda = 0.0;

    check_cublas(cublasDnrm2(handles[r], dim, fm->vals + r*dim, 1, &d_norms[r]), "cublas Dnrm2 -- compute norm");
    // cudaMemcpyAsync(&rec_d_norms[r], &d_norms[r], sizeof(double), cudaMemcpyDeviceToHost, streams[r]);
    rec_d_norms[r] = fabs(d_norms[r]) > 1e-12 ? 1 / d_norms[r] : 0;

    check_cublas(cublasDscal(handles[r], dim, &rec_d_norms[r], fm->vals+r*dim, 1), "cublas Dscal -- divide entries by lambda"); 

  }
  for (int r = 0; r < rank; ++r) {
    check_cuda(cudaStreamSynchronize(streams[r]), "cudaStream sync");
    check_cublas(cublasDestroy(handles[r]), "cublas handle destroy");
    check_cuda(cudaStreamDestroy(streams[r]), "cudaStream destroy");
  }

  // check_cuda(cudaMemcpy(lambda, d_norms, sizeof(double) * rank, cudaMemcpyDeviceToDevice), "cudaMemcpy -- lambda to host");
  check_cuda(cudaDeviceSynchronize(), "normalize fm complete");

  check_cuda(cudaMemcpy(lambda, d_norms, sizeof(double) * rank, cudaMemcpyHostToDevice), "memcpy");


  free(rec_d_norms);
  free(d_norms);
}

void mat_mat_mul(_FType* a, _FType* b, _FType* c, int m, int n, int k, double alpha, double beta) {
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "create cublas handle");
  check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
  m, k, n, &alpha, a, m, b, n, &beta, c, m), "mat_mat_mul");
  cublasDestroy(handle);
}

void mat_vec_mul(_FType* a, _FType* b, _FType* c, int m, int n, double alpha, double beta) {
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "create cublas handle");
  check_cublas(cublasDgemv(handle, CUBLAS_OP_N, 
  m, n, &alpha, a, m, b, 1, &beta, c, 1), "mat_vec_mul");
  cublasDestroy(handle);
}

__global__ void __apply_threshold(double* v, int n, const double th, const double repl) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    v[idx] = (v[idx] < th) ? repl : v[idx];
  }
}

__global__ void __apply_nonnegative_projection_kernel(double* v, double * diff, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double val = v[idx];
    v[idx] = (val >= 0) * val;
    diff[idx] = (val < 0) * val;
  }
}

void apply_nonnegative_projection(double * v, double * diff, int n) {
  __apply_nonnegative_projection_kernel<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, 0>>>(v, diff, n);
}

MAT_GPU * init_mat_gpu(int m, int n) {
  MAT_GPU * _mat = new MAT_GPU(m, n);
  check_cuda(cudaMalloc((void**)&_mat->vals, sizeof(_FType) * m * n), "cudaMalloc init_mat_gpu");
  check_cuda(cudaMemset(_mat->vals, 0, sizeof(_FType) * m * n), "cudaMemset init_mat_gpu");
  return _mat;
}

void copy_mat_gpu(MAT_GPU * dest, const MAT_GPU * src) {
  check_cuda(
    cudaMemcpy(dest->vals, src->vals, sizeof(_FType) * dest->n_cols * dest->n_rows, cudaMemcpyDeviceToDevice), 
    "cudaMemcpy copy_mat_gpu");
}

void free_mat_gpu(MAT_GPU * mat) {
  cudaFree(mat->vals);
}

void free_umat_gpu(UMAT_GPU * umat) {
  cudaFree(umat->vals);
}

// __global__ void vec_add(const double * v1, const double * v2, double * v3, int size) {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < size) {
//         v3[i] = v1[i] + v2[i];
//     }
// }

// __global__ void vec_sub(const double * v1, const double * v2, double * v3, int size) {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < size) {
//         v3[i] = v1[i] - v2[i];
//     }
// }

// __global__ void vec_mult(const double * v1, const double * v2, double * v3, int size) {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < size) {
//         v3[i] = v1[i] * v2[i];
//     }
// }


// __global__ void vec_div(const double * v1, const double * v2, double * v3, int size) {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < size) {
//         if (v2[i] != 0.0f) {
//             v3[i] = v1[i] / v2[i];
//         } else {
//             // Handle division by zero
//             v3[i] = 0.0f; // You can choose an appropriate value here
//         }
//     }
// }

// __global__ void vec_scale(const double * v1, double * v2, int size, double alpha) {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < size) {
//         v2[i] = v1[i] * alpha;
//     }
// }

__global__ void __mat_trace(double * v, int size, int size_row, double * trace) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int row_idx = i / size_row;
    int col_idx = i % size_row;

    if (row_idx == col_idx) {
      atomicAdd(trace, v[i]);
    } 
  }
}

__global__ void __mat_add_diag(double * v, int size, int size_row, double diag_val) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int row_idx = i / size_row;
    int col_idx = i % size_row;

    if (row_idx == col_idx) {
      v[i] += diag_val;
    } 
  }
}

double mat_trace_gpu(const MAT_GPU * mat) {
  assert(mat->n_cols == mat->n_rows);
  int num_tblocks = (mat->n_cols * mat->n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double * trace;
  cudaMalloc((void**)&trace, sizeof(double));
  cudaMemset(trace, 0, sizeof(double));
  __mat_trace<<<num_tblocks, BLOCK_SIZE>>>(mat->vals, mat->n_cols * mat->n_rows, mat->n_rows, trace);
  cudaDeviceSynchronize();
  double _trace;
  cudaMemcpy(&_trace, trace, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(trace);
  return _trace;
}

void mat_add_diag_gpu(MAT_GPU * mat, double val) {
  assert(mat->n_cols == mat->n_rows);
  int num_tblocks = (mat->n_cols * mat->n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  __mat_add_diag<<<num_tblocks, BLOCK_SIZE>>>(mat->vals, mat->n_cols * mat->n_rows, mat->n_rows, val);
  cudaDeviceSynchronize();
}

__global__ void __mat_copy_lower(double * l_vals, double * vals, int size, int size_row) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int row_idx = i / size_row;
    int col_idx = i % size_row;

    if (row_idx <= col_idx) {
      l_vals[i] = vals[i];
    } else {
      l_vals[i] = 0.0;
    }
  }
}

__global__ void __mat_copy_upper(double * u_vals, double * vals, int size, int size_row) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int row_idx = i / size_row;
    int col_idx = i % size_row;

    if (row_idx >= col_idx) {
      u_vals[i] = vals[i];
    } else {
      u_vals[i] = 0.0;
    }
  }
}

void copy_mat_gpu_lower(MAT_GPU * mat_l, MAT_GPU * mat) {
  assert(mat_l->n_rows == mat->n_cols); // sanity check
  int num_tblocks = (mat->n_cols * mat->n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  __mat_copy_lower<<<num_tblocks, BLOCK_SIZE>>>(mat_l->vals, mat->vals, mat->n_cols * mat->n_rows, mat->n_rows);
  cudaDeviceSynchronize();
}

void copy_mat_gpu_upper(MAT_GPU * mat_u, MAT_GPU * mat) {
  assert(mat_u->n_rows == mat->n_cols); // sanity check
  int num_tblocks = (mat->n_cols * mat->n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  __mat_copy_upper<<<num_tblocks, BLOCK_SIZE>>>(mat_u->vals, mat->vals, mat->n_cols * mat->n_rows, mat->n_rows);
  cudaDeviceSynchronize();
}

void mat_cholesky_gpu(MAT_GPU * mat, bool is_lower) {
  assert(mat->n_rows == mat->n_cols);
  int n = mat->n_rows; // mat should be square
  
  cusolverDnHandle_t cusolver = NULL;
  cusolverDnCreate(&cusolver);
  
  // init workspace, lwork, devInfo
  int lwork = 0;
  int *devInfo;
  int devInfo_h = 0; // to check cholesky decomposition
  double * workspace;

  // get workspace size
  cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_LOWER, n, mat->vals, n, &lwork);

  cudaMalloc((void**)&workspace, lwork * sizeof(double));
  cudaMalloc(&devInfo, sizeof(int));
  check_cusolver(
    cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_LOWER, n, mat->vals, n, workspace, lwork, devInfo),
    "potrf cusolver"
  );

  cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

  if (devInfo_h != 0) printf("Cholesky decomposition failed (devInfo_h = %d).\n", devInfo_h);
  cudaFree(workspace);
  cusolverDnDestroy(cusolver);
}

// A * X = B
void mat_cholesky_solve_gpu(const MAT_GPU * A, MAT_GPU * B, bool is_lower) {
  int n = A->n_rows; // number of rows and columns of matrix A
  int nrhs = B->n_cols; // number of colums of matrix X and B
  cusolverDnHandle_t cusolver = NULL;
  cusolverDnCreate(&cusolver);

  cublasFillMode_t uplo = is_lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  int *devInfo;
  int devInfo_h = 0; // to check cholesky decomposition
  cudaMalloc(&devInfo, sizeof(int));

  check_cusolver(
    cusolverDnDpotrs(cusolver, uplo, n, nrhs, A->vals, n, B->vals, n, devInfo),
    "cusolver cholesky solve"
  );

  cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

  if (devInfo_h != 0) printf("Cholesky solve failed (devInfo_h = %d).\n", devInfo_h);
  cusolverDnDestroy(cusolver);

}

template <typename T> 
__global__ void reduce_sum_col_gpu(const T * v, T * sum_vector, int rows, int cols) {
  extern __shared__ T _sdata[];
  
  // Each block processes one column at a time
  unsigned int col = blockIdx.x;
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * rows + tid; // the element within the column blockIdx.x
  // int tid = blockDim.x * blockIdx.x + tid;
  _sdata[tid] = (tid < rows) ? v[i] : 0;
  __syncthreads();

  // Perform reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
          _sdata[tid] += _sdata[tid + s];
      }
      __syncthreads();
  }

  // Write result for this block to global mem
  if (tid == 0) sum_vector[col] += _sdata[0];
}

template <typename T> 
__global__ void reduce_sum_gpu(const T * v, int size, T * sum) {
  extern __shared__ double sdata[];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  sdata[threadIdx.x] = 0.0;

  for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
    sdata[threadIdx.x] += v[i];
  }

  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Store the norm in global memory
  if (threadIdx.x == 0) {
    atomicAdd(sum, sdata[0]);
  }

}

template <> 
__global__ void reduce_sum_gpu(const double * v, int size, double * sum) {
  extern __shared__ double sdata[];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  sdata[threadIdx.x] = 0.0;

  for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
    sdata[threadIdx.x] += v[i];
  }

  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Store the norm in global memory
  if (threadIdx.x == 0) {
    atomicAdd(sum, sdata[0]);
  }
}

// need to delete
__global__ void frob_norm_gpu(const double * v, int rows, int cols, double * norm) {
  extern __shared__ double sdata[];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  sdata[threadIdx.x] = 0.0;

  for (int i = tid; i < rows * cols; i += blockDim.x * gridDim.x) {
    double element = v[i];
    sdata[threadIdx.x] += element * element;
  }
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Store the norm in global memory
  if (threadIdx.x == 0) {
    atomicAdd(norm, sdata[0]);
    *norm = sqrtf(*norm);
  }
}

__global__ void frob_norm_gpu(const double * v1, const double * v2, int rows, int cols, double * norm) {
  extern __shared__ double sdata[];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  sdata[threadIdx.x] = 0.0;

  for (int i = tid; i < rows * cols; i += blockDim.x * gridDim.x) {
    double element = v1[i]-v2[i];
    sdata[threadIdx.x] += element * element;
  }
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Store the norm in global memory
  if (threadIdx.x == 0) {
    atomicAdd(norm, sdata[0]);
    *norm = sqrtf(*norm);
  }
}

UMAT_GPU UMAT_GPU::operator%(const UMAT_GPU & other) const {
  UMAT_GPU bool_mat = UMAT_GPU(n_rows, n_cols);
  check_cuda(cudaMalloc((void**)&bool_mat.vals, sizeof(unsigned int) * n_rows * n_cols), "malloc umat_gpu");
  uvec_mult(this->vals, other.vals, bool_mat.vals, n_rows * n_cols);
    
  return bool_mat;
}
// sum along columns
// threads per block is usually less than R used for TF
// So we are not considering cases where rank exceeds threadsPerBlock
UVEC_GPU UMAT_GPU::sum() const {
  int m = this->n_rows;
  int n = this->n_cols;

  // sum along columns only
  UVEC_GPU sum_vec = UVEC_GPU(n);
  // int num_tblocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // int shared_mem = sizeof(unsigned int) * BLOCK_SIZE;
  // for (int i = 0; i < n; ++i) {
  //   reduce_sum_gpu<unsigned int><<<num_tblocks, BLOCK_SIZE, shared_mem>>>(this->vals + i * m, m, &sum_vec.vals[i]);
  // }
  const int threadsPerBlock = 256;
  const int blocks = this->n_cols;
  int sharedMemSize = threadsPerBlock * sizeof(unsigned int);
  reduce_sum_col_gpu<unsigned int><<<blocks, threadsPerBlock, sharedMemSize>>>(this->vals, sum_vec.vals, m, n);
  check_cuda(cudaDeviceSynchronize(), "reduce sum for all columns in UMAT_GPU");

  return sum_vec;
}

unsigned int UVEC_GPU::sum() const {
  int m = this->size;
  unsigned int * d_sum;
  unsigned int sum = 0;
  check_cuda(cudaMalloc((void**)&d_sum, sizeof(unsigned int)), "cudaMalloc sum -- scalar");
  check_cuda(cudaMemset(d_sum, 0, sizeof(unsigned int)), "cudaMemset");
  // sum along columns only
  int num_tblocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int shared_mem = sizeof(unsigned int) * BLOCK_SIZE;
  reduce_sum_gpu<unsigned int><<<num_tblocks, BLOCK_SIZE, shared_mem>>>(this->vals, m, d_sum);
  check_cuda(cudaDeviceSynchronize(), "reduce sum for all columns in UVEC_GPU");
  cudaMemcpy(&sum, d_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(d_sum);
  return sum;
}

UVEC_GPU operator-(const UVEC_GPU & lhs, const UVEC_GPU & rhs) {
  UVEC_GPU sub_vec = UVEC_GPU(lhs.size);
  uvec_sub(lhs.vals, rhs.vals, sub_vec.vals, sub_vec.size);
  return sub_vec;
}

UVEC_GPU operator+(const UVEC_GPU & lhs, const UVEC_GPU & rhs) {
  UVEC_GPU sum_vec = UVEC_GPU(lhs.size);
  uvec_add(lhs.vals, rhs.vals, sum_vec.vals, sum_vec.size);
  return sum_vec;
}

UVEC_GPU operator%(const UVEC_GPU & lhs, const UVEC_GPU & rhs) {
  UVEC_GPU sum_vec = UVEC_GPU(lhs.size);
  uvec_mult(lhs.vals, rhs.vals, sum_vec.vals, sum_vec.size);
  return sum_vec;
}

// Returns lhs[i] - rhs[i] > 0
UVEC_GPU operator>(const UVEC_GPU & lhs, const UVEC_GPU & rhs) {
  UVEC_GPU sum_vec = UVEC_GPU(lhs.size);
  uvec_sub(lhs.vals, rhs.vals, sum_vec.vals, sum_vec.size);
  UVEC_GPU res_vec = sum_vec > 0;
  return res_vec;
}

UVEC_GPU operator==(const UVEC_GPU & lhs, const UVEC_GPU & rhs) {
  UVEC_GPU sub_vec1 = UVEC_GPU(lhs.size);
  UVEC_GPU sub_vec2 = UVEC_GPU(lhs.size);
  uvec_sub(lhs.vals, rhs.vals, sub_vec1.vals, sub_vec1.size);
  uvec_sub(rhs.vals, lhs.vals, sub_vec2.vals, sub_vec2.size);
  return (sub_vec1 == 0) % (sub_vec2 == 0);
}

// Returns lhs[i] - rhs[i] >= 0
UVEC_GPU operator>=(const UVEC_GPU & lhs, const UVEC_GPU & rhs) {
  UVEC_GPU sub_vec = UVEC_GPU(lhs.size);
  uvec_sub(lhs.vals, rhs.vals, sub_vec.vals, sub_vec.size);
  UVEC_GPU temp = sub_vec > 0; // rhs is strictly larger than lhs
  UVEC_GPU temp2 = lhs == rhs;

  // UVEC_GPU res_vec = (sub_vec > 0) != 1; 
  return temp2 + temp;
}

UVEC_GPU operator<(const UVEC_GPU & lhs, const UVEC_GPU & rhs) {
  UVEC_GPU sum_vec = UVEC_GPU(lhs.size);
  UVEC_GPU bool_vec = UVEC_GPU(lhs.size);
  uvec_sub(rhs.vals, lhs.vals, sum_vec.vals, sum_vec.size);
  // UVEC_GPU res_vec = (sum_vec > 0) == 1;
  return (sum_vec > 0) == 1;
}

template <typename T>
__global__ void value_fill_kernel(T * x, unsigned int n, T val) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) x[index] = val;
}

template <typename T>
void value_fill(T * x, unsigned int n, T val) {
  value_fill_kernel<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(x, n, val);
  check_cuda(cudaGetLastError(), "value_fill_kernel launch");
}