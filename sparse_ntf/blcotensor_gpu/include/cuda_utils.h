#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <iostream>
#include <tuple>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "cuda_runtime.h"
#include "cassert"

#include "config.h"
#include "blco.h"
#include "vector_ops.h"

#define LVL3_MAX_MODE_LENGTH 100
#define EPSILON 0.000001

#define _IType unsigned long long
#define _FType double

// Crashes if the given CUDA error status is not successful
//
// Parameters:
//  - status: the status returned from a CUDA call
//  - message: a provided error string to display alongside the error
// Returns:
//  - none
void check_cuda(cudaError_t status, std::string message);

// Crashes if the given cuBLAS error status is not successful
//
// Parameters:
//  - status: the status returned from a cuBLAS call
//  - message: a provided error string to display alongside the error
// Returns:
//  - none
void check_cublas(cublasStatus_t status, std::string message);
void check_cusolver(cusolverStatus_t status, std::string message);

__global__ void __apply_threshold(double* v, int n, const double th, const double repl);
void apply_nonnegative_projection(double * v, double * diff, int n);
__global__ void __apply_nonnegative_projection_kernel(double* v, double * diff, int n);

enum COMPARE_OP {
  GREATER,
  GREATER_OR_EQ,
  LESS,
  LESS_OR_EQ,
  EQ,
  NEQ
};

// Fills every element of vector x with the same value
// can't use floating point data struct as template param
__global__ void value_dfill_kernel(_FType* x, _IType n, _FType val);
__global__ void value_ufill_kernel(unsigned int * x, _IType n, unsigned int val);
__global__ void value_ufill_idx_based_kernel(unsigned int* x, unsigned int* ind, _IType n, unsigned int val);

void value_ufill_idx_based(unsigned int* x, unsigned int* ind, _IType n, unsigned int val);
void value_ufill(unsigned int * x, _IType n, unsigned int val);
void value_dfill(_FType * x, _IType n, _FType val);

template <typename T>
__global__ void value_fill_kernel(T * x, unsigned int n, T val);

template <typename T>
void value_fill(T * x, unsigned int n, T val);

// void value_fill<unsigned int>(unsigned int * x, unsigned int n, unsigned int val);
//  {
//   value_fill_kernel<<<n / BLOCK_SIZE + 1, BLOCK_SIZE>>>(x, n, val);
// //   check_cuda(cudaGetLastError(), "value_fill_kernel launch");
// }


// #define value_ufill value_fill<unsigned int> (unsigned int * x, unsigned int n, unsigned int val);
template <COMPARE_OP OP, typename v1_type, typename v2_type>
void __compare_elements(const v1_type * vi, v2_type * vo, int size, double th);

struct UVEC_GPU {
  unsigned int size;
  unsigned int * vals;
  
  UVEC_GPU(): size(0) {}
  UVEC_GPU(int n): size(n) {
    check_cuda(cudaMalloc((void**)&vals, sizeof(unsigned int) * size), "cudaMalloc UVEC_GPU");
    // if not explicitly set as zero, there are scenarios where the a new object carrys-on values from
    // "free"-ed old objects -- found out the hard way
    check_cuda(cudaMemset(vals, 0, sizeof(unsigned int) * size), "cudaMemset UVEC_GPU");
  }

  bool empty() {
    return this->sum() == 0;
  }

  void fill(unsigned int val) {
    value_ufill(vals, size, val);
  }

  /**
   * fills values of vector based on indicator vector > 0
   * In attempt to mimic P(arma::find(Cols1)).fill(pbar)
  */
  void idx_based_fill(const UVEC_GPU& indicator_vec, unsigned int val) {
    if (size != indicator_vec.size) return; // error if size doesn't match
    value_ufill_idx_based(this->vals, indicator_vec.vals, size, val);
  }

  /**
   * copies values of vector from other vector based on indicator vector > 0
   * In attempt to mimic Ninf(arma::find(Cols1)) = NotGood(arma::find(Cols1))
  */
  void idx_based_copy(const UVEC_GPU& indicator_vec, const UVEC_GPU& copy_vec) {
    assert(size == indicator_vec.size);
    assert(size == copy_vec.size);
    uvec_copy_idx_based(copy_vec.vals, this->vals, indicator_vec.vals, size);
  }

  /**
   * subtracts value from vector based on indicator vector > 0
   * In attempt to mimic P(arma::find(Cols2)) -= 1
  */
  void idx_based_sub(const UVEC_GPU& indicator_vec, unsigned int val) {
    assert(size == indicator_vec.size);
    uvec_sub_idx_based(vals, indicator_vec.vals, size, val);
  }

  UVEC_GPU operator>(double value) const {
    UVEC_GPU bool_vec = UVEC_GPU(size);
    __compare_elements<GREATER, unsigned int, unsigned int>(this->vals, bool_vec.vals, size, value);
    return bool_vec;  
  }
  UVEC_GPU operator>=(double value) const {
    UVEC_GPU bool_vec = UVEC_GPU(size);
    __compare_elements<GREATER_OR_EQ, unsigned int, unsigned int>(this->vals, bool_vec.vals, size, value);
    return bool_vec;  
  }
  UVEC_GPU operator<(double value) const {
    UVEC_GPU is_greater_or_eq = UVEC_GPU(size);
    __compare_elements<GREATER_OR_EQ>(this->vals, is_greater_or_eq.vals, size, value);
    UVEC_GPU bool_vec = UVEC_GPU(size);
    __compare_elements<NEQ>(is_greater_or_eq.vals, bool_vec.vals, size, 1);
    return bool_vec;
  }
  UVEC_GPU operator==(double value) const {
    UVEC_GPU bool_vec = UVEC_GPU(size);
    __compare_elements<EQ, unsigned int, unsigned int>(this->vals, bool_vec.vals, size, value);
    return bool_vec; 
  }
  UVEC_GPU operator!=(double value) const {
    UVEC_GPU bool_vec = UVEC_GPU(size);
    __compare_elements<NEQ, unsigned int, unsigned int>(this->vals, bool_vec.vals, size, value);
    return bool_vec;
  }

  unsigned int sum() const;
  friend UVEC_GPU operator+(const UVEC_GPU& lhs, const UVEC_GPU& rhs);
  friend UVEC_GPU operator-(const UVEC_GPU& lhs, const UVEC_GPU& rhs);
  friend UVEC_GPU operator%(const UVEC_GPU& lhs, const UVEC_GPU& rhs);
  friend UVEC_GPU operator<(const UVEC_GPU& lhs, const UVEC_GPU& rhs);
  friend UVEC_GPU operator>(const UVEC_GPU& lhs, const UVEC_GPU& rhs);
  friend UVEC_GPU operator>=(const UVEC_GPU& lhs, const UVEC_GPU& rhs);
  friend UVEC_GPU operator==(const UVEC_GPU& lhs, const UVEC_GPU& rhs);

  UVEC_GPU& operator=(const UVEC_GPU & obj) {
    if (this == &obj) {
      return *this;
    }

    if (vals != nullptr)
    check_cuda(cudaFree(vals), "cudaFree assignment overload UVEC_GPU"); // Release existing CUDA memory

    size = obj.size;
    check_cuda(cudaMalloc((void**)&vals, sizeof(unsigned int) * size), "cudaMalloc UVEC_GPU");
    check_cuda(cudaMemcpy(vals, obj.vals, sizeof(unsigned int) * size, cudaMemcpyDeviceToDevice), "cudaMemset UVEC_GPU");
    return *this;
  }
  ~UVEC_GPU() {
    check_cuda(cudaFree(vals), "cudaFree destructor for UVEC_GPU");
  }
};

struct UMAT_GPU {
  int n_rows;
  int n_cols;

  unsigned int * vals;
  UMAT_GPU(int r, int c) : n_rows(r), n_cols(c) {
    check_cuda(cudaMalloc((void**)&vals, sizeof(unsigned int) * r * c), "cudaMalloc UMAT_GPU");
    // if not explicitly set as zero, there are scenarios where the a new object carrys-on values from
    // "free"-ed old objects -- found out the hard way
    check_cuda(cudaMemset(vals, 0, sizeof(unsigned int) * r * c), "cudaMemset UMAT_GPU");
  }
  UMAT_GPU(): n_rows(0), n_cols(0) {}
  UMAT_GPU(const UMAT_GPU& other) {
    int r = other.n_rows;
    int c = other.n_cols;

    check_cuda(cudaMalloc((void**)&vals, sizeof(unsigned int) * r * c), "cudaMalloc UMAT_GPU -- deep copy");
    // if not explicitly set as zero, there are scenarios where the a new object carrys-on values from
    // "free"-ed old objects -- found out the hard way
    check_cuda(cudaMemcpy(vals, other.vals, sizeof(unsigned int) * r * c, cudaMemcpyDeviceToDevice), "UMAT_GPU constructor -- deep copy");
  }

  UMAT_GPU copy() const {
    return UMAT_GPU(*this);
  }

  void fill(const unsigned int val) {
    check_cuda(cudaMalloc(
      (void**)&this->vals,
      sizeof(unsigned int) * n_rows * n_cols), "malloc umat_gpu");
    value_ufill(this->vals, n_rows * n_cols, val);
  }
  UMAT_GPU operator%(const UMAT_GPU& other) const;
  UVEC_GPU sum() const;
  UMAT_GPU operator==(double value) const {
    UMAT_GPU bool_mat = UMAT_GPU(n_rows, n_cols);
    check_cuda(cudaMalloc((void**)&bool_mat.vals, sizeof(unsigned int) * n_rows * n_cols), "malloc umat_gpu");
    __compare_elements<EQ, unsigned int, unsigned int>(this->vals, bool_mat.vals, n_rows * n_cols, value);
    return bool_mat;
  }

  /**
   * multiplies a UVEC_GPU across all rows of UMAT_GPU
   * since UMAT_GPU is stored in column-wise format
   * apply scaling to each column
  */
  void rowwise_mult(const UVEC_GPU& mult_vec) {
    assert(mult_vec.size == this->n_cols);
    uint32_t * mult_vec_vals = (uint32_t*) malloc(mult_vec.size * sizeof(uint32_t));
    cudaMemcpy(mult_vec_vals, mult_vec.vals, sizeof(uint32_t) * mult_vec.size, cudaMemcpyDeviceToHost);
    for (int cidx = 0; cidx < this->n_cols; ++cidx) {
      int offset = cidx * this->n_rows;
      uvec_scale(this->vals+offset, this->vals+offset, mult_vec_vals[cidx], this->n_rows);
    }
    free(mult_vec_vals);
  }

  /**
   * given a UMAT_GPU as indicator and a value to fill
   * fills the matrix accordingly
  */
  void idx_based_fill(const UMAT_GPU& ind_mat, const unsigned int val) {
    assert(ind_mat.n_rows == this->n_rows);
    assert(ind_mat.n_cols == this->n_cols);
    value_ufill_idx_based(this->vals, ind_mat.vals, n_rows * n_cols, val);
  }

  /* The RHS of the = operator is assumed to have GPU memory allocated */
  UMAT_GPU& operator=(const UMAT_GPU & obj) {
    if (this == &obj) {  // Check for self-assignment
      return *this;
    }
    // if (vals != nullptr)
    //   check_cuda(cudaFree(vals), "cudaFree assignment overload UMAT_GPU"); // Release existing CUDA memory
    n_rows = obj.n_rows;
    n_cols = obj.n_cols;
    // check_cuda(cudaMalloc((void**)&vals, sizeof(unsigned int) * n_rows * n_cols), "cudaMalloc UMAT_GPU");
    check_cuda(cudaMemcpy(vals, obj.vals, sizeof(unsigned int) * n_rows * n_cols, cudaMemcpyDeviceToDevice), "overload = operator; cudaMemset UMAT_GPU");
    return *this;
  }

  ~UMAT_GPU() {
    check_cuda(cudaFree(vals), "cudaFree destructor for UMAT_GPU");
  }
};

struct MAT_GPU {
  int n_rows;
  int n_cols;

  // Kind of hacky -- not always defined..
  // Can we use stream[0] as default?
  // if stream is NULL by default it will use 0-th stream
  cudaStream_t stream = NULL;
  
  double * vals;
  // constructor
  MAT_GPU(int r, int c) : n_rows(r), n_cols(c) {
    check_cuda(cudaMalloc((void**)&vals, sizeof(double) * r * c), "cudaMalloc MAT_GPU");
    // if not explicitly set as zero, there are scenarios where the a new object carrys-on values from
    // "free"-ed old objects -- found out the hard way
    check_cuda(cudaMemset(vals, 0, sizeof(double) * r * c), "cudaMemset MAT_GPU");
  }

  MAT_GPU(int r, int c, cudaStream_t stream) : n_rows(r), n_cols(c), stream(stream) {
    check_cuda(cudaMallocAsync((void**)&vals, sizeof(double) * r * c, stream), "cudaMallocAsync MAT_GPU");
    // if not explicitly set as zero, there are scenarios where the a new object carrys-on values from
    // "free"-ed old objects -- found out the hard way
    check_cuda(cudaMemsetAsync(vals, 0, sizeof(double) * r * c, stream), "cudaMemsetAsync MAT_GPU");
  }

  MAT_GPU(): n_rows(0), n_cols(0) {}

  // if columns are continuousm return inplace matrix
  MAT_GPU * cols(const unsigned int start_idx, const unsigned int end_idx) {
    MAT_GPU * col_mat = new MAT_GPU(n_rows, end_idx - start_idx + 1);
    col_mat->vals = this->vals + start_idx * n_rows;
    return col_mat;
  }

  MAT_GPU col(const unsigned int col_idx) {
    MAT_GPU col_mat = MAT_GPU(n_rows, 1);
    check_cuda(cudaMemcpy(col_mat.vals, this->vals+col_idx*n_rows, sizeof(double)*n_rows, cudaMemcpyDeviceToDevice), "cudaMemcopy sub matrix column");
    return col_mat;
  }

  // given a list of col_idx, return corresponding matrix
  MAT_GPU * cols(const unsigned int * col_idx, const unsigned int col_idx_size) {
    MAT_GPU * col_mat = new MAT_GPU(n_rows, col_idx_size);
    check_cuda(cudaMalloc((void**)&col_mat->vals, sizeof(double) * n_rows * col_idx_size), "malloc mat_gpu->cols");

    // malloc -- use streams??
    for (int c = 0; c < col_idx_size; ++c) {
      // copying the mat.col[col_idx[c]] -> col_mat.col[c] // each column is size n_rows
      check_cuda(cudaMemcpy(col_mat->vals+c*n_rows, this->vals+col_idx[c]*n_rows, sizeof(double)*n_rows, cudaMemcpyDeviceToDevice), "cudaMemcopy corresponding column");
    }
    return col_mat;
  }

  /**
   * @brief update MAT_GPU mat based on column index and size of column index
   * 
   * Given the updated values in MAT_GPU form, 
   * update the matrix values given the column indices and corresponding values
   * 
   * @param col_idx The indices of the columns we want to update
   * @param col_idx_size The length of the column indices so that we can iterate over
   * @param upd_mat The src matrix that we can to copy and update from
  */
  void colidx_based_update(const unsigned int * col_idx, const unsigned int col_idx_size, const MAT_GPU * upd_mat) {
    assert(col_idx_size == upd_mat->n_cols); // invalid update
    for (int c = 0; c < col_idx_size; ++c) {
      // printf("%d col_idx[%d]: %d\n", n_rows, c, col_idx[c]);
      check_cuda(cudaMemcpy(this->vals+col_idx[c]*n_rows, upd_mat->vals+c*n_rows, sizeof(double) * n_rows, cudaMemcpyDeviceToDevice), "memcpy colidx_based_update");
    }
  }

  /**
   * @brief update the column vector from a matrix given the column index, the mask, and the updated values
   * 
   * Needed to update columns from solving BPP based LS
   * 
   * @param col The column index
   * @param mask_idx The row_idx that we want to update
   * @param update_vals The values of the column vector
  */
  void masked_col_update(const unsigned int col, const unsigned int * masked_row_idx, const double * update_vals) {
    assert(col < n_cols); // col index to update should be within the n_cols
    dvec_update_only_mask(update_vals, this->vals+col*n_rows, masked_row_idx, n_rows);
  }

  /**
   * applies mask to MAT_GPU and return a new MAT_GPU
   * should only be called on matrices with single cols or rows
   * or square matrices
  */
  MAT_GPU apply_mask(const unsigned int * masked_idx) const {
    MAT_GPU masked_mat = MAT_GPU(n_rows, n_cols);
    if (n_rows == 1 || n_cols == 1) {
      size_t size = n_rows == 1 ? n_cols : n_rows;
      // for every idx in masekd_idx, fill with 0.0
      dvec_apply_mask(vals, masked_mat.vals, masked_idx, size);
    }
    else {
      assert(n_rows == n_cols);
      unsigned int * masked_idx_host = (unsigned int *) malloc(sizeof(unsigned int) * n_rows);
      check_cuda(cudaMemcpy(masked_idx_host, masked_idx, sizeof(unsigned int) * n_rows, cudaMemcpyDeviceToHost), "copy masked idx to host");
      // for every idx in masked_idx fill row and column with 0.0
      for (int col_idx = 0; col_idx < n_cols; ++col_idx) {
        if (masked_idx_host[col_idx] == 0) {
          value_dfill(masked_mat.vals+col_idx*n_rows, n_rows, 0.0);
        }
        else {
          dvec_apply_mask(vals+col_idx*n_rows, masked_mat.vals+col_idx*n_rows, masked_idx, n_rows);
        }
      }
      free(masked_idx_host);
    }
    return masked_mat;
  }

  UMAT_GPU operator>(double value) const {
    UMAT_GPU bool_mat = UMAT_GPU(n_rows, n_cols);
    check_cuda(cudaMalloc((void**)&bool_mat.vals, sizeof(unsigned int) * n_rows * n_cols), "malloc umat_gpu");
    // check_cuda(cudaMemset(bool_mat.vals, 0, sizeof(unsigned int) * n_rows * n_cols), "debugging operator");
    __compare_elements<GREATER, double, unsigned int>(this->vals, bool_mat.vals, n_rows * n_cols, value);
    return bool_mat;
  }
  UMAT_GPU operator==(double value) const {
    UMAT_GPU bool_mat = UMAT_GPU(n_rows, n_cols);
    check_cuda(cudaMalloc((void**)&bool_mat.vals, sizeof(unsigned int) * n_rows * n_cols), "malloc umat_gpu");
    __compare_elements<EQ, double, unsigned int>(this->vals, bool_mat.vals, n_rows * n_cols, value);
    return bool_mat;
  }
  UMAT_GPU operator<(double value) const {
    UMAT_GPU bool_mat = UMAT_GPU(n_rows, n_cols);
    check_cuda(cudaMalloc((void**)&bool_mat.vals, sizeof(unsigned int) * n_rows * n_cols), "malloc umat_gpu");
    __compare_elements<LESS, double, unsigned int>(this->vals, bool_mat.vals, n_rows * n_cols, value);
    return bool_mat;
  }
  ~MAT_GPU() {
    if (stream == NULL) {
      check_cuda(cudaFree(vals), "cudaFree destructor for MAT_GPU");
    } 
    else {
      check_cuda(cudaFreeAsync(vals, stream), "cudaFreeAsync destructor for MAT_GPU");  
    }
  }
};

template <typename T>
T* make_device_copy(T* vector, _IType n, std::string name) {
    T* d_vector = nullptr;
    check_cuda(cudaMalloc(&d_vector, sizeof(T) * n), "cudaMalloc " + name);
    check_cuda(cudaMemcpy(d_vector, vector, sizeof(T) * n, cudaMemcpyHostToDevice), "cudaMemcpy " + name);
    return d_vector;
};

// Hadamard update, i.e. x <-- x .* y
__global__ void hadamard_kernel(_FType* x, _FType* y, int n);
void vector_hadamard(_FType* x, _FType* y, int n);

void normalize_fm(MAT_GPU * fm, _FType * lambda);
void normalize_fm_cublas(MAT_GPU * fm, _FType * lambda);
void normalize_mat_cublas(int m, int n, double * mat_vals, double * lambda);

void mat_mat_mul(_FType* a, _FType* b, _FType* c, int m, int n, int k, double alpha = 1.0, double beta = 0.0);
void mat_vec_mul(_FType* a, _FType* b, _FType* c, int m, int n, double alpha = 1.0, double beta = 0.0);

MAT_GPU * init_mat_gpu(int m, int n);
void copy_mat_gpu(MAT_GPU * dest, const MAT_GPU * src);

double mat_trace_gpu(const MAT_GPU * mat);
void mat_add_diag_gpu(MAT_GPU * mat, double val);

void mat_cholesky_gpu(MAT_GPU * mat, bool is_lower=true);
void mat_cholesky_solve_gpu(const MAT_GPU * A, MAT_GPU * B, bool is_lower=true);

void copy_mat_gpu_lower(MAT_GPU * mat_l, MAT_GPU * mat);
void copy_mat_gpu_upper(MAT_GPU * mat_u, MAT_GPU * mat);

__global__ void __mat_transpose(double * _mat_t, double * _mat, int m, int n);
void mat_transpose_inplace(MAT_GPU * mat);

void free_mat_gpu(MAT_GPU * mat);
void free_umat_gpu(UMAT_GPU * mat);

__global__ void frob_norm_gpu(const double * v, int rows, int cols, double * norm);
__global__ void frob_norm_gpu(const double * v1, const double * v2, int rows, int cols, double * norm);

template <typename T> 
__global__ void reduce_sum_gpu(const T * v, int size, T * sum);
template <>
__global__ void reduce_sum_gpu(const double * v, int size, double * sum);


#endif // CUDA_UTILS_H_