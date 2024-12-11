#pragma once

#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cblas.h"

#include "cuda_runtime.h"
#include "blco.h"
#include "cuda_utils.h"
#include "cublas_operations.h"
#include "cassert"

void pseudoinverse_gpu(cusolverDnHandle_t cusolverHandle, cublasHandle_t cublasHandle,
    cudaStream_t stream, double * A, unsigned int n, double * work, unsigned int lwork, int* info, gesvdjInfo_t gesvd_info);
void solveSvdGemm(const MAT_GPU &A, const MAT_GPU &B, MAT_GPU &X);
void solveSvdGemm(cudaStream_t stream, cublasHandle_t cublasHandle, double * a_vals, double * b_vals, double * x_vals, int m, int n, int k);
void solveDownDate(
    cudaStream_t stream,
    cublasHandle_t cublasHandle,
    double * masked_gram_vals,
    size_t R, const unsigned int * mask, int num_masked_vars, double * masked_b_vals, double * h_x_diag);

void downdate_rank_matrix(cudaStream_t stream, cublasHandle_t cublasHandle, double * d_A, int rows, int cols, int idx_to_drop, double * h_x_diag);
void apply_mask_to_matrix(cudaStream_t stream, double * vo, unsigned int * vi, size_t size);
void apply_mask_to_gram_matrix(cudaStream_t stream, double * masked_gram_vals, int m, unsigned int * idx_to_mask);
void fill_diagonal_matrix(cudaStream_t stream, double * mat_vals, int r, int c, double val);
void add_rho_to_gram_mat(cudaStream_t stream, double * gram_vals, double * trace, int R);