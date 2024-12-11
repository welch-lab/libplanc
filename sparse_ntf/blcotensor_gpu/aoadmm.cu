#include "aoadmm.h"
#include "cublas_operations.h"

__global__ void __compute_t_H_kernel(double * t_fm_vals, double * mttkrp_vals,
  double * fm_vals, double * dual_vals, double rho, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    t_fm_vals[idx] = mttkrp_vals[idx] + rho * (fm_vals[idx] + dual_vals[idx]);
  }
}

__global__ void __compute_U_kernel(double * dual_vals, double * fm_vals, double * t_fm_vals, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dual_vals[idx] += fm_vals[idx] - t_fm_vals[idx];
  }
}

__global__ void __compute_diff_squared_kernel(double * fm_vals, double * prev_vals, double * t_fm_vals, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double diff1 = fm_vals[idx] - prev_vals[idx]; 
    double diff2 = fm_vals[idx] - t_fm_vals[idx];
    prev_vals[idx] = diff1;
    t_fm_vals[idx] = diff2;
  }
}

__global__ void __compute_fm_with_projection_kernel(double * fm_vals, double * t_fm_vals, double * dual_vals, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    fm_vals[idx] = (t_fm_vals[idx] - dual_vals[idx] >= 0.0) * (t_fm_vals[idx] - dual_vals[idx]);
  }
}

void compute_t_H(cudaStream_t stream, double * t_fm_vals, double * mttkrp_vals,
  double * fm_vals, double * dual_vals, double rho, size_t n) {
  __compute_t_H_kernel<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(t_fm_vals, mttkrp_vals, fm_vals, dual_vals, rho, n);
}

void compute_U(cudaStream_t stream, double * dual_vals, double * fm_vals, double * t_fm_vals, size_t n) {
  __compute_U_kernel<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(dual_vals, fm_vals, t_fm_vals, n);
}

void compute_fm_with_projection(cudaStream_t stream, double * fm_vals, double * t_fm_vals, double * dual_vals, size_t n) {
  __compute_fm_with_projection_kernel<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(fm_vals, t_fm_vals, dual_vals, n);
}
void compute_diff_squared(cudaStream_t stream, double * fm_vals, double * prev_vals, double * t_fm_vals, size_t n) {
  __compute_diff_squared_kernel<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(fm_vals, prev_vals, t_fm_vals, n);
}


/**
 * "pure" kernel that computes admm
*/
void admm_gpu_kernel(
  double * gram_vals, 
  double rho, 
  double * primal_vals, // fm
  double * mttkrp_vals, // necessay to compute 
  double * prev_fac_vals, // necessary to store previous vals, also used to store diff
  double * dual_vals, // dual variables -- same size as primals vals
  double * tilde_primal_vals,
  int start_row, int end_row, int rank, int admm_iter, double tolerance, double * admm_condition_vals, int stream_id, cudaStream_t stream)
{
    bool stop_iter = false;
    double alpha, beta;

    size_t _size = (end_row - start_row) * rank * sizeof(double);
    size_t num_rows = end_row - start_row;
   
    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "create cublas handle");

    cublasSetStream(handle, stream);

    cusolverDnHandle_t cusolver = NULL;
    cusolverDnCreate(&cusolver);
    cusolverDnSetStream(cusolver, stream);

    // printf("Running admm_gpu_kernel, num_rows: %d\n", num_rows);
    for (int i = 0; i < admm_iter && !stop_iter; ++i) {
      // primal_vals is always up-to-date
      // this will write on original fm space
      check_cuda(cudaMemcpyAsync(prev_fac_vals, primal_vals, _size, cudaMemcpyDeviceToDevice, stream), "prev_fac <- primal_vals");
      alpha = 1.0; beta = 1.0;

      compute_t_H(stream, tilde_primal_vals, mttkrp_vals, primal_vals, dual_vals, rho, num_rows*rank);

      beta = 0.0; // tilde_primal_vals = gram_vals * tilde_primal_vals
      check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        num_rows, rank, rank, &alpha, tilde_primal_vals,
        num_rows, gram_vals, rank, &beta, tilde_primal_vals, num_rows
      ), "dgemm solve using explicit inverse");

      compute_fm_with_projection(stream, primal_vals, tilde_primal_vals, dual_vals, num_rows * rank);
      compute_U(stream, dual_vals, primal_vals, tilde_primal_vals, num_rows * rank);

      // compute the residuals..
      double * fnorm = admm_condition_vals + 4 * stream_id + 0;
      double * dnorm =  admm_condition_vals + 4 * stream_id + 1;
      double * fres = admm_condition_vals + 4 * stream_id + 2;
      double * dres = admm_condition_vals + 4 * stream_id + 3;
      
      compute_diff_squared(stream, primal_vals, prev_fac_vals, tilde_primal_vals, num_rows * rank);

      check_cublas(cublasDnrm2(handle, num_rows * rank, primal_vals, 1, fnorm), "cublas Dnrm2 - fnorm");
      check_cublas(cublasDnrm2(handle, num_rows * rank, dual_vals, 1, dnorm), "cublas Dnrm2 - dnorm");
      check_cublas(cublasDnrm2(handle, num_rows * rank, prev_fac_vals, 1, fres), "cublas Dnrm2 - fres");
      check_cublas(cublasDnrm2(handle, num_rows * rank, tilde_primal_vals, 1, dres), "cublas Dnrm24");
      cudaDeviceSynchronize();

      // printf("fnorm: %f dnorm: %f fres: %f dres: %f\n", *fnorm, *dnorm, *fres, *dres);

      if (*dres < (tolerance * (*fnorm)) && *fres < (tolerance * (*dnorm))) {
        // printf("admm iteration terminated with iter: %d\n", i);
        stop_iter = true;
      }
    } // end of admm iteration

    cublasDestroy(handle);
    cusolverDnDestroy(cusolver);
}

void aoadmm_blocked_update(
  MAT_GPU * fm, MAT_GPU * aux_fm, MAT_GPU * o_mttkrp_gpu, 
  const MAT_GPU * gram, int block_size, int admm_iter, double tolerance, int num_streams) {

  CuBlasOperations cublasOps[num_streams];
  cudaEvent_t start;
  cudaEvent_t stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float milliseconds = 0;
  
  cudaEventRecord(start);

  int m = fm->n_rows;
  int n = fm->n_cols;

  double rho = mat_trace_gpu(gram) / n;
  rho = (rho > 0) ? rho : 0.01;

  MAT_GPU * tempgram = init_mat_gpu(n, n);
  copy_mat_gpu(tempgram, gram);  
  mat_add_diag_gpu(tempgram, rho);

  MAT_GPU * L = init_mat_gpu(n, n);
  copy_mat_gpu(L, tempgram);
  mat_cholesky_gpu(L); // L is factorized

  // Create explicit (L.T)^-1 * L^-1
  double * explicit_inverse;
  cudaMalloc(&explicit_inverse, sizeof(double) * n * n);
  cudaMemset(explicit_inverse, 0, n * n * sizeof(double));
  fill_diagonal_matrix(0, explicit_inverse, n, n, 1.0); // fill diagonal with 1.0

  cusolverDnHandle_t cusolver = NULL;
  cusolverDnCreate(&cusolver);

  int *devInfo;
  cudaMalloc(&devInfo, sizeof(int));

  check_cusolver(
    cusolverDnDpotrs(cusolver, CUBLAS_FILL_MODE_LOWER, n, 
    n, L->vals, n, explicit_inverse, n, devInfo),
    "cusolver cholesky solve"
  ); // compute explicit inverse

  cudaFree(devInfo); // assume nothing goes wrong
  cusolverDnDestroy(cusolver);
  // Create cuda streams streams

  // Malloc pinned memory on host for streams
  // int block_size = 10000;
  size_t _size = block_size * n * sizeof(double) * num_streams;

  double * prev_fac_vals; // also used to store diff vals
  double * aux_fm_vals = aux_fm->vals; // dual variables U
  double * tilde_h_vals; // tilde_H
  double * admm_condition_vals; // fres, dres, fnorm, dnorm

  // malloc on host
  check_cuda(cudaMallocHost(&admm_condition_vals, sizeof(double) * 4 * num_streams), "cudaMallocHost admm_condition_vals");

  // pass in explicit inverse as gram_vals
  double * gram_vals = explicit_inverse;
  
  // stream 0 as default for now
  check_cuda(cudaMallocAsync(&prev_fac_vals, _size, 0), "cudaMallocAsync prev_fac_vals");
  check_cuda(cudaMallocAsync(&tilde_h_vals, _size, 0), "cudaMallocAsync tilde_h_vals");

  int rank = n;
  int stream_id = 0;

  for (int i = 0; i < m; i += block_size) {

    int end = i + block_size;
    if (end > m) {
        end = m;
    }
    size_t offset = i * rank;
    size_t stream_offset = stream_id * block_size * rank;
    double * primal_vals = fm->vals + offset;
  
    // Process rows from i to end-1
    printf("Processing rows %d to %d stream_id: %d\n", i, end - 1, stream_id);
    admm_gpu_kernel(
      gram_vals, rho, 
      primal_vals,
      o_mttkrp_gpu->vals + offset, 
      prev_fac_vals + stream_offset, // using gpu temp
      aux_fm_vals + offset, 
      tilde_h_vals + stream_offset, // using gpu temp
      i, end, rank, admm_iter, tolerance, admm_condition_vals, stream_id, cublasOps[stream_id].stream);

    // keep track of stream id
    stream_id += 1;
    stream_id %= num_streams;
  }
  // normalize dual fm
  double * temp_lambda; 
  cudaMalloc((void**)&temp_lambda, sizeof(double) * n); // dummy lambda

  normalize_fm_cublas(aux_fm, temp_lambda);
  cudaFree(temp_lambda);

  cudaFreeAsync(prev_fac_vals, 0);
  cudaFreeAsync(tilde_h_vals, 0);
  cudaFreeHost(admm_condition_vals);
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Inner Elapsed time for solving for blocked ADMM: " << milliseconds << " ms\n";

}

void aoadmm_update(
  MAT_GPU * fm, MAT_GPU * aux_fm, MAT_GPU * o_mttkrp_gpu, 
  const MAT_GPU * gram, int admm_iter, double tolerance) {
  bool stop_iter = false;

  int m = fm->n_rows;
  int n = fm->n_cols;

  double _alpha = mat_trace_gpu(gram) / n;
  _alpha = (_alpha > 0) ? _alpha : 0.01;

  MAT_GPU * tempgram = init_mat_gpu(n, n);
  copy_mat_gpu(tempgram, gram);  
  mat_add_diag_gpu(tempgram, _alpha);

  MAT_GPU * L = init_mat_gpu(n, n);
  copy_mat_gpu(L, tempgram);
  mat_cholesky_gpu(L); // L is factorized
  // mat_cholesky_solve_gpu(L, fm, );
  MAT_GPU * prev_fac = init_mat_gpu(m, n);
  MAT_GPU * update_fm = init_mat_gpu(m, n);
  MAT_GPU * temp_fm = init_mat_gpu(m, n);
  MAT_GPU * aux_fm_t = init_mat_gpu(n, m);
  
  copy_mat_gpu(update_fm, fm);
  copy_mat_gpu(prev_fac, fm);

  // For gpu kernel calls
  int num_tblocks = (m * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int fm_size = m * n;

  //--- DEBUG compute norms and stuff delete later
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "create cublas handle");

  double alpha, beta;
  for (int i = 0; i < admm_iter && !stop_iter; ++i) {
    copy_mat_gpu(prev_fac, update_fm);
    //----- aux_fm_t = 1.0 * updated_fm.T + 1.0 * aux_fm.T
    //----- aux_fm_t = mttkrp + alpha * aux_fm_t
    alpha = 1.0; beta = 1.0;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, 
      &alpha, update_fm->vals, m, &beta, aux_fm->vals, m, 
      temp_fm->vals, m), "temp_fm <- 1.0 * updated_fm + 1.0 * aux_fm");

    beta = _alpha;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, 
      &alpha, o_mttkrp_gpu->vals, m, &beta, temp_fm->vals, m, 
      temp_fm->vals, m), "temp_fm <- 1.0 * mttkrp + alpha * temp_fm");

    beta = 0.0;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, 
      &alpha, temp_fm->vals, m, &beta, aux_fm_t->vals, n, 
      aux_fm_t->vals, n), "aux_fm_t <- 1.0 * aux_fm.T");

    mat_cholesky_solve_gpu(L, aux_fm_t, true);

    cudaDeviceSynchronize();

    beta = -1.0;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, 
      &alpha, aux_fm_t->vals, n, &beta, aux_fm->vals, m, 
      update_fm->vals, m), "updated_fm <- 1.0 * aux_fm_t.T - 1.0 * aux_fm");

    __apply_threshold<<<num_tblocks, BLOCK_SIZE>>>(update_fm->vals, fm_size, 0.0, 0.0);
    cudaDeviceSynchronize();

    beta = 1.0;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, 
      &alpha, aux_fm->vals, m, &beta, update_fm->vals, m, 
      aux_fm->vals, m), "aux_fm <- 1.0 * aux_fm + 1.0 * updated_fm");

    beta = -1.0;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, 
      &alpha, aux_fm->vals, m, &beta, aux_fm_t->vals, n, 
      aux_fm->vals, m), "aux_fm <- 1.0 * aux_fm - 1.0 * aux_fm_t.T");

    // aux_fm = aux_fm + updated_fac - aux_fm_t

    //--- Compute fm norm and dual variable norm    
    double fnorm;
    double dnorm;

    check_cublas(cublasDnrm2(handle, fm_size, update_fm->vals, 1, &fnorm), "cublas Dnrm2 - fnorm");
    cudaDeviceSynchronize();

    check_cublas(cublasDnrm2(handle, fm_size, aux_fm->vals, 1, &dnorm), "cublas Dnrm2 - dnorm");
    cudaDeviceSynchronize();

    //--- END

    //--- Compute f_res and d_res    
    double fres;
    double dres;

    double * diff; // used for diffs
    cudaMalloc((void**)&diff, sizeof(double) * fm_size);
    cudaDeviceSynchronize();
    cudaMemcpy(diff, update_fm->vals, sizeof(double) * fm_size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    dvec_sub(diff, prev_fac->vals, diff, fm_size);
    cudaDeviceSynchronize();

    check_cublas(cublasDnrm2(handle, fm_size, diff, 1, &fres), "cublas Dnrm2 - fres");
    cudaDeviceSynchronize();

    beta = -1.0;
    check_cublas(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, 
      &alpha, update_fm->vals, m, &beta, aux_fm_t->vals, n, 
      aux_fm_t->vals, n), "aux_fm_t <- 1.0 * update_fm.T - 1.0 * aux_fm_t");
    check_cublas(cublasDnrm2(handle, fm_size, aux_fm_t->vals, 1, &dres), "cublas Dnrm24");
    cudaDeviceSynchronize();

    // printf("fnorm: %f dnorm: %f fres: %f dres: %f\n", fnorm, dnorm, fres, dres);
    cudaFree(diff);
    //--- END

    // Check convergence
    if (dres < (tolerance * fnorm) && fres < (tolerance * dnorm)) {
      // printf("admm iteration terminated with iter: %d\n", i);
      stop_iter = true;
    }
  } // end of admm iteration

  // Update fm
  copy_mat_gpu(fm, update_fm);
  cudaDeviceSynchronize();
  
  //--- normalize dual variable aux_fm
  double * temp_lambda; 
  cudaMalloc((void**)&temp_lambda, sizeof(double) * n); // dummy lambda
  
  normalize_fm_cublas(aux_fm, temp_lambda);
  cudaFree(temp_lambda);
  
  // check_cuda(cudaGetLastError(), "normalize aux_fm");

  //---

  free_mat_gpu(prev_fac);
  free_mat_gpu(update_fm);
  free_mat_gpu(aux_fm_t);
  free_mat_gpu(temp_fm);

  free_mat_gpu(L);
  free_mat_gpu(tempgram);

  cublasDestroy(handle);
}
