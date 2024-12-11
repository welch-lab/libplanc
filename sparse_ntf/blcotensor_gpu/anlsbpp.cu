#include "anlsbpp.h"
#include "vector_ops.h"
#include "matrix_ops.h"

#define CHUNK_SIZE 100
#define UUMAT arma::Mat<uint32_t>
#define UUVEC arma::Row<uint32_t>
#define DEBUG 0
/**
 * @brief Computes the NLS-BPP update AX=B where X has non-negativity constraint
 * 
 * The function attempts to solve constrained AX=B
 * where gram is A, o_mttkrp_gpu is B and final product of X is copied to fm
*/
void anlsbpp_update(MAT_GPU * fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram) {
  MAT_GPU * Y = init_mat_gpu(o_mttkrp_gpu->n_rows, o_mttkrp_gpu->n_cols);

  MAT_GPU * X = new MAT_GPU(); // Create an empty shell
  X->n_rows = fm->n_rows;
  X->n_cols = fm->n_cols;
  X->vals = fm->vals; // Simple just point to fm

  // MAT_GPU * X = init_mat_gpu(o_mttkrp_gpu->n_rows, o_mttkrp_gpu->n_cols);
  cudaMemset(X->vals, 0.0, sizeof(double) * X->n_rows * X->n_cols);

  int nrhs = o_mttkrp_gpu->n_cols;
  int RANK = o_mttkrp_gpu->n_rows;
  unsigned int k = nrhs; // rank
  unsigned int n = RANK;

  mat_mat_mul(gram->vals, X->vals, Y->vals, gram->n_rows, gram->n_cols, X->n_cols, 1.0, 0.0);

  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "create cublas handle");

  double alpha = 1.0;
  double beta = -1.0;

  check_cublas(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, Y->n_rows, Y->n_cols, 
  &alpha, Y->vals, Y->n_rows, &beta, o_mttkrp_gpu->vals, o_mttkrp_gpu->n_rows, 
  Y->vals, Y->n_rows), "Y <- 1.0 * Y - 1.0 * o_mttkrp_gpu");

  cublasDestroy(handle);

  UMAT_GPU PassiveSet = *X > 0;
  UMAT_GPU LO = * Y < 0;
  UMAT_GPU RO = PassiveSet == 0;

  UMAT_GPU negX = *X < 0;

  UMAT_GPU NonOptSet = LO % RO;
  UMAT_GPU InfeaSet = negX % PassiveSet;

  UVEC_GPU P = UVEC_GPU(k);
  UVEC_GPU Ninf = UVEC_GPU(k);
  unsigned int pbar = 3;
  P.fill(pbar);
  Ninf.fill(n+1); // n+1

  UVEC_GPU SumNonOptSet = NonOptSet.sum();
  UVEC_GPU SumInfeaSet = InfeaSet.sum();

  UVEC_GPU NotGood = SumNonOptSet + SumInfeaSet; // need to add InfeaSet.sum();
  UVEC_GPU NotOptCols = NotGood > 0;

  unsigned int numNonOptCols = NotOptCols.sum();
  // INFO << "numNonOptCols: " << NotOptCols.sum() << std::endl;

  unsigned int MAX_ITERATIONS = nrhs * 5;
  // unsigned int MAX_ITERATIONS = 3;
  bool success = true;

  UVEC_GPU Cols1 = UVEC_GPU(k);
  UVEC_GPU Cols2 = UVEC_GPU(k);
  UVEC_GPU Cols3 = UVEC_GPU(k);
  
  UMAT_GPU PSetBits = NonOptSet;
  UMAT_GPU POffBits = InfeaSet;

  // For debugging

  UUVEC debug_uvec = arma::zeros<UUVEC>(X->n_cols);

  unsigned int iter = 0;
  while(numNonOptCols > 0) {
    iter++;
    if ((MAX_ITERATIONS > 0) && (iter > MAX_ITERATIONS)) {
      success = false;
      break;
    }
    Cols1 = NotOptCols % (NotGood < Ninf);
    Cols2 = NotOptCols % (NotGood >= Ninf) % (P >= 1);
    if (!Cols1.empty()) {
      // INFO << "Cols1 not empty" << "\n";
      P.idx_based_fill(Cols1, pbar);
      Ninf.idx_based_copy(Cols1, NotGood);

      PSetBits = NonOptSet;
      PSetBits.rowwise_mult(Cols1); // UMAT_GPU.rowsize_mult(UVEC_GPU&)

      PassiveSet.idx_based_fill(PSetBits, 1u);

      POffBits = InfeaSet;
      POffBits.rowwise_mult(Cols1);
      PassiveSet.idx_based_fill(POffBits, 0u);
    }

    if (!Cols2.empty()) {
      // INFO << "Cols2 not empty" << "\n";
      P.idx_based_sub(Cols2, 1);
      PSetBits = NonOptSet;

      PSetBits.rowwise_mult(Cols2); // UMAT_GPU.rowsize_mult(UVEC_GPU&)
      PassiveSet.idx_based_fill(PSetBits, 1u);

      POffBits = InfeaSet;
      POffBits.rowwise_mult(Cols2);
      PassiveSet.idx_based_fill(POffBits, 0u);
    }
    if (!Cols3.empty()) {
      INFO << "Cols3 not empty" << "\n";
    }
    // It should only try to solve Ax=b for columns that have nonOptCols
    cudaMemcpy(debug_uvec.memptr(), NotOptCols.vals, sizeof(uint32_t) * NotGood.size, cudaMemcpyDeviceToHost);
    for (int col_idx = 0; col_idx < nrhs; ++col_idx) {
      if (debug_uvec[col_idx] == 1) {
        MAT_GPU MASKED_A = gram->apply_mask(PassiveSet.vals + col_idx * RANK);
        MAT_GPU b = o_mttkrp_gpu->col(col_idx);
        MAT_GPU MASKED_b = b.apply_mask(PassiveSet.vals + col_idx * RANK);

#if DEBUG==1 // for debugging purposes
        if (col_idx == 1) {
          send_mat_to_host(&MASKED_A, &debug_gram);
          INFO << "MASKED_A\n" << debug_gram << std::endl; 

          send_mat_to_host(&MASKED_b, &debug_vector);
          INFO << "MASKED_b\n" << debug_vector << std::endl; 
        }
#endif

        MAT_GPU MASKED_X(MASKED_b.n_rows, MASKED_b.n_cols);
        solveSvdGemm(MASKED_A, MASKED_b, MASKED_X);

#if DEBUG==1 // for debugging purposes
        if (col_idx == 1) {
          send_mat_to_host(&MASKED_X, &debug_vector);
          INFO << "MASKED_X\n" << debug_vector << std::endl; 
        }
#endif

        X->masked_col_update(
        // Update X one column vector at a time
          col_idx,
          PassiveSet.vals + col_idx * RANK, 
          MASKED_X.vals);

        // Update Y here
      }
    } // Update this->X complete

    mat_mat_mul(gram->vals, X->vals, Y->vals, gram->n_rows, gram->n_cols, X->n_cols, 1.0, 0.0);

    cublasHandle_t handle2;
    check_cublas(cublasCreate(&handle2), "create cublas handle2");

    double alpha = 1.0;
    double beta = -1.0;

    check_cublas(cublasDgeam(handle2, CUBLAS_OP_N, CUBLAS_OP_N, Y->n_rows, Y->n_cols, 
    &alpha, Y->vals, Y->n_rows, &beta, o_mttkrp_gpu->vals, o_mttkrp_gpu->n_rows, 
    Y->vals, Y->n_rows), "Y <- 1.0 * gram * X - 1.0 * o_mttkrp_gpu");

    cublasDestroy(handle2);

    // Update Y all at once
    fixAbsNumericalError_gpu(X->vals, 1e-12, 0.0, X->n_cols * X->n_rows);
    fixAbsNumericalError_gpu(Y->vals, 1e-12, 0.0, Y->n_cols * Y->n_rows);
    
    NonOptSet = (*Y < 0) % (PassiveSet == 0);
    InfeaSet = (*X < 0) % PassiveSet;

    NotGood = NonOptSet.sum() + InfeaSet.sum();

    // cudaMemcpy(debug_uvec.memptr(), NotGood.vals, sizeof(uint32_t) * NotGood.size, cudaMemcpyDeviceToHost);
    // INFO << "NotGood--inner\n" << debug_uvec << "\n";

    NotOptCols = NotGood > 0;
    numNonOptCols = NotOptCols.sum();
    // printf("numNonOptCols.sum(): %d\n", numNonOptCols);
  } // End of BPP iteration

  free_mat_gpu(Y);
}

/* PLACEHOLDER IMPLEMENTATION TO TEST API CALLS 
void anlsbpp_update(MAT_GPU * fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram) {
  // printf("fm size: %d %d\n", fm->n_rows, fm->n_cols);
  // printf("o_mttkrp_gpu size: %d %d\n", o_mttkrp_gpu->n_rows, o_mttkrp_gpu->n_cols);
  // This is basically same as UCP
  
  unsigned int col1_idx[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101 };
  unsigned int col2_idx[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100 };

  unsigned int col1_idx_size = 0;
  unsigned int col2_idx_size = 0;

  MAT_GPU * tempgram = init_mat_gpu(gram->n_rows, gram->n_cols);
  copy_mat_gpu(tempgram, gram);  
  mat_cholesky_gpu(tempgram);

  for (int i = 0; i < fm->n_cols; ++i) { // 10
    if (i % 2 == 0) col2_idx_size+=1;
    else col1_idx_size+=1;
  }
  // printf("col1_idx_size: %d col2_idx_size: %d\n", col1_idx_size, col2_idx_size);
  if (col1_idx_size > 0) {
    MAT_GPU * o_mttkrp_gpu_1 = o_mttkrp_gpu->cols(col1_idx, col1_idx_size);
    mat_cholesky_solve_gpu(tempgram, o_mttkrp_gpu_1);
    fm->colidx_based_update(col1_idx, col1_idx_size, o_mttkrp_gpu_1);
    free_mat_on_gpu(o_mttkrp_gpu_1);
  }
  if (col2_idx_size > 0) {
    MAT_GPU * o_mttkrp_gpu_2 = o_mttkrp_gpu->cols(col2_idx, col2_idx_size);
    mat_cholesky_solve_gpu(tempgram, o_mttkrp_gpu_2);
    fm->colidx_based_update(col2_idx, col2_idx_size, o_mttkrp_gpu_2);
    free_mat_on_gpu(o_mttkrp_gpu_2);
  }
  // update fm row

  free_mat_on_gpu(tempgram);

  // update columns of fm accordingly
}
*/
