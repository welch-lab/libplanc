#ifndef NTF_AUNTF_GPU_HPP_
#define NTF_AUNTF_GPU_HPP_


#include <armadillo>
#include <vector>
#include "common/ncpfactors.hpp"
#include "common/ntf_utils.hpp"
#include "common/tensor.hpp"
#include "common/sparse_tensor.hpp"

#include "blco.h"
#include "cuda_utils.h"

namespace planc {

#define TENSOR_DIM (m_input_tensor.dimensions())
#define TENSOR_NUMEL (m_input_tensor.numel())

template <class T>
class AUNTF_GPU {
  protected:
    planc::NCPFactors factors_host;
    MAT * mttkrp_mat_host; // only one matrix which we'll reuse for all modes
    MAT * gram_mat_host; // gram matrix for other modes

    // GPU stuff
    MAT_GPU** factors_gpu;
    MAT_GPU** mttkrp_mat_gpu;
    MAT_GPU* gram_mat_gpu;
    double* lambda_gpu;
    virtual void update_gpu(const int mode, const MAT_GPU * gram, MAT_GPU ** factors, MAT_GPU * o_mttkrp) = 0;

  private:
    const T &m_input_tensor;
    int m_num_it;
    int m_current_it;
    bool m_compute_error;

    const int m_low_rank_k;
    MAT *ncp_krp; // 
    const algotype m_updalgo;
    // planc::Tensor *lowranktensor;
    double m_rel_error;
    double m_normA;

  public:
  AUNTF_GPU(const T &i_tensor, const int i_k, algotype i_algo)
    : factors_host(i_tensor.dimensions(), i_k, false),
    m_input_tensor(i_tensor),
    m_low_rank_k(i_k),
    m_updalgo(i_algo) {
      factors_host.normalize();
      int longest_mode = arma::max(i_tensor.dimensions());
      // Allocate mttkrp_mat_host accordingly
      // mttkrp_mat_host = new MAT(i_k, longest_mode); // already transposed
      mttkrp_mat_host = new MAT[i_tensor.modes()]; // already transposed
      for (int m = 0; m < factors_host.modes(); ++m) {
        mttkrp_mat_host[m].zeros(TENSOR_DIM[m], i_k);
        size_t mttkrp_size = mttkrp_mat_host[m].n_cols * mttkrp_mat_host[m].n_rows * sizeof(double);
        check_cuda(cudaHostRegister(mttkrp_mat_host[m].memptr(), mttkrp_size, cudaHostRegisterDefault), "pin mttkrp_mat_host memory on host");
      }
      gram_mat_host = new MAT(i_k, i_k);
      size_t gram_size = i_k * i_k * sizeof(double);

      // Registor host memory for GPU
      check_cuda(cudaHostRegister(gram_mat_host->memptr(), gram_size, cudaHostRegisterDefault), "pin gram_mat_host memory on host");
      for (int m = 0; m < factors_host.modes(); ++m) {
        size_t fm_size = factors_host.factor(m).n_cols * factors_host.factor(m).n_rows * sizeof(double);
        check_cuda(cudaHostRegister(factors_host.factor(m).memptr(), fm_size, cudaHostRegisterDefault), "pin factors_host memoy on host");
      };
      check_cuda(cudaHostRegister(factors_host.lambda().memptr(), i_k * sizeof(double), cudaHostRegisterDefault), "pin lambda_host memory on host");

      // pin host memory for cuda
      // get largets dim
      mttkrp_mat_gpu = send_mats_to_gpu(mttkrp_mat_host, factors_host.modes());
      factors_gpu = send_mats_to_gpu(factors_host.factors(), factors_host.modes());
      gram_mat_gpu = send_mat_to_gpu(gram_mat_host);
      lambda_gpu = make_device_copy(factors_host.lambda().memptr(), i_k, "lambda_gpu");

      // For performance evaluation
      // m_compute_error = false;
      // m_num_it = 20;

  }

  ~AUNTF_GPU() {
    for (int i = 0; i < m_input_tensor.modes(); i++) {
      mttkrp_mat_host[i].clear();
    }
    delete[] mttkrp_mat_host;
    // Clear cuda stuff too
  }

  NCPFactors &ncp_factors() { return factors_host; }
  double current_error() const { return this->m_rel_error; }
  void compute_error(bool i_error) { this->m_compute_error = i_error; }
  void num_it(const int i_n) { this->m_num_it = i_n; }
  void computeSparseNTF() {
    double wtime;
    double wtime_gram;
    double wtime_mttkrp;
    double wtime_update;
    double wtime_update_fm;

    for (m_current_it = 0; m_current_it < m_num_it; ++m_current_it) {
      INFO << "iter::" << this->m_current_it << std::endl;
      for (int j = 0; j < this->m_input_tensor.modes(); ++j) {
        wtime = omp_get_wtime();
        gram_leave_out_one_gpu(j, factors_host.modes(), factors_gpu, gram_mat_gpu);
        wtime_gram = omp_get_wtime() - wtime;

        wtime = omp_get_wtime();
        m_input_tensor.mttkrp_gpu(j, factors_gpu, mttkrp_mat_gpu[j]);
        wtime_mttkrp = omp_get_wtime() - wtime;

        // update kernel

        wtime = omp_get_wtime();
        update_gpu(j, gram_mat_gpu, factors_gpu, mttkrp_mat_gpu[j]);
        wtime_update = omp_get_wtime() - wtime;

        // factors_gpu[j] is updated, update lambda_gpu accordingly
        wtime = omp_get_wtime();
        // normalize_fm(factors_gpu[j], lambda_gpu);
        normalize_fm_cublas(factors_gpu[j], lambda_gpu);
        wtime_update_fm = omp_get_wtime() - wtime;
        // debug
        // int rank = factors_gpu[0]->n_cols;
        // cudaMemcpy(factors_host.m_lambda.memptr(), lambda_gpu, sizeof(double) * rank, cudaMemcpyDeviceToHost);
        // INFO << factors_host.m_lambda << std::endl;
        // exit(0);               
        printf("[PERF-mode, gram, mttkrp, update, update_fm]\t%d\t%f\t%f\t%f\t%f\n", j, wtime_gram, wtime_mttkrp, wtime_update, wtime_update_fm);
      }
      if (m_compute_error) {
        // compute mttkrp for last mode
        int last_mode = this->m_input_tensor.modes() - 1;
        send_mat_to_host(mttkrp_mat_gpu[last_mode], &mttkrp_mat_host[last_mode]);
        // INFO << arma::norm(mttkrp_mat_host[last_mode], "fro") << std::endl;
        for (int m = 0; m < this->m_input_tensor.modes(); ++m) {
          send_mat_to_host(factors_gpu[m], &factors_host.factor(m));
        }
        int rank = factors_gpu[last_mode]->n_cols;
        cudaMemcpy(factors_host.m_lambda.memptr(), lambda_gpu, sizeof(double) * rank, cudaMemcpyDeviceToHost);
        // INFO << factors_host.m_lambda << std::endl;
        double temp_err = m_input_tensor.err(factors_host, mttkrp_mat_host[last_mode], last_mode);
        this->m_rel_error = temp_err;
        INFO << "relative_error @it " << this->m_current_it
            << "=" << temp_err << std::endl;
      }
    }
  }
}; // class AUNTF_GPU
} // namespace planc

#endif