#include "nmf.hpp"
#include <stdio.h>
#include <string>
#include "parsecommandline.hpp"
#include "utils.hpp"
#include "aoadmm.hpp"
#include "bppnmf.hpp"
#include "hals.hpp"
#include "mu.hpp"
#include "gnsym.hpp"

namespace planc {

class NMFDriver {
 private:
  int m_argc;
  char **m_argv;
  int m_k;
  arma::uword m_m, m_n;
  std::string m_Afile_name;
  std::string m_outputfile_name;
  std::string m_w_init_file_name;
  std::string m_h_init_file_name;
  int m_num_it;
  arma::fvec m_regW;
  arma::fvec m_regH;
  double m_symm_reg;
  int m_symm_flag;
  bool m_adj_rand;
  algotype m_nmfalgo;
  double m_sparsity;
  unsigned int m_compute_error;
  normtype m_input_normalization;
  int m_max_luciters;
  int m_initseed;

  // Variables for creating random matrix
  static const int kW_seed_idx = 1210873;
  static const int kprimeoffset = 17;
#ifdef BUILD_SPARSE
  static const int kalpha = 5;
  static const int kbeta = 10;
#else
  static const int kalpha = 1;
  static const int kbeta = 0;
#endif

  template <class NMFTYPE>
  void callNMF() {
#ifdef BUILD_SPARSE
    arma::sp_mat A;
#else
    arma::mat A;
#endif

    // Generate/Read data matrix
    double t2;
    if (!this->m_Afile_name.empty() ) {
      tic();
#ifdef BUILD_SPARSE
      A.load(this->m_Afile_name, arma::coord_ascii);
#else
      A.load(this->m_Afile_name);
#endif
      t2 = toc();
      INFO << "Successfully loaded input matrix A " << PRINTMATINFO(A)
         << "(" << t2 << " s)" << std::endl;
      this->m_m = A.n_rows;
      this->m_n = A.n_cols;
    } else {
      arma::arma_rng::set_seed(this->kW_seed_idx);
      std::string rand_prefix("rand_");
      std::string type = this->m_Afile_name.substr(rand_prefix.size());
      assert(type == "normal" || type == "lowrank" || type == "uniform");
      tic();
#ifdef BUILD_SPARSE
      if (type == "uniform") {
        if (this->m_symm_flag) {
          double sp = 0.5 * this->m_sparsity;
          A = arma::sprandu<arma::sp_mat>(this->m_m, this->m_n, sp);
          A = 0.5 * (A + A.t());
        } else {
          A = arma::sprandu<arma::sp_mat>(this->m_m, this->m_n,
              this->m_sparsity);
        }
      } else if (type == "normal") {
        if (this->m_symm_flag) {
          double sp = 0.5 * this->m_sparsity;
          A = arma::sprandn<arma::sp_mat>(this->m_m, this->m_n, sp);
          A = 0.5 * (A + A.t());
        } else {
          A = arma::sprandn<arma::sp_mat>(this->m_m, this->m_n,
              this->m_sparsity);
        }
      } else if (type == "lowrank") {
        if (this->m_symm_flag) {
          double sp = 0.5 * this->m_sparsity;
          arma::sp_mat mask = arma::sprandu<arma::sp_mat>(this->m_m, this->m_n,
                          sp);
          mask = 0.5 * (mask + mask.t());
          mask = arma::spones(mask);
          arma::mat Wtrue = arma::randu(this->m_m, this->m_k);
          A = arma::sp_mat(mask % (Wtrue * Wtrue.t()));

          // Free auxiliary space
          Wtrue.clear();
          mask.clear();
        } else {
          arma::sp_mat mask = arma::sprandu<arma::sp_mat>(this->m_m, this->m_n,
                          this->m_sparsity);
          mask = arma::spones(mask);
          arma::mat Wtrue = arma::randu(this->m_m, this->m_k);
          arma::mat Htrue = arma::randu(this->m_k, this->m_n);
          A = arma::sp_mat(mask % (Wtrue * Htrue));

          // Free auxiliary space
          Wtrue.clear();
          Htrue.clear();
          mask.clear();
        }
      }
      // Adjust and project non-zeros
      arma::sp_mat::iterator start_it = A.begin();
      arma::sp_mat::iterator end_it = A.end();
      for (arma::sp_mat::iterator it = start_it; it != end_it; ++it) {
        double curVal = (*it);
        if (this->m_adj_rand) {
          (*it) = ceil(kalpha * curVal + kbeta);
        }
        if ((*it) < 0) (*it) = kbeta;
      }
#else
      if (type == "uniform") {
        if (this->m_symm_flag) {
          A = arma::randu<arma::mat>(this->m_m, this->m_n);
          A = 0.5 * (A + A.t());
        } else {
          A = arma::randu<arma::mat>(this->m_m, this->m_n);
        }
      } else if (type == "normal") {
        if (this->m_symm_flag) {
          A = arma::randn<arma::mat>(this->m_m, this->m_n);
          A = 0.5 * (A + A.t());
        } else {
          A = arma::randn<arma::mat>(this->m_m, this->m_n);
        }
        A.elem(find(A < 0)).zeros();
      } else {
        if (this->m_symm_flag) {
          arma::mat Wtrue = arma::randu<arma::mat>(this->m_m, this->m_k);
          A = Wtrue * Wtrue.t();

          // Free auxiliary variables
          Wtrue.clear();
        } else {
          arma::mat Wtrue = arma::randu<arma::mat>(this->m_m, this->m_k);
          arma::mat Htrue = arma::randu<arma::mat>(this->m_k, this->m_n);
          A = Wtrue * Htrue;

          // Free auxiliary variables
          Wtrue.clear();
          Htrue.clear();
        }
      }
      if (this->m_adj_rand) {
        A = kalpha * (A) + kbeta;
        A = ceil(A);
      }
#endif
      t2 = toc();
      INFO << "generated random matrix A " << PRINTMATINFO(A)
           << "(" << t2 << " s)" << std::endl;
    }

    // Normalize the input matrix
    if (this->m_input_normalization != NONE) {
      tic();
      if (this->m_input_normalization == L2NORM) {
        A = arma::normalise(A);
      } else if (this->m_input_normalization == MAXNORM) {
        double maxnorm = 1 / A.max();
        A = maxnorm * A;
      }
      t2 = toc();
      INFO << "Normalized A (" << t2 << "s)" << std::endl;
    }

    // Set parameters and call NMF
    arma::arma_rng::set_seed(this->m_initseed);
    arma::mat W;
    arma::mat H;
    if (!this->m_h_init_file_name.empty() && !this->m_w_init_file_name.empty()) {
      W.load(m_w_init_file_name, arma::coord_ascii);
      H.load(m_h_init_file_name, arma::coord_ascii);
      this->m_k = W.n_cols;
    }
    else {
    W = arma::randu<arma::mat>(this->m_m, this->m_k);
    H = arma::randu<arma::mat>(this->m_n, this->m_k);
    }
    if (this->m_symm_flag) {
      double meanA = arma::mean(arma::mean(A));
      H = 2 * std::sqrt(meanA / this->m_k) * H;
      W = H;
      if (this->m_symm_reg == 0.0) {
        double symreg = A.max();
        this->m_symm_reg = symreg * symreg;
      }
    }

    NMFTYPE nmfAlgorithm(A.t(), W, H);
    nmfAlgorithm.num_iterations(this->m_num_it);
    nmfAlgorithm.symm_reg(this->m_symm_reg);
    nmfAlgorithm.updalgo(this->m_nmfalgo);
    // Always compute error for shared memory case
    // nmfAlgorithm.compute_error(this->m_compute_error);
    
    if (!this->m_regW.empty()) {
      nmfAlgorithm.regW(this->m_regW);
    }
    if (!this->m_regH.empty()) {
      nmfAlgorithm.regH(this->m_regH);
    }

    INFO << "completed constructor" << PRINTMATINFO(A) << std::endl;
    tic();
    nmfAlgorithm.computeNMF();
    t2 = toc();
    INFO << "time taken:" << t2 << std::endl;

    // Save the factor matrices
    if (!this->m_outputfile_name.empty()) {
      std::string WfileName = this->m_outputfile_name + "_W";
      std::string HfileName = this->m_outputfile_name + "_H";

      nmfAlgorithm.getLeftLowRankFactor().save(WfileName, arma::raw_ascii);
      nmfAlgorithm.getRightLowRankFactor().save(HfileName, arma::raw_ascii);
    }
  }

  void parseCommandLine() {
    ParseCommandLine pc(this->m_argc, this->m_argv);
    pc.parseplancopts();
    this->m_nmfalgo = pc.lucalgo();
    this->m_k = pc.lowrankk();
    this->m_Afile_name = pc.input_file_name();
    this->m_sparsity = pc.sparsity();
    this->m_num_it = pc.iterations();
    this->m_regW = pc.regW();
    this->m_regH = pc.regH();
    this->m_m = pc.globalm();
    this->m_n = pc.globaln();
    this->m_compute_error = pc.compute_error();
    this->m_symm_reg = pc.symm_reg();
    this->m_symm_flag = 0;
    this->m_adj_rand = pc.adj_rand();
    this->m_max_luciters = pc.max_luciters();
    this->m_initseed = pc.initseed();
    this->m_outputfile_name = pc.output_file_name();

    // Put in the default LUC iterations
    if (this->m_max_luciters == -1) {
      this->m_max_luciters = this->m_k;
    }

    // check the conditions for symm nmf
    if (this->m_symm_reg != -1) {
      this->m_symm_flag = 1;
      if (this->m_m != this->m_n) {
        // TODO{seswar3}: Add file prefix check
        ERR << "Symmetric Regularization enabled"
            << " and input matrix is not square"
            << "::m::" << this->m_m << "::n::" << this->m_n
            << std::endl;
        return;
      }
      if ((this->m_nmfalgo != ANLSBPP) && (this->m_nmfalgo != GNSYM)) {
        ERR << "Symmetric Regularization enabled "
            << "is only enabled for ANLSBPP and GNSYM"
            << std::endl;
        return;
      }
    }
    pc.printConfig();
    switch (this->m_nmfalgo) {
      case MU:
#ifdef BUILD_SPARSE
        callNMF<MUNMF<arma::sp_mat> >();
#else   // ifdef BUILD_SPARSE
        callNMF<MUNMF<arma::mat> >();
#endif  // ifdef BUILD_SPARSE
        break;
      case HALS:
#ifdef BUILD_SPARSE
        callNMF<HALSNMF<arma::sp_mat> >();
#else   // ifdef BUILD_SPARSE
        callNMF<HALSNMF<arma::mat> >();
#endif  // ifdef BUILD_SPARSE
        break;
      case ANLSBPP:
#ifdef BUILD_SPARSE
        callNMF<BPPNMF<arma::sp_mat> >();
#else   // ifdef BUILD_SPARSE
        callNMF<BPPNMF<arma::mat> >();
#endif  // ifdef BUILD_SPARSE
        break;
      case AOADMM:
#ifdef BUILD_SPARSE
        callNMF<AOADMMNMF<arma::sp_mat> >();
#else   // ifdef BUILD_SPARSE
        callNMF<AOADMMNMF<arma::mat> >();
#endif  // ifdef BUILD_SPARSE
        break;
      case GNSYM:
#ifdef BUILD_SPARSE
        callNMF<GNSYMNMF<arma::sp_mat> >();
#else   // ifdef BUILD_SPARSE
        callNMF<GNSYMNMF<arma::mat> >();
#endif  // ifdef BUILD_SPARSE
        break;
      default:
        ERR << "Unsupported algorithm " << this->m_nmfalgo << std::endl;
    }
  }

 public:
  NMFDriver(int argc, char *argv[]) {
    this->m_argc = argc;
    this->m_argv = argv;
    this->parseCommandLine();
  }
};  // class NMFDriver

}  // namespace planc

int main(int argc, char *argv[]) {
  try {
    planc::NMFDriver dnd(argc, argv);
    fflush(stdout);
  } catch (const std::exception &e) {
    INFO << "Exception with stack trace " << std::endl;
    INFO << e.what();
  }
}
