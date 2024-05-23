//
// Created by andrew on 12/12/2023.
//

#ifndef PLANC_NMF_LIB_HPP
#define PLANC_NMF_LIB_HPP
#include "nmflib_export.h"
#include "plancopts.h"
#include "bppnmf.hpp"
#include "aoadmm.hpp"
#include "mu.hpp"

extern "C" {
#include "detect_blas.h"
}
namespace planc {
    class NMFLIB_EXPORT nmflib {
    public:
        NMFLIB_EXPORT nmflib();

        template<typename T>
        int NMFLIB_EXPORT runNMF(params opts);

        template<typename T>
        int NMFLIB_EXPORT runINMF(params opts);

    private:
        static void NMFLIB_NO_EXPORT openblas_pthread_off(openblas_handle_t);
        static void NMFLIB_NO_EXPORT openblas_pthread_on(openblas_handle_t);
    };

    enum NMFLIB_EXPORT runNMFindex {outW, outH, objErr};
    //extern std::map<std::string, runNMFindex> NMFindexmap;

    template <class T1, class T2>
    std::array<arma::mat, 3> NMFLIB_NO_EXPORT runNMF(T2 x, arma::uword k,
                                                                    const int& nCores, arma::uword niter,
                                                                    const arma::mat& Winit = arma::mat(1, 1, arma::fill::none),
                                                                    const arma::mat& Hinit = arma::mat(1, 1, arma::fill::none)) {
        arma::uword m = x.n_rows;
        arma::uword n = x.n_cols;
        if (k >= m) {
            std::throw_with_nested("`k` must be less than `nrow(x)`");
        }
        arma::mat W(m, k);
        arma::mat H(n, k);
        if (!Winit.is_empty()) {
            W = Winit;
            if (W.n_rows != m || W.n_cols != k) {
                std::throw_with_nested("Winit must be of size " +
                           std::to_string(m) + " x " + std::to_string(k));
            }
        } else {
            W = arma::randu<arma::mat>(m, k);
        }
        if (!Hinit.is_empty()) {
            H = Hinit;
            if (H.n_rows != n || H.n_cols != k) {
                std::throw_with_nested("Hinit must be of size " +
                           std::to_string(n) + " x " + std::to_string(k));
            }
        } else {
            H = arma::randu<arma::mat>(n, k);
        }
        T1 MyNMF(x, W, H, nCores);
        MyNMF.num_iterations(niter);
        MyNMF.symm_reg(-1);
        // if (!m_regW.empty()) MyNMF.regW(m_regW);
        // if (!m_regH.empty()) MyNMF.regH(m_regH);
        MyNMF.computeNMF();
        std::array<arma::mat, 3>  output;
        output[outW] = MyNMF.getLeftLowRankFactor();
        output[outH] = MyNMF.getRightLowRankFactor();
        output[objErr] = MyNMF.objErr();
        return output;
    }

    template<typename T>
    std::array<arma::mat, 3>  NMFLIB_EXPORT nmf(const T& x,
                                                                 const arma::uword &k,
                                                                 const arma::uword &niter = 30,
                   const std::string &algo = "anlsbpp",
                   const int& nCores = 2,
                   const arma::mat& Winit = arma::mat(1, 1, arma::fill::none),
                   const arma::mat& Hinit = arma::mat(1, 1, arma::fill::none)) {
        std::array<arma::mat, 3>  outlist;
            // Assume using dgCMatrix
            if (algo == "anlsbpp") {
                outlist = runNMF<planc::BPPNMF<T>, T>(
                        x, k, nCores, niter, Winit, Hinit
                );
            } else if (algo == "admm") {
                outlist = runNMF<planc::AOADMMNMF<T>, T>(
                        x, k, nCores, niter, Winit, Hinit
                );
            } else if (algo == "hals") {
                outlist = runNMF<planc::HALSNMF<T>, T>(
                        x, k, nCores, niter, Winit, Hinit
                );
            } else if (algo == "mu") {
                outlist = runNMF<planc::MUNMF<T>, T>(
                        x, k, nCores, niter, Winit, Hinit
                );
            } else {
                std::throw_with_nested(R"(Please choose `algo` from "anlsbpp", "admm", "hals" or "mu".)");
            }
        return outlist;
    }

}

#endif //PLANC_NMF_LIB_HPP
