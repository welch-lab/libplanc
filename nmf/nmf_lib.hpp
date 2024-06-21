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
#include "EmbeddedNMFDriver.hpp"

extern "C" {
#include "detect_blas.h"
}
namespace planc {
    class NMFLIB_EXPORT nmflib {
    public:
        nmflib();

        template<typename T>
        int NMFLIB_EXPORT runNMF(params opts);

        template<typename T>
        int NMFLIB_EXPORT runINMF(params opts);
        static void NMFLIB_NO_EXPORT openblas_pthread_off(openblas_handle_t);
        static void NMFLIB_NO_EXPORT openblas_pthread_on(openblas_handle_t);
    };

    //enum NMFLIB_EXPORT runNMFindex {outW, outH, objErr};
    struct NMFLIB_EXPORT nmfOutput {
      arma::mat outW;
      arma::mat outH;
      double objErr;
    };
    //extern std::map<std::string, runNMFindex> NMFindexmap;

    template <typename T>
    extern nmfOutput NMFLIB_NO_EXPORT nmf(const T &x,
                                const arma::uword &k,
                                const arma::uword &niter = 30,
                                const std::string &algo = "anlsbpp",
                                const int &nCores = 2,
                                const arma::mat &Winit = arma::mat(),
                                const arma::mat &Hinit = arma::mat());
    template nmfOutput NMFLIB_EXPORT nmf<arma::mat>(const arma::mat &x,
                                                  const arma::uword &k,
                                                  const arma::uword &niter,
                                                  const std::string &algo,
                                                  const int &nCores,
                                                  const arma::mat &Winit,
                                                  const arma::mat &Hinit);
    template nmfOutput NMFLIB_EXPORT nmf<arma::sp_mat>(const arma::sp_mat &x,
                                                       const arma::uword &k,
                                                       const arma::uword &niter,
                                                       const std::string &algo,
                                                       const int &nCores,
                                                       const arma::mat &Winit,
                                                       const arma::mat &Hinit);
}

#endif //PLANC_NMF_LIB_HPP
