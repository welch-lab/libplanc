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
        NMFLIB_EXPORT nmflib();

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

    template<typename T>
    nmfOutput NMFLIB_EXPORT nmf(const T& x,
                                                                 const arma::uword &k,
                                                                 const arma::uword &niter = 30,
                   const std::string &algo = "anlsbpp",
                   const int& nCores = 2,
                   const arma::mat& Winit = arma::mat(1, 1, arma::fill::none),
                   const arma::mat& Hinit = arma::mat(1, 1, arma::fill::none)) {
        internalParams<T> options(x, Winit, Hinit);
        options.m_k = k;
        options.m_num_it = niter;
        options.setMLucalgo(algo);
        EmbeddedNMFDriver<T> nmfRunner(options);
        nmfRunner.callNMF();
        nmfOutput outlist{};
        outlist.outW = nmfRunner.getLlf();
        outlist.outH = nmfRunner.getRlf();
        outlist.objErr = nmfRunner.getobjErr();
        return outlist;
    }

}

#endif //PLANC_NMF_LIB_HPP
