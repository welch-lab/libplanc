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
    struct NMFLIB_EXPORT nmfOutput {
        arma::mat outW;
        arma::mat outH;
        double objErr;
    };

    template<typename T>
    class nmflib {
    public:
        NMFLIB_EXPORT nmflib<T>();


        int NMFLIB_EXPORT runNMF(params opts);

        int NMFLIB_EXPORT runINMF(params opts);
        static nmfOutput NMFLIB_EXPORT nmf(const T &x, const arma::uword &k, const arma::uword &niter, const std::string &algo, const int &nCores,
                                    const arma::mat &Winit = arma::mat(), const arma::mat &Hinit = arma::mat());
    };


    extern void NMFLIB_NO_EXPORT openblas_pthread_off(openblas_handle_t);
    extern void NMFLIB_NO_EXPORT openblas_pthread_on(openblas_handle_t);




    //enum NMFLIB_EXPORT runNMFindex {outW, outH, objErr};

    //extern std::map<std::string, runNMFindex> NMFindexmap;

}

#endif //PLANC_NMF_LIB_HPP
