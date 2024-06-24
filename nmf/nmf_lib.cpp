//
// Created by andrew on 12/12/2023.
//
#include "NMFDriver.hpp"
#include "utils.hpp"
#include "nmf_lib.hpp"


extern "C" {
#include "detect_blas.h"
}
template<typename T>
planc::nmflib<T>::nmflib() {
    openblas_pthread_off((get_openblas_handle()));
}


void planc::openblas_pthread_off(openblas_handle_t libloc) {
    if (is_openmp()) {
        if (const std::function openblas_parallel = get_openblas_parallel(libloc))
            {
                if (openblas_parallel() == 1) {
                    const std::function openblas_set = get_openblas_set(libloc);
                    openblas_set(1);
                }
            }
    }
}


void planc::openblas_pthread_on(openblas_handle_t libloc) {if (is_openmp()) {
    if (const std::function openblas_parallel = get_openblas_parallel(libloc))
    {
        if (openblas_parallel() == 1) {
            const std::function openblas_set = get_openblas_set(libloc);
            openblas_set(0);
        }
    }
}
}

template<typename T>
planc::nmfOutput planc::nmflib<T>::nmf(const T &x, const arma::uword &k, const arma::uword &niter, const std::string &algo,
                         const int &nCores, const arma::mat &Winit, const arma::mat &Hinit) {
    return {};
}

template<>
planc::nmfOutput planc::nmflib<arma::mat>::nmf(const arma::mat &x, const arma::uword &k, const arma::uword &niter, const std::string &algo,
                         const int &nCores, const arma::mat &Winit, const arma::mat &Hinit) {
    internalParams<arma::mat> options(x, Winit, Hinit);
    options.m_k = k;
    options.m_num_it = niter;
    options.setMLucalgo(algo);
    EmbeddedNMFDriver<arma::mat> nmfRunner(options);
    nmfRunner.callNMF();
    nmfOutput outlist{};
    outlist.outW = nmfRunner.getLlf();
    outlist.outH = nmfRunner.getRlf();
    outlist.objErr = nmfRunner.getobjErr();
    return outlist;
}

template<>
planc::nmfOutput planc::nmflib<arma::sp_mat>::nmf(const arma::sp_mat &x, const arma::uword &k, const arma::uword &niter, const std::string &algo,
                                               const int &nCores, const arma::mat &Winit, const arma::mat &Hinit) {
    internalParams<arma::sp_mat> options(x, Winit, Hinit);
    options.m_k = k;
    options.m_num_it = niter;
    options.setMLucalgo(algo);
    EmbeddedNMFDriver<arma::sp_mat> nmfRunner(options);
    nmfRunner.callNMF();
    nmfOutput outlist{};
    outlist.outW = nmfRunner.getLlf();
    outlist.outH = nmfRunner.getRlf();
    outlist.objErr = nmfRunner.getobjErr();
    return outlist;
}

template<typename T>
int planc::nmflib<T>::runNMF(planc::params opts) {
    planc::NMFDriver<T> myNMF(opts);
    myNMF.callNMF();
    return 0;
};


template<typename T>
int planc::nmflib<T>::runINMF(planc::params opts) {
    planc::NMFDriver<T> myNMF(opts);
    myNMF.callNMF();
    return 0;
};



//std::map<std::string, planc::runNMFindex> NMFindexmap{{"W", planc::outW},
//                                                      {"H", planc::outH},
//                                                      {"objErr", planc::objErr}};