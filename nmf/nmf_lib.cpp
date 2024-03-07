//
// Created by andrew on 12/12/2023.
//
#include "NMFDriver.hpp"
#include "utils.hpp"
#include "nmf_lib.hpp"
#include "NMFDriver.hpp"

extern "C" {
#include "detect_blas.h"
}
planc::nmflib::nmflib() {
    openblas_pthread_off((get_openblas_handle()));
}

void planc::nmflib::openblas_pthread_off(openblas_handle_t libloc) {
    if (is_openmp()) {
        if (const std::function<int()> openblas_parallel = get_openblas_parallel(libloc))
            {
                if (openblas_parallel() == 1) {
                    const std::function<void(int)> openblas_set = get_openblas_set(libloc);
                    openblas_set(1);
                }
            }
    }
}

void planc::nmflib::openblas_pthread_on(openblas_handle_t libloc) {if (is_openmp()) {
    if (const std::function<int()> openblas_parallel = get_openblas_parallel(libloc))
    {
        if (openblas_parallel() == 1) {
            const std::function<void(int)> openblas_set = get_openblas_set(libloc);
            openblas_set(0);
        }
    }
}
}

template<> int planc::nmflib::runNMF<arma::mat>(planc::params opts) {
    planc::NMFDriver<arma::mat> myNMF(opts);
    myNMF.callNMF();
    return 0;
};
template<> int planc::nmflib::runNMF<arma::sp_mat>(planc::params opts) {
    planc::NMFDriver<arma::sp_mat> myNMF(opts);
    myNMF.callNMF();
    return 0;
};

