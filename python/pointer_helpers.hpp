//
// Created by andrew on 4/5/2025.
//

#ifndef POINTER_HELPERS_HPP
#define POINTER_HELPERS_HPP

#include "config.h"
#include <nmf_lib.hpp>

namespace planc {
    template<typename T, typename eT = typename T::elem_type>
    inmfOutput<eT> bppinmf_py(std::vector<T> objectList, const arma::uword&k,
                              const double&lambda,
                              const arma::uword&niter, const bool&verbose, const int&ncores) {
        return nmflib<T>::bppinmf(nmflib<T>::initMemSharedPtr(objectList), k, lambda,
                                  niter, verbose, ncores);
    }

    template<typename T, typename eT = typename T::elem_type>
    inmfOutput<eT> bppinmf_py(std::vector<T> objectList, const arma::uword&k,
                              const double&lambda,
                              const arma::uword&niter, const bool&verbose,
                              const std::vector<arma::mat>&HinitList,
                              const std::vector<arma::mat>&VinitList, const arma::mat&Winit,
                              const int&ncores) {
        return nmflib<T>::bppinmf(nmflib<T>::initMemSharedPtr(objectList), k, lambda,
                                  niter, verbose, HinitList, VinitList, Winit, ncores);
    }

    template<typename T, typename eT = typename T::elem_type>
    uinmfOutput<eT> uinmf_py(const std::vector<T>&matPtrVec,
                             const std::vector<T>&unsharedPtrVec,
                             std::vector<int> whichUnshared,
                             const arma::uword&k, const int&nCores, const arma::vec&lambda,
                             const arma::uword&niter, const bool&verbose) {
        return nmflib<T>::uinmf(nmflib<T>::initMemSharedPtr(matPtrVec), nmflib<T>::initMemSharedPtr(unsharedPtrVec),
                                whichUnshared, k, nCores, lambda, niter, verbose);
    }

    template<typename T, typename eT = typename T::elem_type>
    oinmfOutput<eT> oinmf_py(std::vector<T> matPtrVec, const arma::uword&k,
                             const int&nCores,
                             const double&lambda, const arma::uword&maxEpoch,
                             const arma::uword&minibatchSize, const arma::uword&maxHALSIter,
                             const arma::uword&permuteChunkSize, const bool&verbose) {
        return nmflib<T>::oinmf(nmflib<T>::initMemSharedPtr(matPtrVec), k, nCores, lambda, maxEpoch,
                                minibatchSize, maxHALSIter, permuteChunkSize, verbose);
    }

    template<typename T, typename eT = typename T::elem_type>
    oinmfOutput<eT> oinmf_py(std::vector<T> matPtrVec,
                             const std::vector<arma::mat>&Hinit,
                             const std::vector<arma::mat>&Vinit, const arma::mat&Winit,
                             const std::vector<arma::mat>&Ainit, const std::vector<arma::mat>&Binit,
                             std::vector<T> matPtrVecNew,
                             const arma::uword&k, const int&nCores, const double&lambda,
                             const arma::uword&maxEpoch,
                             const arma::uword&minibatchSize, const arma::uword&maxHALSIter,
                             const arma::uword&permuteChunkSize, const bool&verbose) {
        return nmflib<T>::oinmf(nmflib<T>::initMemSharedPtr(matPtrVec), Hinit, Vinit, Winit, Ainit, Binit,
                                nmflib<T>::initMemSharedPtr(matPtrVecNew), k, nCores, lambda, maxEpoch, minibatchSize,
                                maxHALSIter, permuteChunkSize, verbose);
    }

    template<typename T, typename eT = typename T::elem_type>
    std::vector<arma::Mat<eT>> oinmf_project_py(std::vector<T> matPtrVec,
                                                const arma::mat&Winit,
                                                std::vector<T> matPtrVecNew,
                                                const arma::uword&k, const int&nCores, const double&lambda) {
        return nmflib<T>::oinmf_project(nmflib<T>::initMemSharedPtr(matPtrVec), Winit,
                                        nmflib<T>::initMemSharedPtr(matPtrVecNew), k, nCores, lambda);
    }
}

#endif //POINTER_HELPERS_HPP
