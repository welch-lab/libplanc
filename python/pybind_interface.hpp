//
// Created by andrew on 7/31/2024.
//

#ifndef PYBIND_INTERFACE_HPP
#define PYBIND_INTERFACE_HPP
#include "casters.h"



template <typename T, typename eT = typename T::elem_type>
using nmfCall  = planc::nmfOutput<eT>(*)(const T&, const arma::uword&, const arma::uword&, const std::string&, const int&);


template <typename T, typename eT = typename T::elem_type>
using nmfFullCall  = planc::nmfOutput<eT>(*)(const T&, const arma::uword&, const arma::uword&, const std::string&, const int&, const arma::Mat<eT>&, const arma::Mat<eT>&);

template <typename T, typename eT = typename T::elem_type>
planc::nmfOutput<eT> nmf(const T& x, const arma::uword &k,
             const arma::uword& niter, const std::string& algo,
             const int& nCores, const arma::Mat<eT>& Winit, const arma::Mat<eT>& Hinit) {
    return planc::nmflib<T>::nmf(x, k, niter, algo, nCores, Winit, Hinit);
}


template <typename T, typename eT = typename T::elem_type>
planc::nmfOutput<eT> nmf(const T& x, const arma::uword &k,
             const arma::uword& niter = 30, const std::string& algo = "anlsbpp",
             const int& nCores = 2) {
    return planc::nmflib<T>::nmf(x, k, niter, algo, nCores);

}

// template <typename T, typename eT = typename T::elem_type>
// planc::nmfOutput<T> zzznmf(const ScipySparseCSC& nda, const arma::uword &k,
//              const arma::uword& niter, const std::string& algo,
//              const int& nCores, const arma::Mat<eT>& Winit, const arma::mat& Hinit) {
//     return planc::nmflib<arma::Mat<eT>>::nmf(sparseToArmadillo(nda), k, niter, algo, nCores, Winit, Hinit);
// }
//
//
// template <typename T, typename eT = typename T::elem_type>
// planc::nmfOutput<T> nmf(const ScipySparseCSC& nda, const arma::uword &k,
//              const arma::uword& niter = 30, const std::string& algo = "anlsbpp",
//              const int& nCores = 2) {
//     return planc::nmflib<arma::Mat<eT>>::nmf(sparseToArmadillo(nda), k, niter, algo, nCores);
// }

#endif //PYBIND_INTERFACE_HPP
