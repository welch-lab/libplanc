//
// Created by andrew on 6/19/2024.
//

#include <carma>
#include "config.h"
#include <nmf_lib.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include "sparse_converters.h"

namespace py = pybind11;

void limitpypythreads() {

    py::object ThreadpoolController = py::module_::import("threadpoolctl").attr("ThreadpoolController");
    py::object tpcontrol = ThreadpoolController();
    py::object limit = tpcontrol.attr("limit");
    py::object platformimpl = py::module_::import("platform").attr("python_implementation");
     py::str platform = platformimpl();
    if (static_cast<std::string>(platform) == "PyPy") {
        using namespace pybind11::literals;
        limit("limits"_a=1, "user_api"_a="blas");
        limit("limits"_a=1, "user_api"_a="openmp");
    }
} // state does not unset, TODO find way to do on import/unload
//Rcpp::List nmf(const SEXP& x, const arma::uword &k, const arma::uword &niter = 30,
//               const std::string &algo = "anlsbpp",
//               const int& nCores = 2,
//               const Rcpp::Nullable<Rcpp::NumericMatrix> &Winit = R_NilValue,
//               const Rcpp::Nullable<Rcpp::NumericMatrix> &Hinit = R_NilValue) {
//    Rcpp::List outlist;
//    try {
//        if (Rf_isS4(x)) {
//            // Assume using dgCMatrix
//            outlist = runNMF<arma::sp_mat>(Rcpp::as<arma::sp_mat>(x), k, algo, niter, nCores, Winit, Hinit);
//        } else {
//            // Assume regular dense matrix
//            outlist = runNMF<arma::mat>(Rcpp::as<arma::mat>(x), k, algo, niter, nCores, Winit, Hinit);
//        }
//    } catch (const std::exception &e) {
//        throw Rcpp::exception(e.what());
//    }
//    return outlist;
//}

arma::mat nullMat = arma::mat{};

template <typename T>
using nmfCall = planc::nmfOutput(*)(const T&, const arma::uword&, const arma::uword&, const std::string&, const int&);

template <typename T>
using nmfFullCall =  planc::nmfOutput(*)(const T&, const arma::uword&, const arma::uword&, const std::string&, const int&, const arma::mat&, const arma::mat&);


// T2 e.g. arma::sp_mat
template <typename T2>
planc::nmfOutput nmf(const T2& x, const arma::uword &k,
             const arma::uword& niter = 30, const std::string& algo = "anlsbpp",
             const int& nCores = 2, const arma::mat& Winit = nullMat, const arma::mat& Hinit = nullMat) {
    limitpypythreads();
    planc::nmfOutput libcall = planc::nmflib<T2>::nmf(x, k, niter, algo, nCores, Winit, Hinit);
    return libcall;
}

template <typename T2>
planc::nmfOutput nmf(const T2& x, const arma::uword &k,
             const arma::uword& niter = 30, const std::string& algo = "anlsbpp",
             const int& nCores = 2) {
    limitpypythreads();
    planc::nmfOutput libcall = planc::nmflib<T2>::nmf(x, k, niter, algo, nCores);
    return libcall;
}

PYBIND11_MODULE(pyplanc, m) {
    m.doc() = "A python wrapper for planc-nmflib";
    using namespace py::literals;
    py::class_<planc::nmfOutput>(m, "nmfOutput").def_readwrite("W", &planc::nmfOutput::outW).def_readwrite("H", &planc::nmfOutput::outH).def_readwrite("objErr", &planc::nmfOutput::objErr);
    m.def("nmf", static_cast<nmfCall<arma::mat>>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2);
    m.def("nmf", static_cast<nmfCall<arma::sp_mat>>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2);
    m.def("nmf", static_cast<nmfFullCall<arma::mat>>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2,
          py::kw_only(), "Winit"_a, "Hinit"_a);
    m.def("nmf", static_cast<nmfFullCall<arma::sp_mat>>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2,
      py::kw_only(), "Winit"_a, "Hinit"_a);
}
