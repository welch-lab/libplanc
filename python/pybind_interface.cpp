//
// Created by andrew on 6/19/2024.
//

#include "config.h"
#include <nmf_lib.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "pybind_interface.hpp"

namespace nb = nanobind;
// void limitpypythreads() {
//     nb::object scope = nb::module_::import_("__main__").attr("__dict__");
//     nb::exec(R"(
//     from threadpoolctl import ThreadpoolController
//     from platform import python_implementation
//     tpcontrol = ThreadpoolController()
//     platform = python_implementation()
//     if platform == "PyPy":
//         tpcontrol.limit(limits=1, user_api="blas")
//         tpcontrol.limit(limits=1, user_api="openmp")
//     )", scope
//     );
    // nb::callable ThreadpoolController = nb::module_::import_("threadpoolctl").attr("ThreadpoolController");
    // nb::object tpcontrol = ThreadpoolController();
    // nb::callable limit = tpcontrol.attr("limit");
    // nb::callable platformimpl = nb::module_::import_("platform").attr("python_implementation");
    // nb::str platform = platformimpl();
    // if (static_cast<std::string>(platform) == "PyPy") {
    //     using namespace nb::literals;
    //     limit("limits"_a=1, "user_api"_a="blas");
    //     limit("limits"_a=1, "user_api"_a="openmp");
    // }
// } // state does not unset, TODO find way to do on import/unload
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


NB_MODULE(pyplanc, m) {
    m.doc() = "A python wrapper for planc-nmflib";
    using namespace nb::literals;
    nb::class_<planc::nmfOutput<double>>(m, "nmfOutput").def_rw("W", &planc::nmfOutput<double>::outW).def_rw("H", &planc::nmfOutput<double>::outH).def_rw("objErr", &planc::nmfOutput<double>::objErr,  nb::rv_policy::move);
    //nb::class_<ScipySparseCSC>(m, "sparseCSCArray").def_rw("data", &ScipySparseCSC::data).def_rw("indices", &ScipySparseCSC::indices).def_rw("indptr", &ScipySparseCSC::indptr).def_rw("shape", &ScipySparseCSC::shape).def(nb::init_implicit<arma::sp_mat>());
    m.def("nmf", static_cast<nmfCall<arma::mat>>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2);
    m.def("nmf", static_cast<nmfCall<arma::sp_mat>>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2);
    m.def("nmf", static_cast<nmfFullCall<arma::mat>>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2,
          nb::kw_only(), "Winit"_a, "Hinit"_a);
    m.def("nmf", static_cast<nmfFullCall<arma::sp_mat>>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2,  nb::kw_only(), "Winit"_a, "Hinit"_a);
}
