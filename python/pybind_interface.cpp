//
// Created by andrew on 6/19/2024.
//

#include "config.h"
#include <nmf_lib.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eval.h>
#include <initializer_list>
#include <utility>
//#include "converters.h"

namespace nb = nanobind;

void limitpypythreads() {
    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(R"(
    import threadpoolctl.ThreadpoolController
    from platform import python_implementation
    tpcontrol = ThreadpoolController()
    platform = python_implementation()
    if platform == "PyPy":
        tpcontrol.limit(limits=1, user_api="blas")
        tpcontrol.limit(limits=1, user_api="openmp")
    )", scope
    );
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

typedef nb::ndarray<double, nb::ndim<2>> DenseNBArray;

struct NBNMFOutput {
    nb::ndarray<nb::numpy, double, nb::ndim<2>> W;
    nb::ndarray<nb::numpy, double, nb::ndim<2>> H;
    double objErr;
};

arma::mat denseToArmadillo(DenseNBArray nda) {
        return {nda.data(), nda.shape(0), nda.shape(1)};
}

nb::ndarray<nb::numpy, double, nb::ndim<2>> armaToNP(arma::mat out) {
    nb::capsule owner(out.memptr(), [](void *p) noexcept {
       delete[] (double *) p;
    });
    return {out.memptr(), {out.n_rows, out.n_cols}, owner};
}

using denseNmfCall = NBNMFOutput(*)(DenseNBArray, const arma::uword&, const arma::uword&, const std::string&, const int&);
//using sparseNmfCall = planc::nmfOutput(*)(nb::ndarray<double, nb::shape<-1, -1>>, const arma::uword&, const arma::uword&, const std::string&, const int&);


using denseNmfFullCall =  NBNMFOutput(*)(DenseNBArray, const arma::uword&, const arma::uword&, const std::string&, const int&, DenseNBArray, DenseNBArray);
//using sparseNmfFullCall =  planc::nmfOutput(*)(nb::ndarray<double, nb::shape<-1, -1>>, const arma::uword&, const arma::uword&, const std::string&, const int&, DenseNBArray, DenseNBArray);


NBNMFOutput nmf(DenseNBArray nda, const arma::uword &k,
             const arma::uword& niter, const std::string& algo,
             const int& nCores, DenseNBArray Winit, DenseNBArray Hinit) {
    limitpypythreads();
    planc::nmfOutput libcall = planc::nmflib<arma::mat>::nmf(denseToArmadillo(std::move(nda)), k, niter, algo, nCores, denseToArmadillo(std::move(Winit)), denseToArmadillo(std::move(Hinit)));
    return {armaToNP(libcall.outH), armaToNP(libcall.outW), libcall.objErr};
}

NBNMFOutput nmf(DenseNBArray nda, const arma::uword &k,
             const arma::uword& niter = 30, const std::string& algo = "anlsbpp",
             const int& nCores = 2) {
    limitpypythreads();
    planc::nmfOutput libcall = planc::nmflib<arma::mat>::nmf(denseToArmadillo(std::move(nda)), k, niter, algo, nCores);
    return {armaToNP(libcall.outH), armaToNP(libcall.outW), libcall.objErr};
}

NB_MODULE(pyplanc, m) {
    m.doc() = "A python wrapper for planc-nmflib";
    using namespace nb::literals;
    nb::class_<NBNMFOutput>(m, "nmfOutput").def_rw("W", &NBNMFOutput::W).def_rw("H", &NBNMFOutput::H).def_rw("objErr", &NBNMFOutput::objErr);
    m.def("nmf", static_cast<denseNmfCall>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2);
//    m.def("nmf", static_cast<sparseNmfCall>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2);
    m.def("nmf", static_cast<denseNmfFullCall>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2,
          nb::kw_only(), "Winit"_a, "Hinit"_a);
//    m.def("nmf", static_cast<nmfFullCall>(nmf), "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2,
//      nb::kw_only(), "Winit"_a, "Hinit"_a);
}
