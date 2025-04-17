//
// Created by andrew on 6/19/2024.
//

#include "config.h"
#include <nmf_lib.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "dense_casters.h"
#include "sparse_casters.h"
#include "pointer_helpers.hpp"


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

NB_MODULE(pyplanc, m) {
    m.doc() = "A python wrapper for planc-nmflib";
    using namespace nb::literals;
    nb::class_<planc::nmfOutput<double>>(m, "nmfOutput").def_rw("W", &planc::nmfOutput<double>::outW).def_rw("H", &planc::nmfOutput<double>::outH).def_rw("objErr", &planc::nmfOutput<double>::objErr,  nb::rv_policy::move);
    nb::class_<planc::inmfOutput<double>>(m, "inmfOutput").def_rw("W", &planc::inmfOutput<double>::outW).def_rw("HList", &planc::inmfOutput<double>::outHList).def_rw("VList", &planc::inmfOutput<double>::outVList).def_rw("objErr", &planc::inmfOutput<double>::objErr,  nb::rv_policy::move);
    nb::class_<planc::uinmfOutput<double>>(m, "uinmfOutput").def_rw("W", &planc::uinmfOutput<double>::outW).def_rw("HList", &planc::uinmfOutput<double>::outHList).def_rw("VList", &planc::uinmfOutput<double>::outVList).def_rw("objErr", &planc::uinmfOutput<double>::objErr,  nb::rv_policy::move).def_rw("UList", &planc::uinmfOutput<double>::outUList);
    nb::class_<planc::oinmfOutput<double>>(m, "oinmfOutput").def_rw("W", &planc::oinmfOutput<double>::outW).def_rw("HList", &planc::oinmfOutput<double>::outHList).def_rw("VList", &planc::oinmfOutput<double>::outVList).def_rw("objErr", &planc::oinmfOutput<double>::objErr,  nb::rv_policy::move).def_rw("AList", &planc::oinmfOutput<double>::outAList).def_rw("BList", &planc::oinmfOutput<double>::outBList);

    nb::class_<planc::H5Mat>(m, "H5Mat").def(nb::init<std::string, std::string>());
    nb::class_<planc::H5SpMat>(m, "H5SpMat").def(nb::init<std::string, std::string, std::string, std::string, arma::uword, arma::uword>());


#define X(T) \
    m.def("nmf", &planc::nmflib<T>::nmf, "A function that calls NMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "algo"_a="anlsbpp", "ncores"_a=2, nb::kw_only(), "Winit"_a = arma::mat(), "Hinit"_a = arma::mat()); \
    m.def("symNMF", &planc::nmflib<T>::symNMF, "A function that calls symNMF with the given arguments", "x"_a, "k"_a, "niter"_a=30, "lambda"_a=0.0, "algo"_a="gnsym", "ncores"_a=2,  nb::kw_only(), "Hinit"_a = arma::mat());
    #include "nmf_types.inc"
#undef X
#define X(T) \
    m.def("bppinmf", nb::overload_cast<std::vector<T>, const arma::uword&, const double&, const arma::uword&, const bool&, const int&>(&planc::bppinmf_py<T>), "A function that calls bppinmf with the given arguments", "objectList"_a, "k"_a, "lambda"_a=5.0, "niter"_a=30, "verbose"_a=false, "ncores"_a=2); \
    m.def("bppinmf", nb::overload_cast<std::vector<T>, const arma::uword&, const double&, const arma::uword&, const bool&, const std::vector<arma::mat>&, const std::vector<arma::mat>&, const arma::mat&, const int&>(&planc::bppinmf_py<T>), "A function that calls bppinmf with the given arguments", "objectList"_a, "k"_a, "lambda"_a=5.0, "niter"_a=30, "verbose"_a=false, "HinitList"_a, "VinitList"_a, "&Winit"_a,"ncores"_a=2); \
    m.def("uinmf", &planc::uinmf_py<T>, "A function that calls uinmf with the given arguments", "objectList"_a, "unsharedList"_a, "whichUnshared"_a, "k"_a=20, "nCores"_a=2, "lambda"_a=5, "niter"_a=30, "verbose"_a=false); \
    m.def("oinmf", nb::overload_cast<std::vector<T>, const arma::uword&, const int&, const double&, const arma::uword&, const arma::uword&, const arma::uword&, const arma::uword&, const bool&>(&planc::oinmf_py<T>), "A function that calls oinmf with the given arguments", "objectList"_a, "k"_a, "nCores"_a=2, "lambda"_a=5, "maxEpoch"_a=5, "minibatchSize"_a=5000, "maxHALSIter"_a=1, "permuteChunkSize"_a=1000, "verbose"_a=false); \
    m.def("oinmf", nb::overload_cast<std::vector<T>, const std::vector<arma::mat>&, const std::vector<arma::mat>&, const arma::mat&, const std::vector<arma::mat>&, const std::vector<arma::mat>&, std::vector<T>, const arma::uword&, const int&, const double&, const arma::uword&, const arma::uword&, const arma::uword&, const arma::uword&, const bool&>(&planc::oinmf_py<T>), "A function that calls oinmf with the given arguments", "objectList"_a, "Hinit"_a, "Vinit"_a, "Winit"_a, "Ainit"_a, "Binit"_a, "objectListNew"_a, "k"_a, "nCores"_a=2, "lambda"_a=5, "maxEpoch"_a=5, "minibatchSize"_a=5000, "maxHALSIter"_a=1, "permuteChunkSize"_a=1000, "verbose"_a=false); \
    m.def("oinmf_project", &planc::oinmf_project_py<T>, "A function that calls oinmf_project with the given arguments", "objectList"_a, "Winit"_a, "objectListNew"_a, "k"_a, "nCores"_a = 2, "lambda"_a=5);
    #include "inmf_types.inc"
#undef X
}
