#pragma once

#include "nmf_lib.hpp"
#include "EmbeddedNMFDriver.hpp"
#include "bppinmf.hpp"

namespace planc {
    template<typename T, typename eT>
    nmfOutput<eT> NMFLIB_EXPORT nmflib<T, eT>::nmf(const T& x, const arma::uword& k, const arma::uword& niter,
                                               const std::string& algo, const int& nCores, const arma::Mat<eT>& Winit, const arma::Mat<eT>& Hinit) {
        internalParams options(x, Winit, Hinit);
        options.setMK(k);
        options.setMNumIt(niter);
        options.setMLucalgo(algo);
        EmbeddedNMFDriver nmfRunner(options);
        nmfRunner.callNMF();
        nmfOutput<eT> outlist{};
        outlist.outW = nmfRunner.getLlf();
        outlist.outH = nmfRunner.getRlf();
        outlist.objErr = nmfRunner.getobjErr();
        return outlist;
    }
    // T1 e.g. BPPNMF<arma::sp_mat>
    // T2 e.g. arma::sp_mat
    template<typename T, typename eT>
    nmfOutput<eT> NMFLIB_EXPORT nmflib<T, eT>::symNMF(const T& x, const arma::uword& k, const arma::uword& niter, const double& lambda, const std::string& algo, const int& nCores,
                         const arma::Mat<eT>& Hinit) {
        internalSymmParams options(x, Hinit);
        options.setMK(k);
        options.setMNumIt(niter);
        options.setMLucalgo(algo);
        options.setMSymmReg(lambda);
        options.setMSymmFlag(1);
        arma::uword m = x.n_rows;
        arma::uword n = x.n_cols;
        // if (m != n) {
        //     Rcpp::stop("Input `x` is not square.");
        // }
        // if (k >= m) {
        //     Rcpp::stop("`k` must be less than `nrow(x)`");
        // }
        symmEmbeddedNMFDriver nmfRunner(options);
        nmfRunner.callNMF();
        nmfOutput<eT> outlist{};
        outlist.outW = nmfRunner.getLlf();
        outlist.outH = nmfRunner.getRlf();
        outlist.objErr = nmfRunner.getobjErr();
        return outlist;
    }
    template <typename T, typename eT>
    inmfOutput<eT> nmflib<T, eT>::bppinmf(const std::vector<T> &objectList, const arma::uword &k, const double &lambda,
                   const arma::uword &niter, const bool &verbose, const int& ncores)
    {
        std::vector<std::unique_ptr<T>> matPtrVec;
        matPtrVec = initMemMatPtr<T>(objectList);
        BPPINMF<T> solver(matPtrVec, k, lambda);
        solver.optimizeALS(niter, verbose, ncores);
        std::vector<std::unique_ptr<arma::mat>> allH = solver.getAllH();
        std::vector<arma::mat> resolvedH{};
        for (unsigned int i = 0; i < allH.size(); ++i) {
            arma::mat* ptr = allH[i].release();
            resolvedH.push_back(*ptr);
        }
        std::vector<std::unique_ptr<arma::mat>> allV = solver.getAllV();
        std::vector<arma::mat> resolvedV{};
        for (unsigned int i = 0; i < allV.size(); ++i) {
            arma::mat* ptr = allV[i].release();
            resolvedV.push_back(*ptr);
        }
        return {solver.getW(), resolvedH, resolvedV, solver.objErr()};
    }
    template <typename T, typename eT>
    inmfOutput<eT> nmflib<T, eT>::bppinmf(const std::vector<T> &objectList, const arma::uword &k, const double &lambda,
                       const arma::uword &niter, const bool &verbose,
                       const std::vector<arma::mat> &HinitList, const std::vector<arma::mat> &VinitList, const arma::mat &Winit,
                       const int& ncores)
    {
        std::vector<std::unique_ptr<T>> matPtrVec;
        matPtrVec = initMemMatPtr<T>(objectList);
        BPPINMF<T> solver(matPtrVec, k, lambda, HinitList, VinitList, Winit);
        solver.optimizeALS(niter, verbose, ncores);
        std::vector<std::unique_ptr<arma::mat>> allH = solver.getAllH();
        std::vector<arma::mat> resolvedH{};
        for (unsigned int i = 0; i < allH.size(); ++i) {
            arma::mat* ptr = allH[i].release();
            resolvedH.push_back(*ptr);
        }
        std::vector<std::unique_ptr<arma::mat>> allV = solver.getAllV();
        std::vector<arma::mat> resolvedV{};
        for (unsigned int i = 0; i < allV.size(); ++i) {
            arma::mat* ptr = allV[i].release();
            resolvedV.push_back(*ptr);
        }
        return {solver.getW(), resolvedH, resolvedV, solver.objErr()};;
    }
}
