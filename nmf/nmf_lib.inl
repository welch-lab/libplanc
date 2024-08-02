#pragma once

#include "nmf_lib.hpp"
namespace planc {
    template<typename T, typename eT>
    nmfOutput<eT> NMFLIB_EXPORT nmflib<T, eT>::nmf(const T& x, const arma::uword& k, const arma::uword& niter,
                                               const std::string& algo, const int& nCores, const arma::Mat<eT>& Winit, const arma::Mat<eT>& Hinit) {
        internalParams options(x, Winit, Hinit);
        options.m_k = k;
        options.m_num_it = niter;
        options.setMLucalgo(algo);
        EmbeddedNMFDriver nmfRunner(options);
        nmfRunner.callNMF();
        nmfOutput<eT> outlist{};
        outlist.outW = nmfRunner.getLlf();
        outlist.outH = nmfRunner.getRlf();
        outlist.objErr = nmfRunner.getobjErr();
        return outlist;
    }

}