#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif
#include "bppnmf.hpp"
#include <vector>
#include <unique_ptr>

namespace planc {

template <class T>
class BPPINMF<T> {
private:
    std::vector<std::unique_ptr<T>> Ei;
    std::vector<std::unique_ptr<arma::mat>> Hi;
    std::vector<std::unique_ptr<arma::mat>> Vi;
    std::unique_ptr<arma::mat> W;
public:

    BPPINMF(std::vector<std::unique_ptr<T>> Es,
            std::vector<std::unique_ptr<arma::mat>> Hs,
            std::vector<std::unique_ptr<arma::mat>> Vs,
            std::unique_ptr<arma::mat> W) {
        Ei = Es;
        Hi = Hs;
        Vi = Vs;
        Wptr = W;

    }
};

}