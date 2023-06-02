#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif
#include "bppnnls.hpp"
#include "inmf.hpp"

namespace planc {

template <class T>
class BPPINMF : INMF<T> {
private:
    void solveH() {
        // implement
    }
    void solveV() {
        // implement
    }
public:
    BPPINMF(std::vector<std::unique_ptr<T>> &Ei, arma::uword k, double lambda) : INMF<T>(Ei, k, lambda) {

    }

    void optimizeALS() {
        // execute private functions here
    }
};

}
