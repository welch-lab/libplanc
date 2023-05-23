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

public:
    BPPINMF() : INMF(Es, Hs, Vs, W) {

    }
};

}
