//
// Created by andrew on 12/12/2023.
//

#ifndef PLANC_NMF_LIB_HPP
#define PLANC_NMF_LIB_HPP
#include "nmflib_export.h"
#include "plancopts.h"

extern "C" {
#include "detect_blas.h"
}
namespace planc {
    class NMFLIB_EXPORT nmflib {
    public:
        NMFLIB_EXPORT nmflib();

        template<typename T>
        int NMFLIB_EXPORT runNMF(params opts);

        template<typename T>
        int NMFLIB_EXPORT runINMF(params opts);

    private:
        static void NMFLIB_NO_EXPORT openblas_pthread_off(openblas_handle_t);
        static void NMFLIB_NO_EXPORT openblas_pthread_on(openblas_handle_t);
    };

}

#endif //PLANC_NMF_LIB_HPP
