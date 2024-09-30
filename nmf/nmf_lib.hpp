//
// Created by andrew on 12/12/2023.
//

#ifndef PLANC_NMF_LIB_HPP
#define PLANC_NMF_LIB_HPP
#include "nmflib_export.h"
#include "NMFDriver.hpp"
#include <data.hpp>


extern "C" {
#include "detect_blas.h"
}


namespace planc {

    extern void NMFLIB_NO_EXPORT openblas_pthread_off(openblas_handle_t);
    extern void NMFLIB_NO_EXPORT openblas_pthread_on(openblas_handle_t);
    template<typename eT>
    struct NMFLIB_EXPORT nmfOutput {
        nmfOutput() = default;
        ~nmfOutput() = default;
        arma::Mat<eT> outW;
        arma::Mat<eT> outH;
        double objErr;
    };
    template<typename eT>
    struct NMFLIB_EXPORT inmfOutput {
        inmfOutput() = default;
        ~inmfOutput() = default;
        arma::Mat<eT> outW;
        std::vector<arma::Mat<eT>> outHList;
        std::vector<arma::Mat<eT>> outVList;
        double objErr;
    };
    template<class T, typename eT = typename T::elem_type>
    std::vector<std::unique_ptr<T>> NMFLIB_NO_EXPORT initMemMatPtr(std::vector<T> objectList)
    {
        std::vector<std::unique_ptr<T>> matPtrVec;
        for (arma::uword i = 0; i < objectList.size(); ++i)
        {
            T E = T(objectList[i]);
            std::unique_ptr<T> ptr = std::make_unique<T>(E);
            matPtrVec.push_back(std::move(ptr));
        }
        return matPtrVec;
    }
    template<typename T, typename eT = typename T::elem_type>
    class NMFLIB_EXPORT nmflib {
    public:
        nmflib() {
            openblas_pthread_off(get_openblas_handle());
        }

        ~nmflib() = default;

        static struct nmfOutput<eT> nmf(const T &x, const arma::uword &k, const arma::uword &niter, const std::string &algo, const int &nCores,
            const arma::Mat<eT> &Winit = arma::Mat<eT>(), const arma::Mat<eT> &Hinit = arma::Mat<eT>());
        static struct nmfOutput<eT> symNMF(const T& x, const arma::uword& k, const arma::uword& niter, const double& lambda, const std::string& algo, const int& nCores,
                         const arma::Mat<eT>& Hinit);
        static struct inmfOutput<eT> bppinmf(const std::vector<T> &objectList, const arma::uword &k, const double &lambda,
                                          const arma::uword &niter, const bool &verbose, const int& ncores);
        static struct inmfOutput<eT> bppinmf(const std::vector<T> &objectList, const arma::uword &k, const double &lambda,
                   const arma::uword &niter, const bool &verbose,
                   const std::vector<arma::mat> &HinitList, const std::vector<arma::mat> &VinitList, const arma::mat &Winit,
                   const int& ncores);
        static int runNMF(params opts) {
            NMFDriver<T> myNMF(opts);
            myNMF.callNMF();
            return 0;
        }
        static int runINMF(params opts) {
            NMFDriver<T> myNMF(opts);
            myNMF.callNMF();
            return 0;
        }
    };

    template<typename eT>
    struct NMFLIB_EXPORT InstOUT : nmfOutput<eT> {};

    template<typename T>
    struct NMFLIB_EXPORT InstCLASS : nmflib<T>{};





    //enum NMFLIB_EXPORT runNMFindex {outW, outH, objErr};

    //extern std::map<std::string, runNMFindex> NMFindexmap;ea
}

                                                    // template<typename T>
                                                    // extern inline NMFLIB_EXPORT planc::nmflib<T> nmflib{};
#endif //PLANC_NMF_LIB_HPP
