//
// Created by andrew on 12/12/2023.
//
#include "nmf_lib.inl"


void planc::openblas_pthread_off(openblas_handle_t libloc) {
    if (is_openmp()) {
        if (const std::function openblas_parallel = get_openblas_parallel(libloc))
            {
                if (openblas_parallel() == 1) {
                    const std::function openblas_set = get_openblas_set(libloc);
                    openblas_set(1);
                }
            }
    }
}


void planc::openblas_pthread_on(openblas_handle_t libloc) {if (is_openmp()) {
    if (const std::function openblas_parallel = get_openblas_parallel(libloc))
    {
        if (openblas_parallel() == 1) {
            const std::function openblas_set = get_openblas_set(libloc);
            openblas_set(0);
        }
    }
}
}

planc::nmfOutput<double> (*ptrnmf)(const arma::Mat<double>&x, const arma::uword&k, const arma::uword&niter,
                                   const std::string&algo, const int&nCores, const arma::Mat<double>&Winit,
                                   const arma::Mat<double>&Hinit)(planc::nmflib<arma::mat>::nmf);
planc::nmfOutput<double> (*ptrspnmf)(const arma::SpMat<double>&x, const arma::uword&k, const arma::uword&niter,
                                   const std::string&algo, const int&nCores, const arma::Mat<double>&Winit,
                                   const arma::Mat<double>&Hinit)(planc::nmflib<arma::sp_mat>::nmf);
planc::nmfOutput<double> (*ptrsymnmf)(const arma::Mat<double>&x, const arma::uword&k, const arma::uword&niter, const double& lambda,
                                   const std::string&algo, const int&nCores, const arma::Mat<double>&Hinit)(planc::nmflib<arma::mat>::symNMF);
planc::nmfOutput<double> (*ptrspsymnmf)(const arma::SpMat<double>&x, const arma::uword&k, const arma::uword&niter, const double& lambda,
                                   const std::string&algo, const int&nCores, const arma::Mat<double>&Hinit)(planc::nmflib<arma::sp_mat>::symNMF);
planc::inmfOutput<double> (*ptrinmf)(std::vector<arma::mat> objectlist, arma::uword k, double lambda, arma::uword niter,
                                     bool verbose, const int &nCores)(planc::nmflib<arma::mat>::bppinmf);
planc::inmfOutput<double> (*ptrspinmf)(std::vector<arma::sp_mat> objectlist, arma::uword k, double lambda, arma::uword niter,
                                     bool verbose, const int &nCores)(planc::nmflib<arma::sp_mat>::bppinmf);
planc::inmfOutput<double> (*ptrh5inmf)(std::vector<planc::H5Mat> objectlist, arma::uword k, double lambda, arma::uword niter,
                                     bool verbose, const int &nCores)(planc::nmflib<planc::H5Mat, double>::bppinmf);
planc::inmfOutput<double> (*ptrh5spinmf)(std::vector<planc::H5SpMat> objectlist, arma::uword k, double lambda, arma::uword niter,
                                     bool verbose, const int &nCores)(planc::nmflib<planc::H5SpMat, double>::bppinmf);
//std::map<std::string, planc::runNMFindex> NMFindexmap{{"W", planc::outW},
//                                                      {"H", planc::outH},
//                                                      {"objErr", planc::objErr}};