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

template<>
std::vector<std::unique_ptr<planc::H5Mat>> planc::initMemMatPtr<planc::H5Mat>(
    const std::vector<std::string> &objectList,
    const std::vector<std::string> &dataPath) {
    std::vector<std::unique_ptr<H5Mat>> matPtrVec;
    for (arma::uword i = 0; i < objectList.size(); ++i) {
        H5Mat h5m(objectList[i], dataPath[i]);
        std::unique_ptr<H5Mat> ptr = std::make_unique<H5Mat>(h5m);
        matPtrVec.push_back(std::move(ptr));
    }
    return matPtrVec;

}

template<>
std::vector<std::unique_ptr<planc::H5SpMat>> planc::initMemMatPtr<planc::H5SpMat>(
    const std::vector<std::string> &objectList,
    const std::vector<std::string> &valuePath,
    const std::vector<std::string> &rowindPath,
    const std::vector<std::string> &colptrPath,
    const std::vector<arma::uword> &nrow,
    const std::vector<arma::uword> &ncol) {
    std::vector<std::unique_ptr<H5SpMat>> matPtrVec;
    for (arma::uword i = 0; i < objectList.size(); ++i) {
        H5SpMat h5spm(objectList[i], rowindPath[i], colptrPath[i], valuePath[i],
            nrow[i], ncol[i]);
        std::unique_ptr<H5SpMat> ptr = std::make_unique<H5SpMat>(h5spm);
        matPtrVec.push_back(std::move(ptr));
    }
    return matPtrVec;

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


//std::map<std::string, planc::runNMFindex> NMFindexmap{{"W", planc::outW},
//                                                      {"H", planc::outH},
//                                                      {"objErr", planc::objErr}};