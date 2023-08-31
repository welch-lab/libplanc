#pragma once
#include "NMFDriver.hpp"

namespace planc {
class SparseNMFDriver : NMFDriver {
protected:
    static const int kalpha = 5;
    static const int kbeta = 10;
    template <class NMFTYPE>
    void CallNMF();

public:
SparseNMFDriver(ParseCommandLine pc) : NMFDriver(pc) {};
void callNMF()
{
switch (this->m_nmfalgo)
    {
    case MU:
        CallNMF<MUNMF<arma::sp_mat>>();
        break;
    case HALS:
        CallNMF<HALSNMF<arma::sp_mat>>();
        break;
    case ANLSBPP:
        CallNMF<BPPNMF<arma::sp_mat>>();
        break;
    case AOADMM:
        CallNMF<AOADMMNMF<arma::sp_mat>>();
        break;
    case GNSYM:
        CallNMF<GNSYMNMF<arma::sp_mat>>();
        break;
    default:
        ERR << "Unsupported algorithm " << this->m_nmfalgo << std::endl;
    };
}
};
}
