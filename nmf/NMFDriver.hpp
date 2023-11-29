#pragma once

#include "utils.hpp"
#include "parsecommandline.hpp"
#include "aoadmm.hpp"
#include "gnsym.hpp"
#include "mu.hpp"
#include "hals.hpp"
#include "bppnmf.hpp"
#include <cstdio>
#include <string>
#include <utility>


namespace planc {
class NMFDriver
{
protected:
    int m_argc;
    char **m_argv;
    int m_k;
    arma::uword m_m, m_n;
    std::string m_Afile_name;
    std::string m_outputfile_name;
    std::string m_w_init_file_name;
    std::string m_h_init_file_name;
    int m_num_it;
    arma::fvec m_regW;
    arma::fvec m_regH;
    double m_symm_reg;
    int m_symm_flag;
    bool m_adj_rand;
    algotype m_nmfalgo;
    double m_sparsity;
    unsigned int m_compute_error;
    normtype m_input_normalization;
    int m_max_luciters;
    int m_initseed;

    // Variables for creating random matrix
    static const int kW_seed_idx = 1210873;
    static const int kprimeoffset = 17;
    static const int kalpha = 1;
    static const int kbeta = 0;
    template<class NMFTYPE>
    void CallNMF();
    void parseCommandLine(ParseCommandLine pc);

public:
explicit NMFDriver(ParseCommandLine pc)
{
    this->parseCommandLine(std::move(pc));
}
virtual void callNMF()
{
switch (this->m_nmfalgo)
    {
    case MU:
        CallNMF<MUNMF<arma::mat>>();
        break;
    case HALS:
        CallNMF<HALSNMF<arma::mat>>();
        break;
    case ANLSBPP:
        CallNMF<BPPNMF<arma::mat>>();
        break;
    case AOADMM:
        CallNMF<AOADMMNMF<arma::mat>>();
        break;
    case GNSYM:
        CallNMF<GNSYMNMF<arma::mat>>();
        break;
    default:
        ERR << "Unsupported algorithm " << this->m_nmfalgo << std::endl;
    }
}
};
    class SparseNMFDriver : NMFDriver {
    protected:
        static const int kalpha = 5;
        static const int kbeta = 10;
        template <class NMFTYPE>
        void CallNMF();

    public:
        explicit SparseNMFDriver(ParseCommandLine pc) : NMFDriver(std::move(pc)) {};
        void callNMF() override
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
            }
        }
    };

}
