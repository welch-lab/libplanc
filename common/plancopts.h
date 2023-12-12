//
// Created by andrew on 12/12/2023.
//

#ifndef PLANC_PLANCOPTS_H
#define PLANC_PLANCOPTS_H

#include "utils.h"

namespace planc {
    struct params {
        // common to all algorithms.
        algotype m_lucalgo;
        normtype m_input_normalization;
        bool m_compute_error;
        int m_num_it;
        int m_num_k_blocks;
        bool m_dim_tree;
        bool m_adj_rand;

        // file names
        std::string m_Afile_name;         // X (features) matrix for jointnmf
        std::string m_outputfile_name;
        std::string m_Sfile_name;         // S (connection) matrix for jointnmf
        // std::string m_init_file_name;

        // nmf related values
        arma::uword m_k;
        arma::uword m_globalm;
        arma::uword m_globaln;
        int m_initseed;

        // algo related values
        arma::fvec m_regW;
        arma::fvec m_regH;
        float m_sparsity;

        // distnmf related values
        int m_pr;
        int m_pc;
        double m_symm_reg;
        double m_tolerance;

        // dist ntf
        int m_num_modes;
        arma::uvec m_dimensions;
        arma::uvec m_proc_grids;
        arma::fvec m_regularizers;

        // LUC params (optional)
        int m_max_luciters;

        // hiernmf related values
        int m_num_nodes;

        // jointnmf values
        double alpha, beta;
        bool feat_type, conn_type;
        double m_gamma;
        int m_unpartitioned;

        // distjointnmf values
        // grid size for the connection matrices (cpr x cpc)
        arma::uvec m_conn_grids;
        int m_cpr;
        int m_cpc;
    };
}

#endif //PLANC_PLANCOPTS_H
