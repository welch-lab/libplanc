//
// Created by andrew on 12/12/2023.
//

#ifndef PLANC_PLANCOPTS_H
#define PLANC_PLANCOPTS_H

#include "utils.h"

namespace planc {
    struct params {
        params() {
            // common to all algorithms.
            this->m_lucalgo = ANLSBPP;
            this->m_input_normalization = NONE;
            this->m_compute_error = false;
            this->m_num_it = 20;
            this->m_num_k_blocks = NULL;
            this->m_dim_tree = true;
            this->m_adj_rand = false;
            // file names
            //this->m_Afile_name = NULL;         // X (features) matrix for jointnmf
            //this->m_outputfile_name = NULL;
            //this->m_Sfile_name = NULL;         // S (connection) matrix for jointnmf
            // std::string m_init_file_name;
            // nmf related values
            this->m_k = 20;
            this->m_globalm = NULL;
            this->m_globaln = NULL;
            this->m_initseed = 193957;
            // algo related values
            this->m_regW = arma::zeros<arma::fvec>(2);
            this->m_regH = arma::zeros<arma::fvec>(2);
            this->m_sparsity = 0.01f;

            // distnmf related values
            this->m_pr = 1;
            this->m_pc = 1;
            this->m_symm_reg = -1;
            this->m_tolerance = -1.;
            // dist ntf
            this->m_num_modes = 1;
            //this->m_dimensions = NULL;
            //this->m_proc_grids = NULL;
            //this->m_regularizers = NULL;
            // LUC params (optional)
            this->m_max_luciters = -1;
            // hiernmf related values
            this->m_num_nodes = 1;
            // jointnmf values
            this->alpha = 0.;
            this->beta = 0.;
            this->feat_type = 2;
            this->conn_type = 0;
            this->m_gamma = 0.;
            this->m_unpartitioned = false;

            // distjointnmf values
            // grid size for the connection matrices (cpr x cpc)
            //this->m_conn_grids = NULL;
            this->m_cpr = 0;
            this->m_cpc = 0;
        }

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
        int m_num_modes = 1;
        arma::uvec m_dimensions;
        arma::uvec m_proc_grids;
        arma::fvec m_regularizers;

        // LUC params (optional)
        int m_max_luciters;

        // hiernmf related values
        int m_num_nodes;

        // jointnmf values
        double alpha, beta;
        int feat_type, conn_type;
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
