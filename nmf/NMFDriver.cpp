#include "NMFDriver.hpp"
#include "aoadmm.hpp"
#include "bppnmf.hpp"
#include "hals.hpp"
#include "mu.hpp"
#include "gnsym.hpp"

namespace planc
{
    template<class NMFTYPE>
    void NMFDriver::CallNMF()
    {
        arma::mat A;

        // Generate/Read data matrix
        double t2;
        if (!this->m_Afile_name.empty())
        {
            tic();
            A.load(this->m_Afile_name);
            t2 = toc();
            INFO << "Successfully loaded input matrix A " << PRINTMATINFO(A)
                 << "(" << t2 << " s)" << std::endl;
            this->m_m = A.n_rows;
            this->m_n = A.n_cols;
        }
        else
        {
            arma::arma_rng::set_seed(planc::NMFDriver::kW_seed_idx);
            std::string rand_prefix("rand_");
            std::string type = this->m_Afile_name.substr(rand_prefix.size());
            assert(type == "normal" || type == "lowrank" || type == "uniform");
            tic();
            if (type == "uniform")
            {
                if (this->m_symm_flag)
                {
                    A = arma::randu<arma::mat>(this->m_m, this->m_n);
                    A = 0.5 * (A + A.t());
                }
                else
                {
                    A = arma::randu<arma::mat>(this->m_m, this->m_n);
                }
            }
            else if (type == "normal")
            {
                if (this->m_symm_flag)
                {
                    A = arma::randn<arma::mat>(this->m_m, this->m_n);
                    A = 0.5 * (A + A.t());
                }
                else
                {
                    A = arma::randn<arma::mat>(this->m_m, this->m_n);
                }
                A.elem(find(A < 0)).zeros();
            }
            else
            {
                if (this->m_symm_flag)
                {
                    auto Wtrue = arma::randu<arma::mat>(this->m_m, this->m_k);
                    A = Wtrue * Wtrue.t();

                    // Free auxiliary variables
                    Wtrue.clear();
                }
                else
                {
                    auto Wtrue = arma::randu<arma::mat>(this->m_m, this->m_k);
                    auto Htrue = arma::randu<arma::mat>(this->m_k, this->m_n);
                    A = Wtrue * Htrue;

                    // Free auxiliary variables
                    Wtrue.clear();
                    Htrue.clear();
                }
            }
            if (this->m_adj_rand)
            {
                A = kalpha * (A) + kbeta;
                A = ceil(A);
            }
            t2 = toc();
            INFO << "generated random matrix A " << PRINTMATINFO(A)
                 << "(" << t2 << " s)" << std::endl;
        }

        // Normalize the input matrix
        if (this->m_input_normalization != NONE)
        {
            tic();
            if (this->m_input_normalization == L2NORM)
            {
                A = arma::normalise(A);
            }
            else if (this->m_input_normalization == MAXNORM)
            {
                double maxnorm = 1 / A.max();
                A = maxnorm * A;
            }
            t2 = toc();
            INFO << "Normalized A (" << t2 << "s)" << std::endl;
        }

        // Set parameters and call NMF
        arma::arma_rng::set_seed(this->m_initseed);
        arma::mat W;
        arma::mat H;
        if (!this->m_h_init_file_name.empty() && !this->m_w_init_file_name.empty())
        {
            W.load(m_w_init_file_name, arma::coord_ascii);
            H.load(m_h_init_file_name, arma::coord_ascii);
            this->m_k = W.n_cols;
        }
        else
        {
            W = arma::randu<arma::mat>(this->m_m, this->m_k);
            H = arma::randu<arma::mat>(this->m_n, this->m_k);
        }
        if (this->m_symm_flag)
        {
            double meanA = arma::mean(arma::mean(A));
            H = 2 * std::sqrt(meanA / this->m_k) * H;
            W = H;
            if (this->m_symm_reg == 0.0)
            {
                double symreg = A.max();
                this->m_symm_reg = symreg * symreg;
            }
        }

        NMFTYPE nmfAlgorithm(A.t(), W, H);
        nmfAlgorithm.num_iterations(this->m_num_it);
        nmfAlgorithm.symm_reg(this->m_symm_reg);
        nmfAlgorithm.updalgo(this->m_nmfalgo);
        // Always compute error for shared memory case
        // nmfAlgorithm.compute_error(this->m_compute_error);

        if (!this->m_regW.empty())
        {
            nmfAlgorithm.regW(this->m_regW);
        }
        if (!this->m_regH.empty())
        {
            nmfAlgorithm.regH(this->m_regH);
        }

        INFO << "completed constructor" << PRINTMATINFO(A) << std::endl;
        tic();
        nmfAlgorithm.computeNMF();
        t2 = toc();
        INFO << "time taken:" << t2 << std::endl;

        // Save the factor matrices
        if (!this->m_outputfile_name.empty())
        {
            std::string WfileName = this->m_outputfile_name + "_W";
            std::string HfileName = this->m_outputfile_name + "_H";

            nmfAlgorithm.getLeftLowRankFactor().save(WfileName, arma::raw_ascii);
            nmfAlgorithm.getRightLowRankFactor().save(HfileName, arma::raw_ascii);
        }
    }
    template void NMFDriver::CallNMF<MUNMF<arma::mat>>();
    template void NMFDriver::CallNMF<BPPNMF<arma::mat>>();
    template void NMFDriver::CallNMF<GNSYMNMF<arma::mat>>();
    template void NMFDriver::CallNMF<AOADMMNMF<arma::mat>>();
    template void NMFDriver::CallNMF<HALSNMF<arma::mat>>();

    void NMFDriver::parseCommandLine(ParseCommandLine pc)
    {
        pc.parseplancopts();
        this->m_nmfalgo = pc.lucalgo();
        this->m_k = pc.lowrankk();
        this->m_Afile_name = pc.input_file_name();
        this->m_sparsity = pc.sparsity();
        this->m_num_it = pc.iterations();
        this->m_regW = pc.regW();
        this->m_regH = pc.regH();
        this->m_m = pc.globalm();
        this->m_n = pc.globaln();
        this->m_compute_error = pc.compute_error();
        this->m_symm_reg = pc.symm_reg();
        this->m_symm_flag = 0;
        this->m_adj_rand = pc.adj_rand();
        this->m_max_luciters = pc.max_luciters();
        this->m_initseed = pc.initseed();
        this->m_outputfile_name = pc.output_file_name();

        // Put in the default LUC iterations
        if (this->m_max_luciters == -1)
        {
            this->m_max_luciters = this->m_k;
        }

        // check the conditions for symm nmf
        if (this->m_symm_reg != -1)
        {
            this->m_symm_flag = 1;
            if (this->m_m != this->m_n)
            {
                // TODO{seswar3}: Add file prefix check
                ERR << "Symmetric Regularization enabled"
                    << " and input matrix is not square"
                    << "::m::" << this->m_m << "::n::" << this->m_n
                    << std::endl;
                return;
            }
            if ((this->m_nmfalgo != ANLSBPP) && (this->m_nmfalgo != GNSYM))
            {
                ERR << "Symmetric Regularization enabled "
                    << "is only enabled for ANLSBPP and GNSYM"
                    << std::endl;
                return;
            }
        }
        pc.printConfig();
        }
 // class NMFDriver

    template<class NMFTYPE>
    void SparseNMFDriver::CallNMF()
    {
        arma::sp_mat A;

        // Generate/Read data matrix
        double t2;
        if (!this->m_Afile_name.empty())
        {
            tic();
            A.load(this->m_Afile_name, arma::coord_ascii);
            t2 = toc();
            INFO << "Successfully loaded input matrix A " << PRINTMATINFO(A)
                 << "(" << t2 << " s)" << std::endl;
            this->m_m = A.n_rows;
            this->m_n = A.n_cols;
        }
        else
        {
            arma::arma_rng::set_seed(planc::SparseNMFDriver::kW_seed_idx);
            std::string rand_prefix("rand_");
            std::string type = this->m_Afile_name.substr(rand_prefix.size());
            assert(type == "normal" || type == "lowrank" || type == "uniform");
            tic();
            if (type == "uniform")
            {
                if (this->m_symm_flag)
                {
                    double sp = 0.5 * this->m_sparsity;
                    A = arma::sprandu<arma::sp_mat>(this->m_m, this->m_n, sp);
                    A = 0.5 * (A + A.t());
                }
                else
                {
                    A = arma::sprandu<arma::sp_mat>(this->m_m, this->m_n,
                                                    this->m_sparsity);
                }
            }
            else if (type == "normal")
            {
                if (this->m_symm_flag)
                {
                    double sp = 0.5 * this->m_sparsity;
                    A = arma::sprandn<arma::sp_mat>(this->m_m, this->m_n, sp);
                    A = 0.5 * (A + A.t());
                }
                else
                {
                    A = arma::sprandn<arma::sp_mat>(this->m_m, this->m_n,
                                                    this->m_sparsity);
                }
            }
            else if (type == "lowrank")
            {
                if (this->m_symm_flag)
                {
                    double sp = 0.5 * this->m_sparsity;
                    auto mask = arma::sprandu<arma::sp_mat>(this->m_m, this->m_n,
                                                                    sp);
                    mask = 0.5 * (mask + mask.t());
                    mask = arma::spones(mask);
                    arma::mat Wtrue = arma::randu(this->m_m, this->m_k);
                    A = arma::sp_mat(mask % (Wtrue * Wtrue.t()));

                    // Free auxiliary space
                    Wtrue.clear();
                    mask.clear();
                }
                else
                {
                    auto mask = arma::sprandu<arma::sp_mat>(this->m_m, this->m_n,
                                                                    this->m_sparsity);
                    mask = arma::spones(mask);
                    arma::mat Wtrue = arma::randu(this->m_m, this->m_k);
                    arma::mat Htrue = arma::randu(this->m_k, this->m_n);
                    A = arma::sp_mat(mask % (Wtrue * Htrue));

                    // Free auxiliary space
                    Wtrue.clear();
                    Htrue.clear();
                    mask.clear();
                }
            }
            // Adjust and project non-zeros
            arma::sp_mat::iterator start_it = A.begin();
            arma::sp_mat::iterator end_it = A.end();
            for (arma::sp_mat::iterator it = start_it; it != end_it; ++it)
            {
                double curVal = (*it);
                if (this->m_adj_rand)
                {
                    (*it) = ceil(kalpha * curVal + kbeta);
                }
                if ((*it) < 0)
                    (*it) = kbeta;
            }
            t2 = toc();
            INFO << "generated random matrix A " << PRINTMATINFO(A)
                 << "(" << t2 << " s)" << std::endl;
        }

        // Normalize the input matrix
        if (this->m_input_normalization != NONE)
        {
            tic();
            if (this->m_input_normalization == L2NORM)
            {
                A = arma::normalise(A);
            }
            else if (this->m_input_normalization == MAXNORM)
            {
                double maxnorm = 1 / A.max();
                A = maxnorm * A;
            }
            t2 = toc();
            INFO << "Normalized A (" << t2 << "s)" << std::endl;
        }

        // Set parameters and call NMF
        arma::arma_rng::set_seed(this->m_initseed);
        arma::mat W;
        arma::mat H;
        if (!this->m_h_init_file_name.empty() && !this->m_w_init_file_name.empty())
        {
            W.load(m_w_init_file_name, arma::coord_ascii);
            H.load(m_h_init_file_name, arma::coord_ascii);
            this->m_k = W.n_cols;
        }
        else
        {
            W = arma::randu<arma::mat>(this->m_m, this->m_k);
            H = arma::randu<arma::mat>(this->m_n, this->m_k);
        }
        if (this->m_symm_flag)
        {
            double meanA = arma::mean(arma::mean(A));
            H = 2 * std::sqrt(meanA / this->m_k) * H;
            W = H;
            if (this->m_symm_reg == 0.0)
            {
                double symreg = A.max();
                this->m_symm_reg = symreg * symreg;
            }
        }

        NMFTYPE nmfAlgorithm(A.t(), W, H);
        nmfAlgorithm.num_iterations(this->m_num_it);
        nmfAlgorithm.symm_reg(this->m_symm_reg);
        nmfAlgorithm.updalgo(this->m_nmfalgo);
        // Always compute error for shared memory case
        // nmfAlgorithm.compute_error(this->m_compute_error);

        if (!this->m_regW.empty())
        {
            nmfAlgorithm.regW(this->m_regW);
        }
        if (!this->m_regH.empty())
        {
            nmfAlgorithm.regH(this->m_regH);
        }

        INFO << "completed constructor" << PRINTMATINFO(A) << std::endl;
        tic();
        nmfAlgorithm.computeNMF();
        t2 = toc();
        INFO << "time taken:" << t2 << std::endl;

        // Save the factor matrices
        if (!this->m_outputfile_name.empty())
        {
            std::string WfileName = this->m_outputfile_name + "_W";
            std::string HfileName = this->m_outputfile_name + "_H";

            nmfAlgorithm.getLeftLowRankFactor().save(WfileName, arma::raw_ascii);
            nmfAlgorithm.getRightLowRankFactor().save(HfileName, arma::raw_ascii);
        }
    }
    template void SparseNMFDriver::CallNMF<MUNMF<arma::sp_mat>>();
    template void SparseNMFDriver::CallNMF<BPPNMF<arma::sp_mat>>();
    template void SparseNMFDriver::CallNMF<GNSYMNMF<arma::sp_mat>>();
    template void SparseNMFDriver::CallNMF<AOADMMNMF<arma::sp_mat>>();
    template void SparseNMFDriver::CallNMF<HALSNMF<arma::sp_mat>>();
    // class SparseNMFDriver


} // namespace planc
