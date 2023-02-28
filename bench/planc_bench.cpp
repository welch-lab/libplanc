#include <stdio.h>
#include <string>
#include "planc_bench.hpp"
#include "bppnnls.hpp"
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <iostream>
#include <vector>
#include "nmf.hpp"
#include <omp.h>
#define ONE_THREAD_MATRIX_SIZE 2000

namespace planc {

class planc_bench
{
private:
    /* data */
    int p_argc;
    char **p_argv;
    int m_k;
    UWORD m_m, m_n;
    int m_num_it;
    int m_num_nodes;
    std::string m_Afile_name;
    std::string m_Bfile_name;
    std::string m_outputfile_name;
    bool m_compute_error;
    int m_num_k_blocks;
    float m_sparsity;
    template<class NNLSTYPE>
    void callNNLS() {
        #ifdef BUILD_SPARSE
        double t2;
        double tben;
        double titer;
        tic();
        fast_matrix_market::matrix_market_header headerA;
        std::ifstream ifsA(this->m_Afile_name);
        std::vector<UWORD> rowA;
        std::vector<UWORD> colA;
        std::vector<double> valueA;
        fast_matrix_market::read_matrix_market_triplet(ifsA, headerA, rowA, colA, valueA);
        arma::uvec urowa(rowA);
        arma::uvec ucola(colA);
        arma::umat ucoo = join_rows(urowa, ucola).t();
        arma::Col uvala(valueA);
        SP_MAT A(ucoo, uvala, headerA.nrows, headerA.ncols);
        #else
        AMAT A;
        #endif
        // Read data matrices
        fast_matrix_market::matrix_market_header headerB;
        std::ifstream ifs(this->m_Bfile_name);
        std::vector<double> Bmem;
        fast_matrix_market::read_matrix_market_array(ifs, headerB, Bmem);
        AMAT B(Bmem);
        B.reshape(headerB.nrows, headerB.ncols);
        t2 = toc();
        INFO << "Successfully loaded input matrices " << PRINTMATINFO(A) << PRINTMATINFO(B)
             << "(" << t2 << " s)" << std::endl;
        this->m_m = A.n_rows;
        this->m_n = A.n_cols;
        UINT numChunks = m_n / ONE_THREAD_MATRIX_SIZE;
        #pragma omp parallel for schedule(dynamic) private(tictoc_stack)
        for (UINT i = 0; i < numChunks; i++)\
        {
            UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
            UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
            if (spanEnd > m_n - 1)
            {
                spanEnd = m_n - 1;
            }
            #pragma omp critical
            tic();
            BPPNNLS<AMAT, VEC> solveProblem(B.t(), (AMAT)A.cols(spanStart, spanEnd), false);
            solveProblem.solveNNLS();
            titer = toc();
            //#ifdef _VERBOSE
            INFO << " start=" << spanStart
                 << ", end=" << spanEnd
                 << ", tid=" << omp_get_thread_num() << " cpu=" << sched_getcpu()
                 << " time taken=" << titer << std::endl;
            //#endif
        };
    }
        void nnlsParseCommandLine()
        {
            NnlsParseCommandLine npc(p_argc, p_argv);
            npc.parseplancopts();
            this->m_k = npc.lowrankk();
            this->m_Afile_name = npc.input_file_name();
            this->m_Bfile_name = npc.input_file_name_2();
            this->m_num_it = npc.iterations();
            this->m_compute_error = npc.compute_error();
#ifdef BUILD_SPARSE
            callNNLS<BPPNNLS<SP_MAT, AMAT>>();
#else // ifdef BUILD_SPARSE
            callNNLS<BPPNNLS<AMAT, AMAT>>();
#endif
        }

    public:
        planc_bench(int argc, char *argv[])
        {
            this->p_argc = argc;
            this->p_argv = argv;
            this->nnlsParseCommandLine();
        }
    };


}
int main(int argc, char *argv[])
{
    try
    {
        planc::planc_bench dnd(argc, argv);
        fflush(stdout);
    }
    catch (const std::exception &e)
    {
        INFO << "Exception with stack trace " << std::endl;
        INFO << e.what();
    }
}

