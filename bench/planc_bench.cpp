#include <stdio.h>
#include <string>
#include "planc_bench.hpp"
#include "bppnnls.hpp"
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <iostream>
#include <vector>
#include "nmf.hpp"
#include <omp.h>
#include <string>
#include <algorithm>
#include <cstddef>
#include <vector>
#include "bm.hpp"


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
    PAIRMAT loadedPair;
    AMAT outmat;
    AMAT* outmatptr;
    template<class NNLSTYPE>
    void callNNLS() {
        double tben;
        #ifdef BUILD_SPARSE
        SP_MAT A = std::get<0>(loadedPair);
        #else // ifdef BUILD_SPARSE
        AMAT A = std::get<0>(loadedPair);
        #endif
        AMAT B = std::get<1>(loadedPair);
        UINT numChunks = m_n / ONE_THREAD_MATRIX_SIZE;
        double start = omp_get_wtime();
        #pragma omp parallel for schedule(auto)
        for (UINT i = 0; i < numChunks; i++)
        {
            UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
            UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
            if (spanEnd > m_n - 1)
            {
                spanEnd = m_n - 1;
            }
            // double start = omp_get_wtime();
            BPPNNLS<AMAT, VEC> solveProblem(B.t(), (AMAT)A.cols(spanStart, spanEnd));
            solveProblem.solveNNLS();
            // double end = omp_get_wtime();
            // titer = end - start;
            //#ifdef _VERBOSE
            // INFO << " start=" << spanStart
            //     << ", end=" << spanEnd
            //     << ", tid=" << omp_get_thread_num() << " cpu=" << sched_getcpu() << std::endl;
            //     // << " time taken=" << titer << std::endl;
            //#endif
            (*outmatptr).rows(spanStart, spanEnd) = solveProblem.getSolutionMatrix().t();
        };
        double end = omp_get_wtime();
        tben = end - start;
        INFO << " total nnls runtime=" << tben << std::endl;
    }
    void loadNNLS() {
        #ifdef BUILD_SPARSE
        double t2;
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
        this->loadedPair = std::make_pair(A, B);
        this->m_n = B.n_cols;
        this->m_m = A.n_cols;
        this->m_k = B.n_rows;
        this->outmat = arma::randu<AMAT>(this->m_m, this->m_k);
    };
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
            loadNNLS();
#else // ifdef BUILD_SPARSE
            callNNLS<BPPNNLS<AMAT, AMAT>>();
#endif
        }

    public:
        void call_NNLS() {
            callNNLS<BPPNNLS<SP_MAT, AMAT>>();
        }
        planc_bench(int argc, char **argv)
        {
            this->p_argc = argc;
            this->p_argv = argv;
            this->nnlsParseCommandLine();
        }
        planc_bench(std::string input_file, std::string input_file2) {
            this->m_Afile_name = input_file;
            this->m_Bfile_name = input_file2;
            this->loadNNLS();
        }
    };
}

const char* prefixes[5] =
{
    "frontal_10k",
    "frontal_50k",
    "frontal_100k",
    "frontal_200k",
    "frontal_250k",
};

int main(int argc, char *argv[])
{
   try
    {
        std::vector<class planc::planc_bench> pbvec;
        for(std::string prefix : prefixes){
            std::string sparse = prefix + ".h5.mtx";
            std::string dense = prefix + ".h5.dense.mtx";
            planc::planc_bench dnd(sparse, dense);
            pbvec.push_back(dnd);
        }
        bm::session<float> mySession;
        mySession = bm::run<float, std::milli>([&pbvec](auto &recorder)
            {
            for (int i = 0; i < 5; i++)
            {
                planc::planc_bench *locpbptr = &pbvec[i];
                recorder.record(prefixes[i], [&locpbptr]
                                {locpbptr->call_NNLS();});
            }},
                                               100 /* iterations */);

        for (const auto& record : mySession.records)
        {
            auto name = record.name;
            auto mean = record.mean();
            auto variance = record.variance();
            auto standard_deviation = record.standard_deviation();
        }
        mySession.to_csv("outbench.csv");
        fflush(stdout);
    }
    catch (const std::exception &e)
    {
        INFO << "Exception with stack trace " << std::endl;
        INFO << e.what();
    }
}
