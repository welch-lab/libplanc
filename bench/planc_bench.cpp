#include <stdio.h>
#include <string>
#include "planc_bench.hpp"
#include "bppnnls.hpp"
#include "nmf.hpp"
#include <omp.h>
#define ONE_THREAD_MATRIX_SIZE 2000

namespace planc {

class planc_bench
{
private:
    /* data */
    int m_argc;
    char **m_argv;
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
        SP_MAT A;
        #else
        MAT A;
        #endif
        MAT B;
        // Read data matrices
        double t2;
        tic();
        #ifdef BUILD_SPARSE
        A.load(this->m_Afile_name, arma::coord_ascii);
        #else
        A.load(this->m_Afile_name);
        #endif
        B.load(this->m_Bfile_name);
        t2 = toc();
        INFO << "Successfully loaded input matrices " << PRINTMATINFO(A) << PRINTMATINFO(B)
             << "(" << t2 << " s)" << std::endl;
        this->m_m = A.n_rows;
        this->m_n = A.n_cols;
        UINT numChunks = m_n / ONE_THREAD_MATRIX_SIZE;
        // #pragma omp parallel for schedule(dynamic)
        for (UINT i = 0; i < numChunks; i++)
        {
            UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
            UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
            if (spanEnd > m_n - 1)
            {
                spanEnd = m_n - 1;
            }

            BPPNNLS<MAT, VEC> solveProblem(B, (MAT)A.cols(spanStart, spanEnd), false);
            solveProblem.solveNNLS();
            #ifdef _VERBOSE
            INFO << "completed " << worh << " start=" << spanStart
                 << ", end=" << spanEnd
                 // << ", tid=" << omp_get_thread_num() << " cpu=" << sched_getcpu()
                 << " time taken=" << t2 << std::endl;
            #endif
        };
    }
        void nnlsParseCommandLine()
        {
            NnlsParseCommandLine pc(this->m_argc, this->m_argv);
            pc.parseplancopts();
            this->m_k = pc.lowrankk();
            this->m_Afile_name = pc.input_file_name();
            this->m_sparsity = pc.sparsity();
            this->m_num_it = pc.iterations();
            this->m_compute_error = pc.compute_error();
            pc.printConfig();
#ifdef BUILD_SPARSE
            callNNLS<BPPNNLS<SP_MAT, MAT>>();
#else // ifdef BUILD_SPARSE
            callNNLS<BPPNNLS<MAT, MAT>>();
#endif
        }

    public:
        planc_bench(int argc, char *argv[])
        {
            this->m_argc = argc;
            this->m_argv = argv;
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

