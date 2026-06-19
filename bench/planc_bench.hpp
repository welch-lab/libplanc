#pragma once

#ifndef ARMA_DONT_USE_WRAPPER
#define ARMA_DONT_USE_WRAPPER
#endif
#define ONE_THREAD_MATRIX_SIZE 2000
#include "parsecommandline.hpp"
#include <vector>
namespace planc {
        template <typename VT>
        struct array_matrix {
                arma::uword nrows = 0, ncols = 0;
                std::vector<VT> vals;
        };

        class BenchParseCommandLine: public BaseParser
        {
        public:

                BenchParseCommandLine() : BaseParser("planc_bench", "1.1.0")
                {
                        this->add_argument("-t", "--iterations")
                                .default_value(20)
                                .scan<'i', int>()
                                .help("iterations");
                        this->add_argument("-o", "--output")
                                .default_value("")
                                .help("output");
                        this->add_argument("-n", "--cores")
                                .default_value(1)
                                .scan<'i', int>()
                                .help("number of parallel cores");
                        this->add_argument("type").choices("dense", "sparse")
                        .required();
                }
                void initClStruct() const override {
                        clStruct->m_type = matrixmap.at(this->get("type"));
                        clStruct->m_num_it = this->get<int>("-t");
                        //clStruct->m_k = this->get<int>("-k");
                        //clStruct->m_Afile_name = this->get("-i");
                        clStruct->m_outputfile_name = this->get("-o");
                        clStruct->set_n_cores(this->get<int>("-n"));
                        //clStruct->m_initseed = this->get<int>("--seed");
                }
                //void printConfig() const override {};
        };
}
