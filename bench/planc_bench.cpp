#include "planc_bench.hpp"
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include "bm.hpp"
#include "nnls_lib.hpp"
#include "utils.hpp"


namespace planc {
    std::pair<arma::sp_mat, arma::mat> loadBenchSparse(params params) {
        double t2;
        tic();
        fast_matrix_market::matrix_market_header headerA;
        std::ifstream ifsA(params.getMAfileName());
        std::vector<arma::uword> rowA;
        std::vector<arma::uword> colA;
        std::vector<double> valueA;
        fast_matrix_market::read_matrix_market_triplet(ifsA, headerA, rowA, colA, valueA);
        arma::uvec urowa(rowA);
        arma::uvec ucola(colA);
        arma::umat ucoo = join_rows(urowa, ucola).t();
        arma::Col uvala(valueA);
        arma::sp_mat A(ucoo, uvala, headerA.nrows, headerA.ncols);
        // Read data matrices
        fast_matrix_market::matrix_market_header headerB;
        std::ifstream ifs(params.getMBfileName());
        std::vector<double> Bmem;
        fast_matrix_market::read_matrix_market_array(ifs, headerB, Bmem);
        arma::mat B(Bmem);
        B.reshape(headerB.nrows, headerB.ncols);
        t2 = toc();
        INFO << "Successfully loaded input matrices " << PRINTMATINFO(A) << PRINTMATINFO(B)
            << "(" << t2 << " s)" << std::endl;
        params.setMK(B.n_rows);
        return std::make_pair(A, B);
    }

    std::pair<arma::mat, arma::mat> loadBenchDense(params params) {
        tic();
        // Read data matrices
        fast_matrix_market::matrix_market_header headerB;
        std::ifstream ifs(params.m_Bfile_name);
        std::vector<double> Bmem;
        fast_matrix_market::read_matrix_market_array(ifs, headerB, Bmem);
        arma::mat B(Bmem);
        B.reshape(headerB.nrows, headerB.ncols);
        arma::mat A(Bmem);
        double t2 = toc();
        INFO << "Successfully loaded input matrices " << PRINTMATINFO(A) << PRINTMATINFO(B)
            << "(" << t2 << " s)" << std::endl;
        return std::make_pair(A, B);
    }

    class planc_bench {
        std::variant<std::pair<arma::sp_mat, arma::mat>, std::pair<arma::mat, arma::mat>> pairvar;
        params instanceParams;
    public:
        planc_bench(const params& params, const argparse::ArgumentParser&type_args) {
            this->instanceParams = params;
            if (type_args.get<bool>("sparse")) {
                this->pairvar = loadBenchSparse(params);
            }
            else if (type_args.get<bool>("dense")) {
                this->pairvar = loadBenchDense(params);
            }
        }
        nnlsOutput runBench(std::variant<nnlslib<arma::sp_mat>, nnlslib<arma::mat>> libstate) {
            auto unpackvar = std::visit([](auto &&arg) -> std::variant<arma::sp_mat, arma::mat> {return arg.first;}, this->pairvar);
            arma::mat unpackstat = std::visit([](auto &&arg) -> arma::mat {return arg.second;}, this->pairvar);
            auto functor = [this, unpackstat, unpackvar](auto& arg) -> nnlsOutput {return arg.runbppnnls(unpackstat, unpackvar, this->instanceParams.n_cores());};
            return std::visit(functor, libstate);
        }

    };

    const char* prefixes[5] =
    {
        "frontal_10k",
        "frontal_50k",
        "frontal_100k",
        "frontal_200k",
        "frontal_250k",
    };
}

    int main(int argc, char *argv[])
    {
        try
        {   argparse::ArgumentParser type_args;
            type_args.add_argument("sparse");
            type_args.add_argument("dense");
            auto secondary_args = type_args.parse_known_args(argc, argv);
            std::variant<planc::nnlslib<arma::sp_mat>, planc::nnlslib<arma::mat>> libstate{};
            if (type_args.get<bool>("sparse")) {
                libstate = planc::nnlslib<arma::sp_mat>();
            }
            else if (type_args.get<bool>("dense")) {
                libstate = planc::nnlslib<arma::mat>();
            }
            planc::BenchParseCommandLine bpc;
            const planc::params initialparams = bpc.getPlancParams(secondary_args.size(),
                reinterpret_cast<const char* const*>(reinterpret_cast<char*>(secondary_args.data())));
            std::vector<planc::planc_bench> pbvec;
            for(std::string prefix : planc::prefixes) {
                planc::params paramCopy = initialparams;
                std::string sparse = prefix + ".h5.mtx";
                std::string dense = prefix + ".h5.dense.mtx";
                paramCopy.setMAfileName(sparse);
                paramCopy.setMBFileName(dense);
                planc::planc_bench dnd(paramCopy, type_args);
                pbvec.push_back(dnd);
            }
            const bm::session<float> mySession = bm::run<float, std::milli>([&pbvec, libstate](auto&recorder) {
                                                                          for (int i = 0; i < 5; i++) {
                                                                              planc::planc_bench* locpbptr = &pbvec[i];
                                                                              recorder.record(planc::prefixes[i], [&locpbptr, libstate] {
                                                                                  locpbptr->runBench(libstate);
                                                                              });
                                                                          }
                                                                      },
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
        } return 1;
    }