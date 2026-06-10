#include "parsecommandline.hpp"
#include "nmf_lib.hpp"
#include <cstdio>

int main(int argc, char *argv[]) {
//  try {

    planc::ParseCommandLine args;
    std::variant<planc::nmflib<arma::sp_mat>, planc::nmflib<arma::mat>> libstate{};
    planc::params params = args.getPlancParams({argv, argv + argc});
    if (params.m_type == SPARSE) {
        libstate = planc::nmflib<arma::sp_mat>();
    }
    else if (params.m_type == DENSE) {
        libstate = planc::nmflib<arma::mat>();
    }
    auto functor = [params](auto& arg) -> int {return arg.runNMF(params);};
    int status = std::visit(functor, libstate);
    fflush(stdout);
    return status;
//  } catch (const std::exception &e) {
//    INFO << "Exception with stack trace " << std::endl;
//    INFO << e.what();
//  }
}
