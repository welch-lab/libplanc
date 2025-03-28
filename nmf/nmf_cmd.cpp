#include "parsecommandline.hpp"
#include "nmf_lib.hpp"
#include <cstdio>

int main(int argc, char *argv[]) {
//  try {

    argparse::ArgumentParser type_args;
    type_args.add_argument("sparse");
    type_args.add_argument("dense");
    auto secondary_args = type_args.parse_known_args(argc, argv);
    planc::ParseCommandLine dnd;
    std::variant<planc::nmflib<arma::sp_mat>, planc::nmflib<arma::mat>> libstate{};
    planc::params params = dnd.getPlancParams(secondary_args.size(),
        reinterpret_cast<const char* const*>(reinterpret_cast<char*>(secondary_args.data())));
    if (type_args.get<bool>("sparse")) {
        libstate = planc::nmflib<arma::sp_mat>();
    }
    else if (type_args.get<bool>("dense")) {
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
