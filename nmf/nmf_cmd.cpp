#include "parsecommandline.hpp"
#include "nmf_lib.hpp"
#include <cstdio>
#include <string>


int main(int argc, char *argv[]) {
//  try {
    planc::ParseCommandLine dnd(argc, argv);
    auto libstate = planc::nmflib();
    int status = libstate.runNMF<arma::sp_mat>(dnd.getPlancParams());
    fflush(stdout);
    return status;
//  } catch (const std::exception &e) {
//    INFO << "Exception with stack trace " << std::endl;
//    INFO << e.what();
//  }
}
