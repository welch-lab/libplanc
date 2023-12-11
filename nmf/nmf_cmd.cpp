#include "NMFDriver.hpp"
#include <cstdio>
#include <string>


int main(int argc, char *argv[]) {
  try {
    planc::ParseCommandLine dnd(argc, argv);
    planc::NMFDriver<arma::mat> myNMF(dnd);
    myNMF.callNMF();
    fflush(stdout);
  } catch (const std::exception &e) {
    INFO << "Exception with stack trace " << std::endl;
    INFO << e.what();
  }
}
