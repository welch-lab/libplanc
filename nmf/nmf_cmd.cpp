#include "NMFDriver.hpp"
#include "SparseNMFDriver.hpp"
#include "parsecommandline.hpp"
#include <stdio.h>
#include <string>
#include "utils.hpp"

int main(int argc, char *argv[]) {
  try {
    planc::ParseCommandLine dnd(argc, argv);
    planc::NMFDriver myNMF(dnd);
    myNMF.callNMF();
    fflush(stdout);
  } catch (const std::exception &e) {
    INFO << "Exception with stack trace " << std::endl;
    INFO << e.what();
  }
}
