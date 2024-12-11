/* Copyright Ramakrishnan Kannan 2017 */

#include <iostream>
#include "common/utils.h"
#include "common/ncpfactors.hpp"
#include "common/ntf_utils.hpp"
#include "common/parsecommandline.hpp"
#include "common/tensor.hpp"
#include "ntf/ntfanlsbpp.hpp"
#include "ntf/ntfaoadmm.hpp"
#include "ntf/ntfhals.hpp"
#include "ntf/ntfmu.hpp"
#include "ntf/ntfnes.hpp"

// ntf -d "2 3 4 5" -k 5 -t 20

namespace planc {

class NTFDriver {
 public:
  template <template<class T> class NTFType, class T>
  void callNTF(planc::ParseCommandLine pc) {
    int test_modes = pc.num_modes();
    UVEC dimensions(test_modes);
    T my_tensor(pc.dimensions());
    std::string rand_prefix("rand_");
    std::string filename = pc.input_file_name();
    std::cout << "Input filename = " << filename << std::endl;
    if (!filename.empty() &&
        filename.compare(0, rand_prefix.size(), rand_prefix) != 0) {
      my_tensor.read(pc.input_file_name());
      my_tensor.print();
    }
    NTFType<T> ntfsolver(my_tensor, pc.lowrankk(), pc.lucalgo());
    ntfsolver.num_it(pc.iterations());
    ntfsolver.compute_error(pc.compute_error());
    if (pc.dim_tree()) {
      ntfsolver.dim_tree(true);
    }
    ntfsolver.computeNTF();
    // ntfsolver.ncp_factors().print();
  }
  NTFDriver() {}
};  // class NTF Driver

}  // namespace planc

int main(int argc, char* argv[]) {
  planc::ParseCommandLine pc(argc, argv);
  pc.parseplancopts();
  planc::NTFDriver ntfd;
  switch (pc.lucalgo()) {
    case MU:
      ntfd.callNTF<planc::NTFMU, planc::Tensor>(pc);
      break;
    case HALS:
      ntfd.callNTF<planc::NTFHALS, planc::Tensor>(pc);
      break;
    case ANLSBPP:
      ntfd.callNTF<planc::NTFANLSBPP, planc::Tensor>(pc);
      break;
    case AOADMM:
      ntfd.callNTF<planc::NTFAOADMM, planc::Tensor>(pc);
      break;
    case NESTEROV:
      ntfd.callNTF<planc::NTFNES, planc::Tensor>(pc);
      break;
    default:
      ERR << "Wrong algorithm choice. Quitting.." << pc.lucalgo() << std::endl;
  }
}
