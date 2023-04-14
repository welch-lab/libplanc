/* Copyright 2017 Ramakrishnan Kannan */

#include <armadillo>
#include "ncpfactors.hpp"
#include "ntf_utils.hpp"
#include "tensor.hpp"
#include "utils.h"

int main(int argc, char* argv[]) {
  int test_order = 5;
  int low_rank = 2;
  arma::uvec dimensions(test_order);
  arma::fmat* mttkrps = new arma::fmat[test_order];
  // arma::uvec dimensions(4);
  for (int i = 0; i < test_order; i++) {
    dimensions(i) = i + 2;
  }
  // dimensions(3) = 6;
  PLANC::NCPFactors cpfactors(dimensions, low_rank);
  cpfactors.print();
  arma::fmat krp_2 = cpfactors.krp_leave_out_one(2);
  cout << "krp" << endl << "------" << endl << krp_2;
  PLANC::Tensor my_tensor = cpfactors.rankk_tensor();
  // cout << "input tensor" << endl << "--------" << endl;
  // my_tensor.print();
  for (int i = 0; i < test_order; i++) {
    mttkrps[i] = arma::zeros<arma::fmat>(dimensions(i), low_rank);
  }
  for (int i = 0; i < test_order; i++) {
    mttkrp(i, my_tensor, cpfactors, &mttkrps[i]);
    cout << "mttkrp " << i << endl << "---------" << endl << mttkrps[i];
  }
}
