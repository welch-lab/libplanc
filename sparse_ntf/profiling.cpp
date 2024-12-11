#include <iostream>
#include "common/utils.h"
#include "common/ncpfactors.hpp"
#include "common/ntf_utils.hpp"
#include "common/parsecommandline.hpp"
#include "common/sparse_tensor.hpp"
#include "common/alto_tensor.hpp"
#include "common/blco_tensor.hpp"

#if ALTO_MASK_LENGTH == 64
  typedef unsigned long long LIType;
#elif ALTO_MASK_LENGTH == 128
  typedef unsigned __int128 LIType;
#else
  #pragma message("Using default 128-bit.")
  typedef unsigned __int128 LIType;
  // typedef unsigned long long LIType;

#endif

#define RANK 16

int main(int argc, char* argv[]) {
  if (argc < 2) {
    INFO << "Usage: " << argv[0] << " <full_path_to_input_tensor>" << std::endl;
    return 1;
  }

  INFO << "Profiling sparse mttkrp implementations..." << std::endl;
  std::string filename = argv[1];
  // std::string filename = "/home/users/ypsoh/hpctensor/synthetic/large_sparse_tensor.tns";oo
  // std::string filename = "/home/users/ypsoh/hpctensor/data/flickr-4d.tns";
  // std::string filename = "/home/users/ypsoh/hpctensor/data/uber.tns";
  int num_iters = 1;
  
  {
    double wtime = omp_get_wtime();
    planc::SparseTensor t(filename);
    printf("tensor reading took %f s\n", omp_get_wtime()-wtime);
    planc::NCPFactors fms(t.dimensions(), RANK, false);
    
    fms.normalize();
    
    MAT* o_mttkrp = new MAT[t.modes()];
    for (int m = 0; m < t.modes(); ++m) {
      o_mttkrp[m].zeros(RANK, t.dimensions(m));
    }
    
    int num_iters = 1;
    for (int n = 0; n < num_iters; ++n) {
      for (int m = 0; m < t.modes(); ++m) {
        wtime = omp_get_wtime();
        t.mttkrp(m, fms.factors(), &o_mttkrp[m]);
        printf("mode: %d\t time: %f\n", m, omp_get_wtime()-wtime);
      }
    }
  }
  
  
  {
    planc::ALTOTensor<LIType> at(filename);
    planc::NCPFactors fms(at.dimensions(), RANK, false);
    
    fms.normalize();
    
    double wtime;

    MAT* o_mttkrp = new MAT[at.modes()];
    for (int m = 0; m < at.modes(); ++m) {
      o_mttkrp[m].zeros(RANK, at.dimensions(m));
    }
    
    for (int n = 0; n < 1; ++n) {
      for (int m = 0; m < at.modes(); ++m) {
        wtime = omp_get_wtime();
        at.mttkrp(m, fms.factors(), &o_mttkrp[m]);
        printf("[ALTO] mode: %d\t time: %f\n", m, omp_get_wtime()-wtime);
        printf("[ALTO] norm of output -- mode: %d: %f\n", m, arma::norm(o_mttkrp[m], "fro"));
      }
    }
  }
  
  
  {
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    planc::BLCOTensor<LIType> bt(filename);
    planc::NCPFactors fms(bt.dimensions(), RANK, false);
    
    fms.normalize();
    
    double wtime_s, wtime;

    MAT* o_mttkrp = new MAT[bt.modes()];
    for (int m = 0; m < bt.modes(); ++m) {
      o_mttkrp[m].zeros(RANK, bt.dimensions(m));
    }
    
    for (int n = 0; n < num_iters; ++n) {
      for (int m = 0; m < bt.modes(); ++m) {
        wtime = omp_get_wtime();
        bt.mttkrp(m, fms.factors(), &o_mttkrp[m]);
        printf("[BLCO] mode: %d\t time: %f\n", m, omp_get_wtime()-wtime);
        printf("[BLCO] norm of output -- mode: %d: %f\n", m, arma::norm(o_mttkrp[m], "fro"));
      }
    }
  }  
}