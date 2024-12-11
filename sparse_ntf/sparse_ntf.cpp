#include <iostream>
#include "common/utils.h"
#include "common/ncpfactors.hpp"
#include "common/ntf_utils.hpp"
#include "common/parsecommandline.hpp"
#include "common/sparse_tensor.hpp"
#include "common/alto_tensor.hpp"

#if USE_BLCO==1
#include "common/blco_tensor.hpp"
#include "sparse_ntf/ntfmu_gpu.hpp"
#include "sparse_ntf/ntfhals_gpu.hpp"
#include "sparse_ntf/ntfaoadmm_gpu.hpp"
#include "sparse_ntf/ntfanlsbpp_gpu.hpp"
#endif

#include "ntf/ntfanlsbpp.hpp"
#include "ntf/ntfaoadmm.hpp"
#include "ntf/ntfhals.hpp"
#include "ntf/ntfmu.hpp"
#include "ntf/tfucp.hpp"
#include "ntf/ntfnes.hpp"

#if ALTO_MASK_LENGTH == 64
  typedef unsigned long long LIType;
#elif ALTO_MASK_LENGTH == 128
  typedef unsigned __int128 LIType;
#else
  #pragma message("Using default 128-bit.")
  typedef unsigned __int128 LIType;
  // typedef unsigned long long LIType;

#endif


namespace planc {

class SparseNTFDriver {
  public:
    // template <class NTFType>
    template <template<class T> class NTFType, class T>
    void callNTF(planc::ParseCommandLine pc) {
      std::string filename = pc.input_file_name();
      if (filename.empty()) {
        std::cout << "Input filename required for SparseNTF operations..." << std::endl;
        exit(1);
      }
      double wtime = omp_get_wtime();
      T my_tensor(filename);
      printf("[PERF-tensor load]\t%f\n", omp_get_wtime()-wtime);
      my_tensor.print();
      
      wtime = omp_get_wtime();
      NTFType<T> ntfsolver(my_tensor, pc.lowrankk(), pc.lucalgo());
      printf("[PERF-fm init]\t%f\n", omp_get_wtime()-wtime);
      
      // Setting flags. does it need to be here?
      ntfsolver.num_it(pc.iterations());
      ntfsolver.compute_error(pc.compute_error());
      ntfsolver.computeSparseNTF();
    }
    SparseNTFDriver() {}
};
}

int main(int argc, char* argv[]) {
  planc::ParseCommandLine pc(argc, argv);
  pc.parseplancopts();
  planc::SparseNTFDriver sntfd;

#if USE_BLCO==1
  //======== hard coded
  cudaError_t cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaSetDevice failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
    return 1;
  }
  //==========
#endif

  switch (pc.lucalgo())
  {
    case MU:
      switch (pc.gpu_type())
      {
      case CPU_ONLY:
        INFO << "\n\n ===== Running CPU only =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFMU, planc::ALTOTensor<LIType>>(pc);
        break;
#if USE_BLCO==1
      case PARTIAL:
        INFO << "\n\n ===== Running partial GPU =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFMU, planc::BLCOTensor<LIType>>(pc);
        break;
      case FULL:
        INFO << "\n\n ===== Running full GPU =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFMU_GPU, planc::BLCOTensor<LIType>>(pc);
        break;
#endif
      }
      break;
    case HALS:
      switch (pc.gpu_type())
      {
      case CPU_ONLY:
        INFO << "\n\n ===== Running CPU only =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFHALS, planc::ALTOTensor<LIType>>(pc);
        break;
#if USE_BLCO==1
      case PARTIAL:
        INFO << "\n\n ===== Running partial GPU =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFHALS, planc::BLCOTensor<LIType>>(pc);
        break;
      case FULL:
        INFO << "\n\n ===== Running full GPU =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFHALS_GPU, planc::BLCOTensor<LIType>>(pc);
        break;
#endif
      }
      break;
    case ANLSBPP:
      switch (pc.gpu_type())
      {
      case CPU_ONLY:
        INFO << "\n\n ===== Running CPU only =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFANLSBPP, planc::ALTOTensor<LIType>>(pc);
        break;
#if USE_BLCO==1
      case PARTIAL:
        INFO << "\n\n ===== Running partial GPU =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFANLSBPP, planc::BLCOTensor<LIType>>(pc);
        break;
      case FULL:
        INFO << "\n\n ===== Running full GPU =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFANLSBPP_GPU, planc::BLCOTensor<LIType>>(pc);
        break;
#endif
      }
      break;
    case AOADMM:
      switch (pc.gpu_type())
      {
      case CPU_ONLY:
        INFO << "\n\n ===== Running CPU only =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFAOADMM, planc::ALTOTensor<LIType>>(pc);
        break;
#if USE_BLCO==1
      case PARTIAL:
        INFO << "\n\n ===== Running partial GPU =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFAOADMM, planc::BLCOTensor<LIType>>(pc);
        break;
      case FULL:
        INFO << "\n\n ===== Running full GPU =====" << "\n\n\n";
        sntfd.callNTF<planc::NTFAOADMM_GPU, planc::BLCOTensor<LIType>>(pc);
        break;
#endif
      }
      break;    // Leave out NESTEROV for now since it requires a bit of refactoring --
    // it computes the objective error using the lowranktensor which
    // we don't explicitly use at all in the Sparse TF case...
    // case NESTEROV:
    //   sntfd.callNTF<planc::NTFNES, planc::ALTOTensor<LIType>>(pc);
    //   break;
    default:
      ERR << "Wrong algorithm choice. Quitting.." << pc.lucalgo() << std::endl;
  }
}