#ifndef COMMON_ALTO_TENSOR_HPP_
#define COMMON_ALTO_TENSOR_HPP_

#ifdef MKL_FOUND
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <armadillo>
#include <fstream>
#include <ios>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include "common/utils.h"
#include "common/ncpfactors.hpp"
#include "common/bitops.hpp"

#define MAX_NUM_MODES 5
#define MIN_FIBER_REUSE 4

typedef enum PackOrder_ { LSB_FIRST, MSB_FIRST } PackOrder;
typedef enum ModeOrder_ { SHORT_FIRST, LONG_FIRST, NATURAL } ModeOrder;

typedef struct Inverval_ {
  int start;
  int stop;
} Interval;

struct MPair {
  int mode;
  int bits;
};

namespace planc {

template <typename LIT>
class ALTOTensor : public SparseTensor {
  public:
    ALTOTensor() = default;
    explicit ALTOTensor(const std::string& filename);
    ~ALTOTensor() = default;

    void mttkrp(int target_mode, MAT* i_factors, MAT* o_mttkrp) const override;

  private:
    void setupPackedALTO(PackOrder pack_order, ModeOrder mode_order);
    void sortALTO();
    void partitionWorkload();
    void createDirectAccessMemory(int rank) const;

  protected:
    int num_partitions_ = 0;
    LIT alto_mask_ = 0;
    std::vector<LIT> mode_masks_;
    std::vector<double> alto_data_;
    std::vector<LIT> alto_indices_;
    std::vector<int> partition_ptr_;
    std::vector<std::vector<Interval>> partition_intervals_;
    mutable bool is_optimized_for_mttkrp_ = false;
    mutable std::vector<std::vector<double>> output_fibers_;
    std::vector<LIT> alto_cr_masks_;
};

template <typename LIT>
ALTOTensor<LIT>::ALTOTensor(const std::string& filename) : SparseTensor(filename) {
    double wtime_s, wtime;

    INFO << "Generating ALTO tensor" << std::endl;
    mode_masks_.resize(m_modes, 0);
    UWORD numel = this->numel();
    alto_data_.resize(numel);
    alto_indices_.resize(numel);

    wtime_s = omp_get_wtime();
    setupPackedALTO(LSB_FIRST, SHORT_FIRST);
    wtime = omp_get_wtime() - wtime_s;
    printf("ALTO: setup time = %f (s)\n", wtime);

    // Linearization
    wtime_s = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < numel; i++) {
        LIT alto = 0;
        alto_data_[i] = m_data[i];
        for (int j = 0; j < m_modes; j++) {
            alto |= pdep(static_cast<LIT>(m_compact_indices[j][i]), mode_masks_[j]);
        }
        alto_indices_[i] = alto;
    }
    wtime = omp_get_wtime() - wtime_s;
    printf("ALTO: Linearization time = %f (s)\n", wtime);

    //Sort the nonzeros based on their line position.
    wtime_s = omp_get_wtime();
    sortALTO();
    wtime = omp_get_wtime() - wtime_s;
    printf("ALTO: sort time = %f (s)\n", wtime);

    wtime_s = omp_get_wtime();
    partitionWorkload();
    wtime = omp_get_wtime() - wtime_s;
    printf("ALTO: partition time = %f (s)\n", wtime);
}

template <typename LIT>
void ALTOTensor<LIT>::setupPackedALTO(PackOrder po, ModeOrder mo) {
  LIT ALTO_MASKS[MAX_NUM_MODES] = {};
  int alto_bits_min = 0;
  int alto_bits_max = 0;

  int max_num_bits = 0;
  int min_num_bits = sizeof(int) * 8;

  // std::vector<MPair> mode_bits(m_modes);
  MPair* mode_bits = (MPair*)malloc(m_modes * sizeof(MPair));

  // initial mode values
  for (int n = 0; n < m_modes; ++n) {
    // this is hard-coded to accomodate 64 bits representation
    int mbits = (sizeof(unsigned long long) * 8) - clz(m_dimensions(n)-1);
    mode_bits[n].mode = n;
    mode_bits[n].bits = mbits;
    alto_bits_min += mbits;
    max_num_bits = std::max(max_num_bits, mbits);
    min_num_bits = std::min(min_num_bits, mbits);
    printf("num_bits for mode-%d=%d\n", n + 1, mbits);
  }

  alto_bits_max = max_num_bits * m_modes;
  //printf("range of mode bits=[%d %d]\n", min_num_bits, max_num_bits);
  printf("alto_bits_min=%d, alto_bits_max=%d\n", alto_bits_min, alto_bits_max);

  int alto_bits = (int)0x1 << std::max<int>(3, (sizeof(int) * 8) - __builtin_clz(alto_bits_min));
  printf("alto_bits=%d\n", alto_bits);

  double alto_storage = 0;
  alto_storage = m_numel * (sizeof(double) + sizeof(LIT));
  printf("Alto format storage:    %g Bytes\n", alto_storage);

  alto_storage = m_numel * (sizeof(double) + (alto_bits >> 3));
  printf("Alto-power-2 format storage:    %g Bytes\n", alto_storage);

  alto_storage = m_numel * (sizeof(double) + (alto_bits_min >> 3));
  printf("Alto-opt format storage:    %g Bytes\n", alto_storage);

  {//Dilation & shifting.
    int level = 0, shift = 0, inc = 1;

    if (mo == SHORT_FIRST)
      std::sort(mode_bits, mode_bits + m_modes, [&](MPair a, MPair b) { return a.bits < b.bits; });
    else if(mo == LONG_FIRST)
      std::sort(mode_bits, mode_bits + m_modes, [&](MPair a, MPair b) { return a.bits > b.bits; });
    if (po == MSB_FIRST) {
      shift = alto_bits_min - 1;
      inc = -1;
    }

    bool done;
    do {
      done = true;

      for (int n = 0; n < m_modes; ++n) {
        if (level < mode_bits[n].bits) {
          ALTO_MASKS[mode_bits[n].mode] |= (LIT)0x1 << shift;
          shift += inc;
          done = false;
        }
      }
      ++level;
    } while (!done);

    assert(level == (max_num_bits+1));
    assert(po == MSB_FIRST ? (shift == -1) : (shift == alto_bits_min));
  }

  for (int n = 0; n < m_modes; ++n) {
    mode_masks_[n] = ALTO_MASKS[n];
    alto_mask_ |= ALTO_MASKS[n];
    printf("ALTO_MASKS[%d] = 0x%llx\n", n, ALTO_MASKS[n]);
  }
  if (alto_bits > 64) { // 128 bit mask
    printf("alto_mask = 0x%llx%llx\n", (unsigned long long)(alto_mask_ >> 64), (unsigned long long)alto_mask_);
  } else {
    printf("alto_mask = 0x%llx\n", alto_mask_);
  }
  free(mode_bits);
}

template <typename LIT>
void ALTOTensor<LIT>::sortALTO() {
    UWORD numel = this->numel();
    std::vector<int> inds(numel);
    std::iota(inds.begin(), inds.end(), 0);

    std::sort(inds.begin(), inds.end(), [this](int a, int b) {
        return alto_indices_[a] < alto_indices_[b];
    });

    std::vector<LIT> temp_inds(numel);
    std::vector<double> temp_val(numel);

    #pragma omp parallel for
    for (int i = 0; i < numel; i++) {
        temp_inds[i] = alto_indices_[inds[i]];
        temp_val[i] = alto_data_[inds[i]];
    }

    #pragma omp parallel for
    for (int i = 0; i < numel; i++) {
        alto_indices_[i] = temp_inds[i];
        alto_data_[i] = temp_val[i];
    }
}

template <typename LIT>
void ALTOTensor<LIT>::partitionWorkload() {
  num_partitions_ = omp_get_max_threads();
  partition_ptr_.resize(num_partitions_ + 1);
  partition_intervals_.resize(num_partitions_);

  for (int p = 0; p < num_partitions_; ++p) {
    partition_intervals_[p].resize(m_modes);
  }

  // needed for conflict resolution implementation
  alto_cr_masks_.resize(m_modes);

  // roughly how much nnzs per partition
  int nnz_partition = (m_numel + num_partitions_ - 1) / num_partitions_;
  printf("num_partitions=%d, nnz_per_partition=%llu\n", num_partitions_, nnz_partition);

  LIT ALTO_MASKS[MAX_NUM_MODES];
  for (int n = 0; n < m_modes; ++n) {
    ALTO_MASKS[n] = mode_masks_[n];
  }

  int alto_bits = popcount(alto_mask_);
  partition_ptr_[0] = 0;

  #pragma omp parallel for schedule(static,1) proc_bind(close)
  for (int p = 0; p < num_partitions_; ++p) {
    int start_i = p * nnz_partition;
    int end_i = (p + 1) * nnz_partition;

    if (end_i > m_numel)
        end_i = m_numel;

    if (start_i > end_i)
        start_i = end_i;

    // partition pointer points to the index where the nnz index ends for that partition
    // e.g. m_partition_ptr[1] indicates the index where 1st partition nnz ends
    partition_ptr_[p + 1] = end_i;
  }// omp parallel

  // O(storage requirements) for conflict resolution,
  // using dense/direct-access storage,
  // can be computed in constant time from the subspace id. 
  // The code below finds tighter bounds
  // using interval analysis in linear time (where nnz>> nptrn>> m_modes).
  // e.g. partition_intervals_[0] has the start and stop indices for 1st partition for all modes
  // partition_intervals_[0][1]: start and stop indices for the 1st partition for 2nd mode
  #pragma omp parallel for schedule(static,1) proc_bind(close)
  for (int p = 0; p < num_partitions_; ++p) {
      Interval fib[MAX_NUM_MODES];
      for (int n = 0; n < m_modes; ++n) {
        fib[n].start = m_dimensions(n);
        fib[n].stop = 0;
      }

      // for all elements in the p-th partition
      for (int i = partition_ptr_[p]; i < partition_ptr_[p + 1]; ++i) {
        LIT alto_idx = alto_indices_[i];
        for (int n = 0; n < m_modes; ++n) {
          // retrieve original index for mode n
          int mode_idx = pext(alto_idx, ALTO_MASKS[n]);
          // get the start and stop index for each mode
          // for each 
          fib[n].start = std::min(fib[n].start, mode_idx);
          fib[n].stop = std::max(fib[n].stop, mode_idx);
        }
      }

      for (int n = 0; n < m_modes; ++n) {
          partition_intervals_[p][n].start = fib[n].start;
          partition_intervals_[p][n].stop = fib[n].stop;
      }
  }
}

template <typename LIT>
void ALTOTensor<LIT>::createDirectAccessMemory(int rank) const {
  output_fibers_.resize(num_partitions_);
  
  {
    double total_storage = 0.0;
    #pragma omp parallel for reduction(+: total_storage) proc_bind(close)
    for (int p = 0; p < num_partitions_; p++) {
      int num_fibs = 0;
      // default mode - allocate enough da_mem for all modes with reuse > threshold
      for (int n = 0; n < m_modes; ++n) {
        float fib_reuse = static_cast<float>(m_numel) / dimensions(n);
        // printf("fib reuse: %f, (%d, %d)\n", fib_reuse, m_numel, dimensions(n));
        if (fib_reuse > MIN_FIBER_REUSE) {
          Interval const intvl = partition_intervals_[p][n];
          int const mode_fibs = intvl.stop - intvl.start + 1;
          num_fibs = std::max(num_fibs, mode_fibs);
        }
      }
      output_fibers_[p].resize(num_fibs * rank);
      total_storage += ((double) num_fibs * rank * sizeof(double)) / (1024.0*1024.0);
    } // for all partitions
    printf("ofibs storage/prtn: %f MB\n", total_storage/(double)num_partitions_);
  }
  
  is_optimized_for_mttkrp_ = true;
}

template <typename LIT>
void ALTOTensor<LIT>::mttkrp(const int target_mode, MAT* i_factors, MAT* o_mttkrp) const {
    o_mttkrp->zeros();
    unsigned int rank = i_factors[target_mode].n_cols;

    if (!is_optimized_for_mttkrp_) {
        createDirectAccessMemory(rank);
    }

    if (m_numel / (o_mttkrp->n_cols) <= MIN_FIBER_REUSE) {
        // Use atomic updates
        LIT ALTO_MASKS[MAX_NUM_MODES];
        for (int n = 0; n < m_modes; ++n) {
          ALTO_MASKS[n] = mode_masks_[n];
        }

        // Use atomic
        #pragma omp parallel for schedule(static,1) proc_bind(close)
        for (int p = 0; p < num_partitions_; ++p) {
          double row[rank];

          int const nnz_s = partition_ptr_[p];
          int const nnz_e = partition_ptr_[p + 1];

          for (int i = nnz_s; i < nnz_e; ++i) {
            double const val = alto_data_[i];
            LIT const alto_idx = alto_indices_[i];

            #pragma omp simd
            for (int r = 0; r < rank; ++r) {
              row[r] = val;
            }

            for (int m = 0; m < m_modes; ++m) {
              if (m != target_mode) { //input fibers
                int const row_id = pext(alto_idx, ALTO_MASKS[m]);
                #pragma omp simd
                for (int r = 0; r < rank; r++) {
                  row[r] *= i_factors[m](row_id, r);
                }
              }
            }

            //Output fibers
            int const row_id = pext(alto_idx, ALTO_MASKS[target_mode]);
            for (int r = 0; r < rank; ++r) {
              #pragma omp atomic update
              o_mttkrp->at(r, row_id) += row[r];
            }
          } //nnzs
        } //prtns
    } else {
        // Use pull-based method
        LIT ALTO_MASKS[MAX_NUM_MODES];
        for (int n = 0; n < m_modes; ++n) {
          ALTO_MASKS[n] = mode_masks_[n];
        }

        double row[rank];
        // Use pull based
        #pragma omp parallel proc_bind(close)
        {
          #pragma omp for schedule(static, 1)
          for (int p = 0; p < num_partitions_; ++p) {
            double row[rank];

            std::vector<double> &out = output_fibers_.at(p);
            
            Interval const intvl = partition_intervals_[p][target_mode];

            int const offset = intvl.start;
            int const stop = intvl.stop;

            std::fill(out.begin(), out.end(), 0.0);

            int const nnz_s = partition_ptr_[p];
            int const nnz_e = partition_ptr_[p + 1];

            for (int i = nnz_s; i < nnz_e; ++i) {
              double const val = alto_data_[i];
              LIT const alto_idx = alto_indices_[i];

              #pragma omp simd
              for (int r = 0; r < rank; ++r) {
                row[r] = val;
              }

              for (int m = 0; m < m_modes; ++m) {
                if (m != target_mode) { //input fibers
                  int const row_id = pext(alto_idx, ALTO_MASKS[m]);

                  #pragma omp simd 
                  for (int r = 0; r < rank; ++r) {
                    row[r] *= i_factors[m](row_id, r);
                  }
                }
              }

              //Output fibers
              int row_id = pext(alto_idx, ALTO_MASKS[target_mode]) - offset;

              row_id *= rank;
              #pragma omp simd
              for (int r = 0; r < rank; ++r) {
                out[row_id + r] += row[r];
              }
            } //nnzs
          } //prtns
          //pull-based accumulation
          #pragma omp for schedule(static)
          for (int i = 0; i < dimensions(target_mode); ++i) {
            for (int p = 0; p < num_partitions_; p++)
            {
              std::vector<double> &out = output_fibers_.at(p);
              Interval const intvl = partition_intervals_[p][target_mode];
              int const offset = intvl.start;
              int const stop = intvl.stop;

              if ((i >= offset) && (i <= stop)) {
                int const j = i - offset;
                #pragma omp simd
                for (int r = 0; r < rank; r++) {
                  o_mttkrp->at(r, i) += out[j * rank + r];
                }
              }
            } //prtns
          } //ofibs
        } // omp parallel
    }
}
} // namespace planc
#endif // COMMON_ALTO_TENSOR_HPP_