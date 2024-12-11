#ifndef COMMON_BLCO_TENSOR_HPP_
#define COMMON_BLCO_TENSOR_HPP_

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
#include <chrono>
#include "common/utils.h"
#include "common/ncpfactors.hpp"
#include "common/bitops.hpp"
#include "common/alto_tensor.hpp"
// GPU kernels
#include "blco.h"

#include "cassert"

#define _IType unsigned long long

namespace planc {  
  template <typename LIT>
  class BLCOTensor : public planc::ALTOTensor<LIT> {
    public:
      // for BLCO tensors
      unsigned long long * m_blco_indices;
      double* m_blco_values;

      int * m_blco_mode_pos;
      unsigned long long * m_blco_mode_mask;

      BLCOBlock ** m_blocks = nullptr;
      int m_num_blocks = 0;
      int m_max_block_size = 16777216; // currently at 2**24, paper says it supports up to 2**27

      // for GPU
      mutable BLCOTensorGPU* bt_gpu = nullptr;
      bool use_stream = false;

      BLCOTensor(std::string filename, bool use_stream = false) : ALTOTensor<LIT>(filename), use_stream(use_stream) {

        double wtime_s, wtime;

        // Pin memory on host
        this->m_blco_indices = (unsigned long long *)malloc(sizeof(unsigned long long) * this->m_numel);
        this->m_blco_values = (double *) malloc(sizeof(double) * this->m_numel);

        // BLCO mode_mask uses only 64bits
        this->m_blco_mode_pos = (int *)malloc(sizeof(int) * MAX_NUM_MODES);
        this->m_blco_mode_mask = (unsigned long long *)malloc(sizeof(unsigned long long) * MAX_NUM_MODES);
        
        wtime_s = omp_get_wtime();

        int truncated_bitcounts[MAX_NUM_MODES];
        // if alto mode mask is 0100101 for a certain mode
        // truncated_bitcounts stores the number of 1s, so 3 in this case
        // if you sum it all for all modes
        // it basically represents the total number of bits used to encode ALTO tensor
        for (int m = 0; m < this->m_modes; ++m) {
          unsigned long long mask = lhalf(this->mode_masks_[m]);
          truncated_bitcounts[m] = popcount(mask);
        }
        
        // construct mask and pos for BLCO
        // untangling the bit masks from alto and using consecutive bit masks for each mode
        for (int m = 0; m < this->m_modes; ++m) {
          this->m_blco_mode_mask[m] = ((unsigned long long) 1 << truncated_bitcounts[m]) - 1;
          this->m_blco_mode_pos[m] = (m == 0) ? 0 : this->m_blco_mode_pos[m-1] + truncated_bitcounts[m-1];
        }

        for (int n = 0; n < this->m_modes; ++n) {
          printf("BLCO_MASKS[%d] = 0x%llx\n", n, this->m_blco_mode_mask[n]);
        }

        // Relinearize and copy into BLCO
        #pragma omp parallel for schedule(static)
        for (unsigned long long i = 0; i < this->m_numel; i++) {
          LIT index = this->alto_indices_[i];
          unsigned long long blco_idx = 0;
          for (int n = 0; n < this->m_modes; ++n) {
            // recover original index, re encode to linearized format
            unsigned long long mode_idx = (unsigned long long) pext(index, this->mode_masks_[n]) & this->m_blco_mode_mask[n];
            blco_idx |= (mode_idx << this->m_blco_mode_pos[n]);
          }
          this->m_blco_indices[i] = blco_idx;
          this->m_blco_values[i] = this->alto_data_[i];
        }

        wtime = omp_get_wtime() - wtime_s;
        printf("BLCO: Relinearize time = %f (s)\n", wtime);

        wtime_s = omp_get_wtime();
        // determine possible number of blocks
        int block_count = 0;
        for (int i = 0; i < this->m_modes; i++) {
          // each dimension -- count the leading zeros
          // the smaller the dimension, the larger the clz count would be
          // the larger the dimension, the smaller the clz would be
          // substract it from 64, the smaller dimensions would have smaller block_count
          // the larger dimension would have larger block count. If dimension has zero leading zero
          // then block count would be 64
          block_count += (sizeof(unsigned long long) * 8) - clz(this->m_dimensions[i] - 1);
        }
        // XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX // 128 bit
        // XXXX XXXX XXXX XXXX OOOO OOOO OOOO OOOO // front 64 bits are used for BLCO
        // need to count how much bits of "X" we need to use for block indexing
        // for all modes, 64 - clz(dim_size), basically computes how many Os we need to represent
        // a certain mode, after the sum for all modes, substract it from 64, 
        // which basically tells us how many X bits we need for block indexing
        block_count = block_count - sizeof(unsigned long long) * 8; // possible negative at this point
        block_count = std::max(0, block_count);
        // it can be 1 or 2^block_count
        block_count = 1 << block_count;

        // block_histogram counts the number of nonzeros each block has
        // note that each block uses its key (uhalf(alto_index) as its index.
        unsigned long long* block_histogram = new unsigned long long[block_count];
        unsigned long long* block_prefix_sum = new unsigned long long[block_count + 1];
        #pragma omp parallel for
        for (unsigned long long i = 0; i < block_count; i++) block_histogram[i] = 0;

        wtime = omp_get_wtime() - wtime_s;
        printf("BLCO: Setup time = %f (s), BLOCK COUNT: %d\n", wtime, block_count);

        // Construct block histogram. OpenMP 4.5+ required
        wtime_s = omp_get_wtime();
        if (block_count > 1) {
          #pragma omp parallel for reduction(+:block_histogram[:block_count])
          for (unsigned long long i = 0; i < this->m_numel; i++) {
            // Get upper half of linearized index and use that as block_id
            unsigned long long block_id = uhalf(this->alto_indices_[i]); 
            block_histogram[block_id] += 1;
          }
        } else {
          // one block has all non-zeros
          block_histogram[0] = this->m_numel;
        }
        wtime = omp_get_wtime() - wtime_s;
        printf("BLCO: Histogram time = %f (s)\n", wtime);
        
        // we may have to split the blocks further if it contains too many nnzs
        wtime_s = omp_get_wtime();
        unsigned long long total_blocks_split = 0;
        block_prefix_sum[0] = 0;
        if (this->m_max_block_size <= 0) this->m_max_block_size = this->m_numel;
        for (unsigned long long i = 0; i < block_count; i++) {
          // block_histogram[i] has the nnz for block index i
          // total_blocks_split tracks how many blocks we need to split
          // if there is too many nnz in a single block i, 
          // or in some cases some blocks might have 0 nnzs
          total_blocks_split += (block_histogram[i] + this->m_max_block_size - 1) / this->m_max_block_size;
          // block_prefix_sum accumulates the nnz for each blocks
          block_prefix_sum[i + 1] = block_prefix_sum[i] + block_histogram[i];
        }

        // Construct split blocks
        // "block_count" is the number of blocks purely based on exceeding bits > 64
        // "total_blocks_split" is "block_count" + a where a is determined based on how many
        // blocks didn't "fit" based on criteria, if no block exceeds limit, block_count == total_blocks_split
        wtime_s = omp_get_wtime();
        // BLCOBlock** blocks = new BLCOBlock*[total_blocks_split];

        this->m_blocks = new BLCOBlock*[total_blocks_split];
        this->m_num_blocks = total_blocks_split;
        // this->block_count = total_blocks_split;

        int curr_blocks = 0;
        for (int block = 0; block < block_count; block++) {
          int start = block_prefix_sum[block];
          int end = block_prefix_sum[block + 1];
          int nnz = end - start; // the nnz each block has
          if (nnz > 0) {
            // Generate block coordinates, tricky because we can only use "block"
            unsigned long long block_coords[MAX_NUM_MODES];
            for (int i = 0; i < this->m_modes; i++) {
              // YONGSEOK: note this->mode_masks is from ALTOTensor,, 
              // block_coords is required to reverse back to the original ALTO idnex
              unsigned long long mode_mask = uhalf(this->mode_masks_[i]);
              // from ALTO perspective, mode_idx will contain the bit mask information 
              // if block index is 101 
              // and mode_mask for uhalf for a mode is 011, 
              // alto needs to know that 001 (101 & 011 -> 001) to get original coordinate for that mode
              unsigned long long mode_idx = pext(block, mode_mask);
              mode_idx <<= truncated_bitcounts[i]; // shift it back up to block idx space
              block_coords[i] = mode_idx;
            }

            // Generate split block, indices are just offset pointers into main array
            for (int stride = 0; stride < nnz; stride += this->m_max_block_size) {
              int nnz_split_block = std::min(this->m_max_block_size, nnz - stride);
              BLCOBlock* blk = generate_blco_block_host(this->m_modes, nnz_split_block);
              blk->idx = this->m_blco_indices + start + stride;
              blk->vals = this->m_blco_values + start + stride;
              this->m_blocks[curr_blocks] = blk;
              curr_blocks++;
              for (int i = 0; i < this->m_modes; i++) blk->block_coords[i] = block_coords[i];
            }
          }
        }

        wtime = omp_get_wtime() - wtime_s;
        printf("BLCO: Blocking time = %f (s)\n", wtime);
        printf("Num. blocks: %d\tTotal blocks: %d\n\n", block_count, total_blocks_split);

        // Generate / Allocate BLCOtensor on GPU
        this->bt_gpu = generate_blco_tensor_gpu(
          m_max_block_size,
          this->m_modes,
          (unsigned long long *)this->m_dimensions.memptr(),
          this->m_numel,
          this->m_blco_mode_mask,
          this->m_blco_mode_pos,
          this->m_num_blocks,
          this->m_blocks,
          use_stream // if using stream will not alloc cudaMemory for blocks in constructor
        );

        if (!use_stream) {
          for (int b = 0; b < this->m_num_blocks; ++b) {
            // copying block coords, idx, vals to cudaMem
            send_blco_block_gpu(this->m_blocks[b], this->bt_gpu->m_blocks[b]);
          }
        }
        // for (int block_idx[] = 0; block_idx < total_blocks_split; ++block_idx) {
        // print block stuff
        //   BLCOBlock * b = this->m_blocks[block_idx];
        //   printf("bidxDimensions: %d, nnz: %d\n", block_idx, b->m_numel);
        //   for (int m = 0; m < b->m_modes; ++m) {
        //     printf("block coord for mode: %d\t 0x%llx\n", m, b->block_coords[m]);
        //   }
        // }
        // free ALTO related stuff
        //  -- not ideal and kind of hacky since we're using std::vectors
        this->alto_data_.clear(); // doesn't guarantee
        this->alto_data_.shrink_to_fit(); // C++11
        this->alto_indices_.clear();
        this->alto_indices_.shrink_to_fit();
        this->partition_intervals_.clear();
        this->partition_intervals_.shrink_to_fit();
        this->partition_ptr_.clear();
        this->partition_ptr_.shrink_to_fit();
      }

      ~BLCOTensor() {
        // Clean up BLCOTensor
        free(this->m_blco_indices);
        free(this->m_blco_values);
        free(this->m_blco_mode_pos);
        free(this->m_blco_mode_mask);
      }

      // mttkrp_gpu is a mttkrp kernel that assumes everything is in GPU
      void mttkrp_gpu(const int target_mode, MAT_GPU ** i_factors_gpu, MAT_GPU *o_mttkrp_gpu) const {
        // Do mttkrp_gpu for BLCO Tensor
        // this specific kernel assumes blcotensor is sent to gpu
        int target_mode_dim = i_factors_gpu[target_mode]->n_rows;
        int rank = i_factors_gpu[target_mode]->n_cols;

        // o_mttkrp_gpu has size of longest mode of all factor matrices
        // only zero out based on current mode
        check_cuda(cudaMemset(
          o_mttkrp_gpu->vals, 0, 
          sizeof(double) * target_mode_dim * rank), "memset o_mttkrp_gpu to zero");

        if (!use_stream) {
          for (_IType b = 0; b < bt_gpu->m_num_blocks; ++b) {
            mttkrp_lvl1(
              bt_gpu->m_blocks[b],
              o_mttkrp_gpu, // MAT_GPU*
              i_factors_gpu, // MAT_GPU**
              target_mode, // int
              rank,
              bt_gpu->dims,
              bt_gpu->m_streams[0] // use single stream
            );
          }
          // No need to free m_blocks since it will be reused
        }
        else { // streaming blocks
          // alloc memory required for streaming
          for (_IType b = 0; b < bt_gpu->m_num_blocks; ++b) {
            bt_gpu->m_blocks[b] = generate_blco_block_gpu(this->m_blocks[b]);
            check_cuda(cudaStreamCreate(&bt_gpu->m_streams[b]), "cudaStreamCreate");
          }

          _IType thread_coal_factor = 4;
          _IType stream_id = bt_gpu->m_num_blocks - 1;
          for (_IType b = 0; b < bt_gpu->m_num_blocks; ++b) {
            stream_id = (stream_id + 1) % bt_gpu->m_num_blocks;
            cudaStream_t stream = bt_gpu->m_streams[stream_id];

            send_blco_block_gpu_async(m_blocks[b], bt_gpu->m_blocks[b], stream);

            mttkrp_lvl2(
              bt_gpu->m_blocks[b],
              o_mttkrp_gpu, // MAT_GPU*
              i_factors_gpu, // MAT_GPU**
              target_mode, // int
              rank,
              target_mode_dim,
              thread_coal_factor,
              stream
            );
            // after stream sync should release m_blocks once we're done here??
          }
        }
        check_cuda(cudaDeviceSynchronize(), "mttkrp complete");
      }

      // mttkrp kernel, unlike mttkrp_gpu kernel
      // assumes only the mttkrp operation being offloaded to the GPU
      // and does all the memcpy operations before and after
      // whereas mttkrp_gpu kernel assumes everything is offloaded to GPU
      // and does not do so
      void mttkrp(const int target_mode, MAT *i_factors, MAT *o_mttkrp) const {
        // pin o_mttkrp memory on cpu
        size_t o_mttkrp_size = o_mttkrp->n_cols * o_mttkrp->n_rows * sizeof(double);
        check_cuda(cudaHostRegister(o_mttkrp->memptr(), o_mttkrp_size, cudaHostRegisterDefault),
         "pin o_mttkrp memory on host");
        MAT_GPU * o_mttkrp_gpu = send_mat_to_gpu(o_mttkrp);
        MAT_GPU ** i_factors_gpu = send_mats_to_gpu(i_factors, this->m_modes);

        set_mat_to_zero(o_mttkrp_gpu);
        
        int target_mode_dim = o_mttkrp->n_cols;
        int rank = o_mttkrp->n_rows;

        if (!use_stream) {
          for (_IType b = 0; b < bt_gpu->m_num_blocks; ++b) {
            mttkrp_lvl1(
              bt_gpu->m_blocks[b],
              o_mttkrp_gpu, // MAT_GPU*
              i_factors_gpu, // MAT_GPU**
              target_mode, // int
              rank,
              bt_gpu->dims,
              bt_gpu->m_streams[0] // use single stream
            );
          }
          // No need to free m_blocks since it will be reused
        }
        else { // streaming blocks
          // alloc memory required for streaming
          for (_IType b = 0; b < bt_gpu->m_num_blocks; ++b) {
            bt_gpu->m_blocks[b] = generate_blco_block_gpu(this->m_blocks[b]);
            check_cuda(cudaStreamCreate(&bt_gpu->m_streams[b]), "cudaStreamCreate");
          }

          _IType thread_coal_factor = 4;
          _IType stream_id = bt_gpu->m_num_blocks - 1;
          for (_IType b = 0; b < bt_gpu->m_num_blocks; ++b) {
            stream_id = (stream_id + 1) % bt_gpu->m_num_blocks;
            cudaStream_t stream = bt_gpu->m_streams[stream_id];

            send_blco_block_gpu_async(m_blocks[b], bt_gpu->m_blocks[b], stream);

            mttkrp_lvl2(
              bt_gpu->m_blocks[b],
              o_mttkrp_gpu, // MAT_GPU*
              i_factors_gpu, // MAT_GPU**
              target_mode, // int
              rank,
              target_mode_dim,
              thread_coal_factor,
              stream
            );
            // after stream sync should release m_blocks once we're done here??
          }
  
          check_cuda(cudaDeviceSynchronize(), "copy mttkrp result back to host");

          // free blocks since assumption for streaming is that it doesn't entirely fit in GPU
          for (_IType b = 0; b < bt_gpu->m_num_blocks; ++b) {
            cudaFree(bt_gpu->m_blocks[b]->vals);
            cudaFree(bt_gpu->m_blocks[b]->idx);
            cudaFree(bt_gpu->m_blocks[b]->block_coords);
          }
        }
        
        // Copy o_mttkrp_gpu back to host
        send_mat_to_host(o_mttkrp_gpu, o_mttkrp);
        check_cuda(cudaDeviceSynchronize(), "copy mttkrp result back to host");
        
        // Clean up gpu stuff
        // Unregister host memory
        cudaHostUnregister(o_mttkrp->memptr());
        cudaFree(o_mttkrp_gpu->vals);
        for (int m = 0; m < this->m_modes; ++m) cudaFree(i_factors_gpu[m]->vals);
      }
  };
}

#endif