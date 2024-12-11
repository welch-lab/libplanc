#ifndef COMMON_BLCO_HPP_
#define COMMON_BLCO_HPP_

#include <iostream>
#include "cuda_utils.h"
#include <cstdint>
#define ARMA_ALLOW_FAKE_GCC // to use arma::mat
#include <armadillo>
// this imports causes following warning
// "WARNING: this compiler is pretending to be GCC but it may not be fully compatible;"
#include "common/utils.h"
// #include <cooperative_groups.h>
// #include <chrono>

struct MAT_GPU; // declared in cuda_utils.h
struct UMAT_GPU; 

struct BLCOBlock {
  int m_modes;
  _IType m_numel;

  double * vals; // offset pointers into main memory 
  _IType * idx;  // offset pointers into main memory

  // Length `nmodes` array, the coordinates of this block in the BLCO tensor
  // needed to retreive original alto idx and to eventually original index
  unsigned long long * block_coords = nullptr;

  // double** pmatrices_staging_ptr = nullptr;
  // double** pmatrices = nullptr;
};

// almost identical as BLCOTensor but only contains essential stuff for gpu execution
struct BLCOTensorGPU {
  int m_modes;
  BLCOBlock ** m_blocks;
  unsigned long long * dims;

  _IType m_numel;
  int m_num_blocks = 0;
  int m_max_block_size = 16777216;

  int * m_blco_mode_pos;
  unsigned long long * m_blco_mode_mask;

  cudaStream_t * m_streams;
  _IType warp_info_length = 0;
  _IType * warp_info = nullptr;
  _IType * warp_info_gpu = nullptr;
};

BLCOTensorGPU * generate_blco_tensor_gpu(
  _IType max_block_size,
  int num_modes,
  unsigned long long * dimensions,
  _IType numels,
  _IType * mode_mask,
  int * mode_pos,
  _IType block_count,
  BLCOBlock ** blocks, bool use_stream);

BLCOBlock * generate_blco_block_host(int nmodes, _IType nnz);
BLCOBlock * generate_blco_block_gpu(BLCOBlock * block);

void send_blco_block_gpu(BLCOBlock * blk_host, BLCOBlock * blk_dev);
void send_blco_block_gpu_async(BLCOBlock * blk_host, BLCOBlock * blk_dev, cudaStream_t stream);

void send_blco_to_gpu();

void send_mat_to_host(MAT_GPU * o_mat_gpu, MAT * o_mat_host);
void send_umat_to_host(UMAT_GPU * o_mat_gpu, arma::Mat<unsigned int> * o_mat_host); // for dev purposes (BPP algo)

MAT_GPU * send_mat_to_gpu(MAT * mat);
MAT_GPU ** send_mats_to_gpu(MAT * mats, int num_modes);
void free_mats_on_gpu(MAT_GPU ** mats_gpu, int num_modes);
void free_mat_on_gpu(MAT_GPU * mat_gpu);
void set_mat_to_zero(MAT_GPU * mat);
__global__ void initializeMatrix(double* vals, int size);

void mttkrp_lvl1(BLCOBlock * blco_block, MAT_GPU * o_mttkrp_gpu, MAT_GPU ** i_factors_gpu, int target_mode, int rank, _IType* dimensions, cudaStream_t stream);
__global__ void mttkrp_lvl1_3d_kernel();
__global__ void mttkrp_lvl1_4d_kernel(const _IType* __restrict__ lidx, double * vals, const _IType nnz, const _IType * block_coords, 
                                        _FType* output, _FType* f0, _FType* f1, _FType* f2, _FType* f3,
                                        int target_mode, int rank, const _IType* dimensions);

void mttkrp_lvl2(BLCOBlock * blco_block, MAT_GPU * o_mttkrp_gpu, MAT_GPU ** i_factors_gpu, int target_mode, int rank, int target_mode_dim, _IType thread_coal_factor, cudaStream_t stream);
__global__ void mttkrp_lvl2_3d_kernel();
__global__ void mttkrp_lvl2_4d_kernel(const _IType* __restrict__ lidx, double * vals, const _IType nnz, const _IType * block_coords, 
                                        _FType* output, _FType* f0, _FType* f1, _FType* f2, _FType* f3,
                                        int target_mode, int rank, int target_mode_dim, _IType thread_coal_factor);

void gram_leave_out_one_gpu(int mode, int num_modes, MAT_GPU ** factors, MAT_GPU * gram);

__global__ void value_fill_kernel(_FType* x, _IType n, _FType val);
void value_fill(_FType* x, _IType n, _FType val);

#endif