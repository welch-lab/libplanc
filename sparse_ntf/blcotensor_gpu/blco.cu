#include <iostream>
#include "blco.h"
#include "cuda_utils.h"

// #include "blco_tensor.hpp"
#include <cooperative_groups.h>
#include <chrono>

namespace cg = cooperative_groups;

__constant__ _IType MASKS[MAX_NUM_MODES];
__constant__  int POS[MAX_NUM_MODES];


template <typename LIT>
__device__ inline _IType alt_pext(LIT x, int pos, _IType mask, _IType block_coord) {
    return ((x >> pos) & mask) | block_coord;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
  
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

BLCOBlock * generate_blco_block_host(int nmodes, _IType nnz) {
  BLCOBlock * b = new BLCOBlock;
  b->block_coords = (_IType *) malloc(sizeof(_IType) * nmodes);
  b->m_modes = nmodes;
  b->m_numel = nnz;
  return b;
}

void set_mat_to_zero(MAT_GPU * mat) {
  int block_size = TILE_SIZE;
  int num_elements = mat->n_cols * mat->n_rows;
  int num_blocks = (num_elements + block_size - 1) / block_size;
  initializeMatrix<<<num_blocks, block_size>>>(mat->vals, num_elements);
  check_cuda(cudaDeviceSynchronize(), "set mat_gpu values to zero");
}


__global__ void initializeMatrix(double* vals, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    vals[idx] = 0.0f;
  }
}

MAT_GPU * send_mat_to_gpu(MAT * mat) {
  
  int num_elements = mat->n_rows * mat->n_cols;
  MAT_GPU * _mat = new MAT_GPU(mat->n_rows, mat->n_cols);
  
  check_cuda(cudaMalloc((void**)&_mat->vals, sizeof(double) * num_elements), "cudaMalloc mat to gpu");
  check_cuda(cudaMemcpy(_mat->vals, mat->memptr(), sizeof(double) * num_elements, cudaMemcpyHostToDevice), "cudaMemcpy mat to gpu");
  cudaDeviceSynchronize(); // Wait for the data transfer to complete

  return _mat;
};

MAT_GPU ** send_mats_to_gpu(MAT * mats, int num_modes) {
  MAT_GPU ** _mats_gpu = new MAT_GPU*[num_modes];
  for (int m = 0; m < num_modes; ++m) {
    _mats_gpu[m] = send_mat_to_gpu(&mats[m]);
  }
  return _mats_gpu;
};

void send_mat_to_host(MAT_GPU * o_mat_gpu, MAT * o_mat_host) {
  int num_elements = o_mat_host->n_rows * o_mat_host->n_cols;
  check_cuda(cudaMemcpy(o_mat_host->memptr(), o_mat_gpu->vals, sizeof(_FType) * num_elements, cudaMemcpyDeviceToHost), "cudaMemcpy mat to host");
}

void send_umat_to_host(UMAT_GPU * o_mat_gpu, arma::Mat<unsigned int> * o_mat_host) {
  int num_elements = o_mat_host->n_rows * o_mat_host->n_cols;
  check_cuda(cudaMemcpy(o_mat_host->memptr(), o_mat_gpu->vals, sizeof(unsigned int) * num_elements, cudaMemcpyDeviceToHost), "cudaMemcpy umat to host");
}


// Function to free GPU memory for a MAT_GPU object
void free_mat_on_gpu(MAT_GPU * mat_gpu) {
    cudaFree(mat_gpu->vals);  // Free the GPU data memory
    delete mat_gpu;           // Delete the MAT_GPU object
}

void free_mats_on_gpu(MAT_GPU ** mats_gpu, int num_modes) {
    for (int m = 0; m < num_modes; ++m) {
        // Assuming you have a function to free GPU memory for MAT_GPU
        free_mat_on_gpu(mats_gpu[m]);
    }
    delete[] mats_gpu;
}

// Not streaming
void send_blco_block_gpu(BLCOBlock * blk_host, BLCOBlock * blk_dev) {
  check_cuda(cudaMemcpy(blk_dev->block_coords, blk_host->block_coords, blk_host->m_modes * sizeof(_IType), cudaMemcpyHostToDevice), "cudaMemcpy block_coords");
  check_cuda(cudaMemcpy(blk_dev->idx, blk_host->idx, blk_host->m_numel * sizeof(_IType), cudaMemcpyHostToDevice), "cudaMemcpy block idx");
  check_cuda(cudaMemcpy(blk_dev->vals, blk_host->vals, blk_host->m_numel * sizeof(_FType), cudaMemcpyHostToDevice), "cudaMemcpy block vals");
  check_cuda(cudaDeviceSynchronize(), "cudaMemcpy BLCOblocks");
}

void send_blco_block_gpu_async(BLCOBlock * blk_host, BLCOBlock * blk_dev, cudaStream_t stream) {
  check_cuda(cudaMemcpyAsync(blk_dev->block_coords, blk_host->block_coords, blk_host->m_modes * sizeof(_IType), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync block_coords");
  check_cuda(cudaMemcpyAsync(blk_dev->idx, blk_host->idx, blk_host->m_numel * sizeof(_IType), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync block idx");
  check_cuda(cudaMemcpyAsync(blk_dev->vals, blk_host->vals, blk_host->m_numel * sizeof(_FType), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync block vals");
}

BLCOBlock * generate_blco_block_gpu(BLCOBlock * block) {
  int m_modes = block->m_modes;
  _IType nnz = block->m_numel;

  BLCOBlock * b = new BLCOBlock;
  b->m_modes = m_modes;
  b->m_numel = nnz;

  check_cuda(cudaMalloc(&b->block_coords, sizeof(_IType) * m_modes), "cudaMalloc block_coords");
  check_cuda(cudaMalloc(&b->idx, sizeof(_IType) * nnz), "cudaMalloc block idx");
  check_cuda(cudaMalloc(&b->vals, sizeof(_FType) * nnz), "cudaMalloc block vals");

  check_cuda(cudaDeviceSynchronize(), "cudaMalloc BLCOblocks");
  return b;
}

BLCOTensorGPU * generate_blco_tensor_gpu(
  _IType max_block_size,
  int num_modes,
  unsigned long long * dimensions,
  _IType numels,
  _IType * mode_mask,
  int * mode_pos,
  _IType block_count,
  BLCOBlock ** blocks, 
  bool use_stream) {
    BLCOTensorGPU * bt = new BLCOTensorGPU;
    bt->m_modes = num_modes;
    bt->m_blco_mode_mask = make_device_copy(mode_mask, num_modes, "cudaMemcpy mode_masks");
    bt->m_blco_mode_pos = make_device_copy(mode_pos, num_modes, "cudaMemcpy mode_pos");
    bt->dims = make_device_copy(dimensions, num_modes, "cudaMemcpy dimensions");
    
    bt->m_numel = numels;
    bt->m_num_blocks = block_count;
    bt->m_blocks = new BLCOBlock*[block_count];
    // bt->m_blocks_dev_staging -- no need yet
    // bt->m_blocks_dev_ptr
    bt->m_streams = new cudaStream_t[block_count];

    cudaMemcpyToSymbol(MASKS, mode_mask, num_modes * sizeof(_IType));
    cudaMemcpyToSymbol(POS, mode_pos, num_modes * sizeof(int));

    printf("====== BLCO_GPU Info ======\n");
    printf("nnz: %d, num_blocks: %d, num_modes: %d\n",
      bt->m_numel, bt->m_num_blocks, bt->m_modes);
    for (int m = 0; m < num_modes; ++m) {
      printf("dim: %llu, mode_mask[%d]: %llx, mode_pos: %d\n", dimensions[m], m, mode_mask[m], mode_pos[m]);
    }
  
    // what is this for?
    bool do_batching = false;
    if (do_batching) {
      // set up warp stuff
    } else {
      bt->warp_info_length = 0;
      bt->warp_info = nullptr;
      bt->warp_info_gpu = nullptr;
    }

    if (!use_stream) {
      for (int b = 0; b < block_count; ++b) {
        bt->m_blocks[b] = generate_blco_block_gpu(blocks[b]);
        // Create stream as default and use stream[0] when not actually using
        check_cuda(cudaStreamCreate(&bt->m_streams[b]), "cudaStreamCreate");
      }
    }
    return bt;
};

void mttkrp_lvl1(BLCOBlock * b_block, MAT_GPU * o_mttkrp_gpu, MAT_GPU ** i_factors_gpu, int target_mode, int rank, _IType* dimensions, cudaStream_t stream) {
  float etime = 0; 
  float ms;
  cudaEvent_t start_event, stop_event;
  check_cuda(cudaEventCreate(&start_event), "cudaEventCreate");
  check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate");

  int nnz_block = TILE_SIZE; // threads per block
  _IType blocks = (b_block->m_numel + nnz_block - 1) / nnz_block; // num. of blocks

  // better sizing for shared memory?
  int smem_sz = nnz_block * (sizeof(_FType) + (b_block->m_modes+1) * sizeof(_IType) + sizeof(int));

  cudaEventRecord(start_event, 0);
  
  if (b_block->m_modes == 3) {
    mttkrp_lvl1_3d_kernel<<<blocks, nnz_block, smem_sz>>>();
  } else if (b_block->m_modes == 4) {
    mttkrp_lvl1_4d_kernel<<<blocks, nnz_block, smem_sz, stream>>>(
      b_block->idx, b_block->vals, b_block->m_numel, b_block->block_coords, 
      o_mttkrp_gpu->vals, i_factors_gpu[0]->vals, i_factors_gpu[1]->vals, i_factors_gpu[2]->vals, i_factors_gpu[3]->vals,
      target_mode, rank, dimensions);
      // check_cuda(cudaDeviceSynchronize(), "sync after mttkrp_lvl1_kernel");
  } else {
    printf("Error: BLCO only supports tensors of 3 or 4 modes!\n");
  }
  
  check_cuda(cudaEventRecord(stop_event, 0), "cudaEventRecord");
  check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchroize"); // implicit barrier
  check_cuda(cudaEventElapsedTime(&ms, start_event, stop_event), "cudaEventElapsedTime");
  check_cuda(cudaGetLastError(), "mttkrp_lvl1_kernel launch. Exceeded shared mem space?");
  
  etime += ms;

  // compute norm for output

  // float norm = 0.0;
  // compute_norm<<<blocks, nnz_block>>>
  // printf("actual kernel: %f\n", etime);
}

// Register- and smem-based conflict resolution using tile-based execution with thread coarsening
void mttkrp_lvl2(
  BLCOBlock * b_block, MAT_GPU * o_mttkrp_gpu, MAT_GPU ** i_factors_gpu, 
  int target_mode, int rank, int target_mode_dim, _IType thread_coal_factor, cudaStream_t stream) {

  int nnz_block = TILE_SIZE * thread_coal_factor; // threads per block
  _IType blocks = (b_block->m_numel + nnz_block - 1) / nnz_block; // num. of blocks

  // better sizing for shared memory?
  int smem_sz = TILE_SIZE * (sizeof(_FType) + (b_block->m_modes+1) * sizeof(_IType) + sizeof(int));
  smem_sz += (rank + 1) * STASH_SIZE * sizeof(_FType);
  
  if (b_block->m_modes == 3) {
    mttkrp_lvl2_3d_kernel<<<blocks, TILE_SIZE, smem_sz, stream>>>();
  } else if (b_block->m_modes == 4) {
    mttkrp_lvl2_4d_kernel<<<blocks, TILE_SIZE, smem_sz, stream>>>(
      b_block->idx, b_block->vals, b_block->m_numel, b_block->block_coords, 
      o_mttkrp_gpu->vals, i_factors_gpu[0]->vals, i_factors_gpu[1]->vals, i_factors_gpu[2]->vals, i_factors_gpu[3]->vals,
      target_mode, rank, target_mode_dim, thread_coal_factor);
      // check_cuda(cudaDeviceSynchronize(), "sync after mttkrp_lvl2_kernel");
  } else {
    printf("Error: BLCO only supports tensors of 3 or 4 modes!\n");
  }
  // printf("actual kernel: %f\n", etime);
}


__global__ void mttkrp_lvl2_3d_kernel() {}

__global__ void mttkrp_lvl2_4d_kernel(
  const _IType* __restrict__ lidx, double * vals,
  const _IType nnz, 
  const _IType * block_coords,
  _FType* output, _FType* f0, _FType* f1, _FType* f2, _FType* f3,
  const int tmode, const int rank, const int target_mode_dim, _IType thread_coal_factor) {
  
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<TILE_SIZE>(block); // setting the tile size and block size as the same
  const int tid = block.thread_rank();

  // setup caches (shared memory) within thread block
  extern __shared__ int count[]; // [block.size()] 
  _IType *nnz_idx = (_IType*) (count + block.size()); // nnz_idx for each thread
  _IType *nnz_out = (_IType*) (nnz_idx + 4 * block.size()); // [block.size()] 
  _FType* nnz_val = (_FType*) (nnz_out + block.size()); // [block.size()] 
  _FType* data = (_FType*) (nnz_val + block.size()); // [rank * STASH_SIZE]
  _IType* tags = (_IType*) (data + rank * STASH_SIZE); // Output row ID [STASH_SIZE]
  for (int i = tid; i < STASH_SIZE; i += block.size()) {
      tags[i] = INVALID_ID;
  }
  _IType curr_elem = block.group_index().x * block.size() * thread_coal_factor; // Index of start element
  _IType end_elem = min(nnz, curr_elem + block.size() * thread_coal_factor); // Index of last element
      
  const int mp0 = POS[0];
  const int mp1 = POS[1];
  const int mp2 = POS[2];
  const int mp3 = POS[3];
  const _IType mm0 = MASKS[0];
  const _IType mm1 = MASKS[1];
  const _IType mm2 = MASKS[2];
  const _IType mm3 = MASKS[3];
  const _IType bc0 = block_coords[0];
  const _IType bc1 = block_coords[1];
  const _IType bc2 = block_coords[2];
  const _IType bc3 = block_coords[3];

  while (curr_elem < end_elem) {
    // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
    count[tid] = 0;
    _IType idx; // should be LIT
    _IType x, y, z, w, output_row;
    if (curr_elem + tid < end_elem) { // within each thread block
      idx = lidx[curr_elem + tid]; // lidx for non-zero curr_elem + tid
      
      x = alt_pext(idx, mp0, mm0, bc0);
      y = alt_pext(idx, mp1, mm1, bc1);
      z = alt_pext(idx, mp2, mm2, bc2);
      w = alt_pext(idx, mp3, mm3, bc3);
      
      if (tmode == 0) output_row = x;
      else if (tmode == 1) output_row = y;
      else if (tmode == 2) output_row = z;
      else output_row = w;
    } else {
      x = y = z = w = output_row = (_IType) - 1; // invalid value.. work is done
    }
    // if (curr_elem == 0) {
    // // for (int m = 0; m < 4; ++m) {
    //   printf("x: %llu, y: %llu, z: %llu, w: %llu\n", x, y, z, w);
    // // }
    // }
    block.sync();

    // Sorting nnzs based for hierarchical conflict resolution
    int sg_mask = tile.match_any(output_row); // bitmask 1 for threads that have matching output_row in tile
    auto sg = cg::labeled_partition(tile, sg_mask); // threads that have matching output_row
    int sg_rank = sg.thread_rank(); // 
    int sg_id = sg.meta_group_rank(); // rank of the cg within parent thread block
    if (sg_rank == 0) count[sg_id+1] = sg.size(); // OOB writes will be overwritten later. counting sort histogram
    
    // count has size block.size()
    // each element is initilized as 0 from count[tid] = 0
    // sg_mask has 1 if current thread matches output_row
    // sg is the partition based on sg_mask (same output_row value)
    // sg_rank and sg_id retrieves the rank of the thread and the meta group respectively
    // for only the master thread, update the count of the meta_group -- histogram
    // so if first meta group has 3 matching to non-zeros
    // count[0] = 0, count[1] = 3, 
    // if second meta group has 2 matching to non-zeros???...
    // count[2] = 2
    block.sync();

    // Scan for counting sort
    sg_mask = count[tid];
    //block.sync();
    #pragma unroll
    for (int j = 1; j < tile.size(); j <<= 1) { // tile size is 32, 2^^5
    // value to be shuffled: sg_mask, distance it will be shuffled: j
    // shuffles the value of sg_mask from a distance of j threads up the warp
      int temp = tile.shfl_up(sg_mask, j);
      // if thread is beyond the distance j from the start of the warp
      // shuffled value is added to sg_mask...? why?
      if (tid >= j) sg_mask += temp;
    }
    count[tid] = sg_mask;
    block.sync();

    // Sorted rank
    sg_rank += count[sg_id];

    // Strided access to facilitate broadcast later
    nnz_idx[sg_rank * 4]  = x;
    nnz_idx[sg_rank * 4 + 1]  = y;
    nnz_idx[sg_rank * 4 + 2 ]  = z;
    nnz_idx[sg_rank * 4 + 3 ]  = w;
    nnz_out[sg_rank] = output_row;
    if (curr_elem+tid < end_elem) nnz_val[sg_rank] = vals[curr_elem+tid];
    
    // Segmented scan structure (reuse sg_mask).
    if (sg.thread_rank() == 0) sg_mask = 1<<sg_rank;
    else sg_mask = 0;

    #pragma unroll
    for (int j = tile.size()/2; j > 0; j >>= 1) {
        sg_mask |= tile.shfl_down(sg_mask, j);
    }
    sg_mask = tile.shfl(sg_mask, 0);

    // Now threads perform rank-wise operations.
    int n = 0;
    while (n < block.size() &&  (curr_elem + n) < end_elem) {
    //block.sync();
    // Prep stash line
    const _IType output_row = nnz_out[n];
    int stash_line = (int) output_row & (STASH_SIZE - 1); // Modulo hash function
    if (tags[stash_line] == INVALID_ID) {
        // Initialize cache line
        for (_IType i = tid; i < rank; i += block.size()) {
            data[stash_line * rank + i] = 0.0;
        }
        if (tid == 0) tags[stash_line] = output_row;
    }
    else if (tags[stash_line] != output_row) {
        // Evict cache line to global mem (evict-first policy)
        for (_IType i = tid; i < rank; i += block.size()) {
            atomicAdd(output + tags[stash_line] * rank + i, data[stash_line * rank + i]);
            data[stash_line * rank + i] = 0.0;
        } 
        if (tid == 0) tags[stash_line] = output_row;
      }
    block.sync();

    // Perform update
    const int next_n = n;
    for (_IType i = tid; i < rank; i += block.size()) {               
        // Register-based accumlation
        _FType value = 0.0;
        n = next_n;
        do {
            // Broadcast 
            _FType val = nnz_val[n];
            x = nnz_idx[n * 4];
            y = nnz_idx[n * 4 + 1];
            z = nnz_idx[n * 4 + 2];
            w = nnz_idx[n * 4 + 3];
            
            if (tmode == 0) val *= f1[rank * y + i] * f2[rank * z + i] * f3[rank * w + i];
            else if (tmode == 1) val *= f0[rank * x + i] * f2[rank * z + i] * f3[rank * w + i];
            else if (tmode == 2) val *= f0[rank * x + i] * f1[rank * y + i] * f3[rank * w + i];                 
            else val *= f0[rank * x + i] * f1[rank * y + i] * f2[rank * z + i];
            
            value += val;
            ++n;
        } while (n < block.size() && !(sg_mask & (1<<n)));
        data[stash_line * rank + i] += value;                   
    } // rank
    // broadcast n
    n = tile.shfl(n, 0);


    } // block.size()
    curr_elem += block.size();
  }
  // Write STASH to global
  #pragma unroll
  for (int stash_line = 0; stash_line < STASH_SIZE; stash_line++) {
      _IType output_row = tags[stash_line];
      if (output_row != INVALID_ID) {
          for (_IType i = tid; i < rank; i += block.size()) {
              atomicAdd(output + output_row * rank + i, data[stash_line * rank + i]);
          }
      }
  }
}

__global__ void mttkrp_lvl1_3d_kernel() {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<TILE_SIZE>(block);
  const int tid = block.thread_rank();

  printf("tid: %d\n", tid);

  // set up caches (stash)
};

__global__ void mttkrp_lvl1_4d_kernel(
  const _IType* __restrict__ lidx, double * vals,
  const _IType nnz, 
  const _IType * block_coords,
  _FType* output, _FType* f0, _FType* f1, _FType* f2, _FType* f3,
  const int tmode, const int rank, const _IType* dimensions) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<TILE_SIZE>(block); // setting the tile size and block size as the same
  const int tid = block.thread_rank();

  // setup caches (shared memory) within thread block
  extern __shared__ int count[]; // [block.size()] 
  _IType *nnz_idx = (_IType*) (count + block.size()); // nnz_idx for each thread
  _IType *nnz_out = (_IType*) (nnz_idx + 4 * block.size()); // [block.size()] 
  _FType* nnz_val = (_FType*) (nnz_out + block.size()); // [block.size()] 

  // Identify block-level workload
  _IType curr_elem = block.group_index().x * block.size(); // Index of start element
  _IType end_elem = min(nnz, curr_elem + block.size()); // Index of last element

  const int mp0 = POS[0];
  const int mp1 = POS[1];
  const int mp2 = POS[2];
  const int mp3 = POS[3];
  const _IType mm0 = MASKS[0];
  const _IType mm1 = MASKS[1];
  const _IType mm2 = MASKS[2];
  const _IType mm3 = MASKS[3];
  const _IType bc0 = block_coords[0];
  const _IType bc1 = block_coords[1];
  const _IType bc2 = block_coords[2];
  const _IType bc3 = block_coords[3];

  while (curr_elem < end_elem) {
    // Threads collaborate to perform On-the-fly delinerization, sorting, and segmented scan. 
    count[tid] = 0;
    _IType idx; // should be LIT
    _IType x, y, z, w, output_row;
    if (curr_elem + tid < end_elem) { // within each thread block
      idx = lidx[curr_elem + tid]; // lidx for non-zero curr_elem + tid
      
      x = alt_pext(idx, mp0, mm0, bc0);
      y = alt_pext(idx, mp1, mm1, bc1);
      z = alt_pext(idx, mp2, mm2, bc2);
      w = alt_pext(idx, mp3, mm3, bc3);
      
      if (tmode == 0) output_row = x;
      else if (tmode == 1) output_row = y;
      else if (tmode == 2) output_row = z;
      else output_row = w;
    } else {
      x = y = z = w = output_row = (_IType) - 1; // invalid value.. work is done
    }
    // if (curr_elem == 0) {
    // // for (int m = 0; m < 4; ++m) {
    //   printf("x: %llu, y: %llu, z: %llu, w: %llu\n", x, y, z, w);
    // // }
    // }
    block.sync();

    // Sorting nnzs based for hierarchical conflict resolution
    int sg_mask = tile.match_any(output_row); // bitmask 1 for threads that have matching output_row in tile
    auto sg = cg::labeled_partition(tile, sg_mask); // threads that have matching output_row
    int sg_rank = sg.thread_rank(); // 
    int sg_id = sg.meta_group_rank(); // rank of the cg within parent thread block
    if (sg_rank == 0) count[sg_id+1] = sg.size(); // OOB writes will be overwritten later. counting sort histogram
    
    // count has size block.size()
    // each element is initilized as 0 from count[tid] = 0
    // sg_mask has 1 if current thread matches output_row
    // sg is the partition based on sg_mask (same output_row value)
    // sg_rank and sg_id retrieves the rank of the thread and the meta group respectively
    // for only the master thread, update the count of the meta_group -- histogram
    // so if first meta group has 3 matching to non-zeros
    // count[0] = 0, count[1] = 3, 
    // if second meta group has 2 matching to non-zeros???...
    // count[2] = 2
    block.sync();

    // Scan for counting sort
    sg_mask = count[tid];
    //block.sync();
    #pragma unroll
    for (int j = 1; j < tile.size(); j <<= 1) { // tile size is 32, 2^^5
    // value to be shuffled: sg_mask, distance it will be shuffled: j
    // shuffles the value of sg_mask from a distance of j threads up the warp
      int temp = tile.shfl_up(sg_mask, j);
      // if thread is beyond the distance j from the start of the warp
      // shuffled value is added to sg_mask...? why?
      if (tid >= j) sg_mask += temp;
    }
    count[tid] = sg_mask;
    block.sync();

    // Sorted rank
    sg_rank += count[sg_id];

    // Strided access to facilitate broadcast later
    nnz_idx[sg_rank * 4]  = x;
    nnz_idx[sg_rank * 4 + 1]  = y;
    nnz_idx[sg_rank * 4 + 2 ]  = z;
    nnz_idx[sg_rank * 4 + 3 ]  = w;
    nnz_out[sg_rank] = output_row;
    if (curr_elem+tid < end_elem) nnz_val[sg_rank] = vals[curr_elem+tid];
    
    // Segmented scan structure (reuse sg_mask).
    if (sg.thread_rank() == 0) sg_mask = 1<<sg_rank;
    else sg_mask = 0;

    #pragma unroll
    for (int j = tile.size()/2; j > 0; j >>= 1) {
        sg_mask |= tile.shfl_down(sg_mask, j);
    }
    sg_mask = tile.shfl(sg_mask, 0);

    // Now threads perform rank-wise operations.
    int n = 0;
    while (n < block.size() && (curr_elem + n) < end_elem) {
      // block.sync();  
      // Perform update
      const _IType output_row = nnz_out[n];
      const int next_n = n;

      _IType d0 = dimensions[0]; // x
      _IType d1 = dimensions[1]; // y 
      _IType d2 = dimensions[2]; // z
      _IType d3 = dimensions[3]; // w

      _IType target_mode_dim = dimensions[tmode];

      for (_IType i = tid; i < rank; i += block.size()) {
        // Register-based accumlation
        _FType value = 0.0;
        n = next_n;
        do {
          // Broadcast 
          _FType val = nnz_val[n];
          x = nnz_idx[n * 4];
          y = nnz_idx[n * 4 + 1];
          z = nnz_idx[n * 4 + 2];
          w = nnz_idx[n * 4 + 3];
          
          
          // if (tmode == 0) val *= f1[rank * y + i] * f2[rank * z + i] * f3[rank * w + i];
          // else if (tmode == 1) val *= f0[rank * x + i] * f2[rank * z + i] * f3[rank * w + i];
          // else if (tmode == 2) val *= f0[rank * x + i] * f1[rank * y + i] * f3[rank * w + i];                 
          // else val *= f0[rank * x + i] * f1[rank * y + i] * f2[rank * z + i];

          // f1, f2, .. are stored in column major format (I x R)
          if (tmode == 0) val *= f1[d1 * i + y] * f2[d2 * i + z] * f3[d3 * i + w];
          else if (tmode == 1) val *= f0[d0 * i + x] * f2[d2 * i + z] * f3[d3 * i + w];
          else if (tmode == 2) val *= f0[d0 * i + x] * f1[d1 * i + y] * f3[d3 * i + w];                 
          else val *= f0[d0 * i + x] * f1[d1 * i + y] * f2[d2 * i + z];
          
          value += val;
          ++n;
        } while (n < block.size() && !(sg_mask & (1<<n)));

        // row-major R x I
        // atomicAdd(output + output_row * rank + i, value);

        // column major I x R
        atomicAdd(output + target_mode_dim * i + output_row, value);
      } // rank
      // broadcast n
      // if (curr_elem == 0) {
      // // for (int m = 0; m < 4; ++m) {
      //   printf("x:_ %llu, y:_ %llu, z:_ %llu, w:_ %llu\n", nnz_idx[n*4], nnz_idx[n*4+1], nnz_idx[n*4+2], nnz_idx[n*4+3]);
      // }
      n = tile.shfl(n, 0);
    } // block.size()
    curr_elem += block.size();
  };
};

void gram_leave_out_one_gpu(int mode, int num_modes, MAT_GPU ** factors, MAT_GPU * gram) {
  int gram_m = gram->n_rows;
  int gram_n = gram->n_cols;

  double alpha = 1.0;
  double beta = 0.0;
  // Init gram matrix to ones
  double * temp;
  check_cuda(cudaMalloc((void**)&temp, sizeof(double) * gram_m * gram_n), "cudaMalloc temp matrix for gram mat");
  value_dfill(gram->vals, gram_m * gram_n, 1.0);
  check_cuda(cudaDeviceSynchronize(), "value fill");

  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "create cublas handle");
  int m, n;

  for (int j = 0; j < num_modes; ++j) {
    if (j != mode) {
      // gram *= A.T * A
      m = factors[j]->n_rows;
      n = factors[j]->n_cols;
      check_cublas(
        cublasDgemm(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, 
          n, n, m, &alpha, factors[j]->vals, m, factors[j]->vals, m, &beta, temp, n),
        "computing gram matrix (A.T * A)"
      );
      vector_hadamard(gram->vals, temp, gram_m * gram_n);
      check_cuda(cudaDeviceSynchronize(), "vector hadamard");
    }
  }
  cudaFree(temp);
  cublasDestroy(handle);
}
