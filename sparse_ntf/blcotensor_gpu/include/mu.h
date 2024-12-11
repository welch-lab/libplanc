#ifndef MU_GPU_H_
#define MU_GPU_H_

#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "blco.h"
#include <iostream>

void mu_update(MAT_GPU * fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram);
void mu_update_opt(MAT_GPU * fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram);
#endif // MU_GPU_H_