#ifndef HALS_GPU_H_
#define HALS_GPU_H_

#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "blco.h"
#include "vector_ops.h"
#include <iostream>

void hals_update(MAT_GPU * fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram);
#endif // HALS_GPU_H_