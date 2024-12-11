#ifndef ANLSBPP_GPU_H_
#define ANLSBPP_GPU_H_

#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "blco.h"

#include <iostream>

#define SUBPROBLEM_SIZE 100

void anlsbpp_update(MAT_GPU * fm, MAT_GPU * o_mttkrp_gpu, const MAT_GPU * gram);
#endif // ANLSBPP_GPU_H_