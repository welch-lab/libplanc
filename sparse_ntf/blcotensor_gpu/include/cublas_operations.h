#pragma once

#include "cusolverDn.h"
#include "cublas_v2.h"
#include "cblas.h"

#include "cuda_runtime.h"
#include "blco.h"
#include "cuda_utils.h"
#include "cassert"

#include <cublas_v2.h>
#include <cuda_runtime.h>

class CuBlasOperations {
public:
    cudaStream_t stream;
    // cublasHandle_t cublasHandle;

    CuBlasOperations() {
        // printf("creating cublasoperation object\n");
        // Initialize CUDA Stream
        cudaStreamCreate(&stream);
        // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        // Initialize cuBLAS Handle
        // cublasCreate(&cublasHandle);
        // cublasSetStream(cublasHandle, stream);
    }

    ~CuBlasOperations() {
        // Destroy cuBLAS Handle
        // cublasDestroy(cublasHandle);
        // Destroy CUDA Stream
        cudaStreamDestroy(stream);
    }

    // Synchronize the stream to wait for all operations to finish
    void synchronize() {
        cudaStreamSynchronize(stream);
    }
};
