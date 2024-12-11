# Sparse NTF
## Description
- This implementation is based on the following [work](https://dl.acm.org/doi/abs/10.1145/3673038.3673128) by Soh *et al* (ICPP24).
- Supports nonnegative sparse tensor factorization using [ALTO](https://arxiv.org/abs/2102.10245) and [BLCO](https://dl.acm.org/doi/abs/10.1145/3524059.3532363) sparse tensor formats
## Install Instructions

This program depends on:

- [Armadillo](https://arma.sourceforge.net) library for linear algebra operations.
- [OpenBLAS](https://github.com/xianyi/OpenBLAS) for optimized kernels. If building with OpenBLAS and MKL is discoverable by `cmake`, use `-DCMAKE_IGNORE_MKL=1` while building. 
- C++ Compiler with BMI2 and Parallel Sort Support:
  - The program requires a compiler that supports the -mbmi2 flag for bit manipulation instructions (e.g., GCC or Clang).
  - It also uses GCC's parallel sort feature, enabled by defining GLIBCXX_PARALLEL.
- CUDA (Optional): If CUDA is found, the program enables CUDA support to use BLCO-based sparse NTF (Nonnegative Tensor Factorization). This requires:
  - A CUDA-capable GPU.
  - The blcotensor_gpu subdirectory, which is added as a dependency when CUDA is available.
  - CUDA-specific properties like separable compilation (CUDA_SEPARABLE_COMPILATION).

Once the above steps are completed, set the following environment variables.

````
export ARMADILLO_INCLUDE_DIR=/home/rnu/libraries/armadillo-6.600.5/include/
export LIB=$LIB:/home/rnu/libraries/openblas/lib:
export INCLUDE=$INCLUDE:/home/rnu/libraries/openblas/include:$ARMADILLO_INCLUDE_DIR:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rnu/libraries/openblas/lib/:
export NMFLIB_DIR=/ccs/home/ramki/rhea/research/nmflib/
export INCLUDE=$INCLUDE:$ARMADILLO_INCLUDE_DIR:$NMFLIB_DIR
export CPATH=$CPATH:$INCLUDE:
export MKLROOT=/ccs/compilers/intel/rh6-x86_64/16.0.0/mkl/
````
If you have got MKL, please source `mklvars.sh` before running `make`/`cmake`

* Create a build directory. 
* Change to the build directory 
* In case of MKL, `source` the ````$MKLROOT/bin/mkl_vars.sh intel64````
* run `cmake` [PATH TO THE CMakeList.txt]. 
* `make`

Refer to the build scripts found in the [build](build/README.md) directory for a sample script.

### Input types
Supports sparse input tensors in COO formats. Refer to [FROSTT](http://frostt.io/tensors/) for sample tensors

### Other Macros

`cmake` macros
2. `-DCMAKE_BUILD_TYPE=Debug`: For creating debug builds.
3. `-DCMAKE_WITH_BARRIER_TIMING`: Default is set to barrier timing. To disable build with `-DCMAKE_WITH_BARRIER_TIMING:BOOL=OFF`.
4. `-DCMAKE_BUILD_CUDA`: Default is off.
5. `-DCMAKE_IGNORE_MKL`: Ignores MKL when discoverable by `cmake`. If using OpenBLAS, set `-DCMAKE_IGNORE_MKL=1` while building.


## Runtime usage

Tell OpenBlas how many threads you want to use. For example on a quad core system use the following.

```
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:MKL_LIB
```

## Command Line options
For a full list of command line options please run `./sparse_ntf -h`. Shares most options in `./dense_ntf` but has following additional options specifically for sparse ntf.
* `-i` Input tensor to run on. Should point to path to `*.tns` file.
* `--gpu_offload={0, 2}` 
  * 0: CPU based Sparse NTF (using ALTO format)
  * 2: GPU based Sparse NTF (using BLCO format -- fully GPU offloaded, requires **CUDA**)

## Citation

Please refer to the [papers](papers.md) section for the appropriate reports to cite.
