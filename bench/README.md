# Benchmarking PLANC NNLS: a guide

## A. Running the benchmark

### 1. Clone this repository

        git clone https://www.github.com/theAeon/planc.git

### 2. Obtain prerequisites

#### Windows

Install [MSYS2](https://github.com/msys2/msys2-installer/releases/download/2023-03-18/msys2-x86_64-20230318.exe) to the default directory.

Once installed, open "MSYS2 UCRT64" and run `pacman -Syu`. Accept all prompts and relaunch.

After relaunching, run `pacman -Syu` again, followed by `pacman -S pactoys`.

Install build dependencies with `pacboy -S make:u cmake:u armadillo:u`

#### Mac OS

Install [Homebrew](https://brew.sh/). Once installed, run `brew install gcc cmake`. If you plan on using Apple's Accelerate framework, **do not install armadillo through homebrew**. Instead, obtain a copy of armadillo from their [website](https://arma.sourceforge.io), untar it, and make sure to specify its location in the configure step.

#### Linux

Refer to your distribution's documentation to obtain packages. Ensure you have your distribution's glibc++ build tools installed. Obtain cmake, armadillo, and a provider of the standard C BLAS API. If you don't know which to use, OpenBLAS is never a bad option.

### 3. Configure build

From the root of the cloned repository, run `cmake --preset $preset` where $preset is

|OS|$preset|
|---|---|
|Windows|x64-release|
|Linux|linux-release|
|Mac OS|macos-release|

You may need to provide cmake with some extra command line flags to help it find dependencies. Common command line options include `-DARMADILLO_INCLUDE_DIR=/armadillo/header/location` and `-DBLA_VENDOR=$vendor` where $vendor is one of those listed [here](https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors). On Mac OS, you should be able to use vendor "Apple" but will need to locate the folder where `cblas.h` is located and provide it as `-DCMAKE_CXX_FLAGS=-I$cblas_location`. In my experience with recent versions of Mac OS it is located at `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/` but I am aware that Apple moves things around from version to version. Search engines may be your best bet here if that path doesn't work. You also may need to provide cmake the location of your compiler, particularly on Mac OS or HPC.  This can be done with `-DCMAKE_C_COMPILER`and`-DCMAKE_CXX_COMPILER`.

Due to oddities with our HPC's environment, link-time optimization is disabled by default in the linux-release prefix. Please attempt configuration with `-DCMAKE_CXX_FLAGS=-flto`. If the build (below) fails, delete CMakeCache.txt and configure without link-time optimization.

If you're feeling adventurous, append `-march=native` (`-march=native -mtune=native` on non-x86 machines) to `-DCMAKE_CXX_FLAGS` for a benchmark specialized for your machine.

### 4. Build PLANC

    cmake --build out/build/$prefix

If there are any errors, see [Troubleshooting](#troubleshooting) below.

### 5. Preparing the benchmark

From the same shell, change directory to `build/bench_in` and run `zstd -d` on each of the .mtx.zst files. Ensure your computer is under minimal load.

### 6. Running the benchmark

From the `bench_in` folder, run ```planc_bench```.

Examples:

- On Windows:

        UMHS+robbiand@MM-HLFVRQ3 UCRT64 /c/users/robbiand/planc/bench/bench_in
        $ ../../out/build/x64-release/bench/planc_bench.exe
        Successfully loaded input matrices ::A::1054x6115::B::50x1054(0.090007 s)
        Successfully loaded input matrices ::A::1274x30578::B::50x1274(0.403846 s)
        Successfully loaded input matrices ::A::1304x61157::B::50x1304(0.789816 s)
        Successfully loaded input matrices ::A::1312x122314::B::50x1312(1.52011 s)
        Successfully loaded input matrices ::A::1410x156167::B::50x1410(2.03139 s)
        total nnls runtime=0.00200009
        total nnls runtime=0
        ...you get the idea

- On Linux/Mac OS:

        [robbiand@gl-login3]
        /home/robbiand/planc/bench/bench_in ->$ ../../out/build/linux-release/bench/planc_bench
        Successfully loaded etc etc etc

### 7. Submitting the benchmarks

The executable should have dropped a file in `bench/bench_in` called `outbench.csv`. Send this file along with your system details (CPU Model `e.g. Intel i5-1235U`, RAM, OS, OS version, compiler, compiler version, any extra compiler flags, anything else that seems relevant) to [Andrew Robbins](mailto:robbiand@med.umich.edu)

## Troubleshooting

Any issues solved will be written up here. In the meantime, contact me above or message me on [Matrix](matrix:u/andrew:robbinsa.me?action=chat)!

## Known good configurations

This benchmark has been tested on 64-bit Windows 11 with the MSYS2 UCRT gcc/openblas toolchain, on virtualized Ubuntu 22.04 with AMD's optimized BLIS implementation [AOCL](https://www.amd.com/en/developer/aocl.html), and on UMich's standard [Great Lakes](https://arc.umich.edu/greatlakes/configuration/) node with personally compiled builds of gcc 12 and OpenBLAS.
