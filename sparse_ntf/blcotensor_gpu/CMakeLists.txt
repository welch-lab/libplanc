cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(blco_gpu LANGUAGES CUDA CXX)

# Set CUDA standards
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

add_library(blco_gpu STATIC ${CUDA_SOURCES})

# if necessary / not sure what it does
set_target_properties(blco_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Specify CUDA architectures (adjust based on your target hardware)
# set_target_properties(blco_gpu PROPERTIES CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -O3 -gencode=arch=compute_80,code=sm_80")

# Export the library for use in other projects
target_include_directories(blco_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(blco_gpu PRIVATE ${CUDA_LIBRARIES} -lcusolver)