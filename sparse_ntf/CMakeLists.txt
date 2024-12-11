cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
set(ARMA_USE_WRAPPER true)

project(SPNTF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SPNTFLIB_COMMON ${SPNTF_SOURCE_DIR}/../common)
set(SPNNLS_SRC ${SPNTF_SOURCE_DIR}/../nnls/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mbmi2 -O3 -march=native") # for bit operations for ALTO

message(STATUS "SPNTF_SOURCE_DIR =  ${SPNTF_SOURCE_DIR}")
message(STATUS "SPNTFLIB_COMMON =  ${SPNTFLIB_COMMON}")

include(${SPNTFLIB_COMMON}/cmake_aux/NMFLIB_FindARMA.cmake)

message(STATUS "CMAKE_CXX_FLAGS           = ${CMAKE_CXX_FLAGS}"          )
message(STATUS "CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_REQUIRED_INCLUDES   = ${CMAKE_REQUIRED_INCLUDES}"  )

message(STATUS "*** libraries:")
message(STATUS "*** NTFLIB_LIBS = ${NTFLIB_LIBS} ${NMFLIB_LIBS}")

# GCC uses parallel sort (e.g. std::sort)
add_definitions(-D_GLIBCXX_PARALLEL)

include_directories(
  ${ARMADILLO_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${SPNTF_SOURCE_DIR}
  ${SPNTF_SOURCE_DIR}/../
  ${SPNTFLIB_COMMON}
  ${SPNNLS_SRC}
  ${Boost_INCLUDE_DIR}
  ${MKL_INCLUDE_DIR}
  ${OPENBLAS_INCLUDE_DIR}
)

add_executable(sparse_ntf sparse_ntf.cpp)
target_compile_options(sparse_ntf PRIVATE -O3)
option(USE_BLCO "Use BLCO tensors (requires CUDA)" 0)

if(CUDA_FOUND)
  message(STATUS "CUDA found, enabling CUDA support and enabling BLCO based sparse NTF")
  enable_language(CUDA)
  add_subdirectory(blcotensor_gpu)
  add_compile_definitions(USE_BLCO=1)
  target_include_directories(sparse_ntf PRIVATE ${CMAKE_SOURCE_DIR}/blcotensor_gpu/include)
  target_link_libraries(sparse_ntf PRIVATE ${NTFLIB_LIBS} ${NMFLIB_LIBS} blco_gpu)
  set_target_properties(sparse_ntf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
  message(STATUS "CUDA not found, just using ALTO based sparse NTF")
  add_compile_definitions(USE_BLCO=0)
  target_link_libraries(sparse_ntf PRIVATE ${NTFLIB_LIBS} ${NMFLIB_LIBS})
endif()
