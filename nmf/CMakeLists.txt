#Copyright 2016 Ramakrishnan Kannan

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
include(GenerateExportHeader)
include(FetchContent)

#include(${PROJECT_SOURCE_DIR}/common/cmake_aux/NMFLIB_FindARMA.cmake)

# the settings below will be automatically configured by the rest of this script

#add_library(sparse_nmf OBJECT SparseNMFDriver.cpp)
#add_library(dense_nmf OBJECT NMFDriver.cpp)
add_library(nmflib
        nmf_lib.cpp bppnmf.cpp
        ../common/utils.cc)
# add_executable(${DENSE_OR_SPARSE}_inmf inmf.cpp)
# target_compile_features(${DENSE_OR_SPARSE}_inmf PRIVATE cxx_std_17)
# add_compile_options(-O0)
target_include_directories(nmflib PUBLIC ${PLANC_COMMON} ${NNLS_SRC} ${OpenMP_CXX_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(nmflib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nmflib PUBLIC progressWrapper utils hw_detect detect_blas)
if(BUILD_EXECUTABLE)
    add_executable(plancnmf nmf_cmd.cpp)
    find_package(argparse REQUIRED)
    target_link_libraries(plancnmf PRIVATE nmflib argparse::argparse)
    target_include_directories(plancnmf PRIVATE ${PLANC_COMMON} ${CMAKE_CURRENT_BINARY_DIR})
endif()
generate_export_header(nmflib)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
if(NOT BUILD_RCPP AND pybind11_DIR)
    FetchContent_Declare(
            carma
            URL https://github.com/RUrlus/carma/archive/refs/tags/v0.8.0.tar.gz
            URL_HASH SHA256=A404687FB7C41880794CB215CB678BC59CDB0ECDF8BAA4336103E7834BA6750A
    )
    fetchcontent_makeavailable(carma)
    pybind11_add_module(pyplanc ../interfaces/pybind_interface.cpp)
    target_link_libraries(pyplanc PRIVATE carma nmflib)
endif()
# target_link_libraries(${DENSE_OR_SPARSE}_inmf ${NMFLIB_LIBS})
#install(TARGETS  plancnmf nmflib
#        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )
