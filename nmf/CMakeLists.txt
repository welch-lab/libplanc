#Copyright 2016 Ramakrishnan Kannan

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
include(GenerateExportHeader)

#include(${PROJECT_SOURCE_DIR}/common/cmake_aux/NMFLIB_FindARMA.cmake)

# the settings below will be automatically configured by the rest of this script

#add_library(sparse_nmf OBJECT SparseNMFDriver.cpp)
#add_library(dense_nmf OBJECT NMFDriver.cpp)
add_executable(plancnmf nmf_cmd.cpp)
add_library(nmflib
        nmf_lib.cpp bppnmf.cpp)
# add_executable(${DENSE_OR_SPARSE}_inmf inmf.cpp)
# target_compile_features(${DENSE_OR_SPARSE}_inmf PRIVATE cxx_std_17)
# add_compile_options(-O0)
find_library(GETOPT_LIB getopt)
target_include_directories(nmflib PRIVATE ${PLANC_COMMON} ${NNLS_SRC} ${OpenMP_CXX_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(plancnmf PRIVATE ${PLANC_COMMON} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(nmflib PUBLIC HighFive progressWrapper utils hw_detect detect_blas)
target_link_libraries(plancnmf PRIVATE nmflib $<$<BOOL:${GETOPT_LIB}>:${GETOPT_LIB}>)
generate_export_header(nmflib)
# target_link_libraries(${DENSE_OR_SPARSE}_inmf ${NMFLIB_LIBS})
install(TARGETS plancnmf nmflib
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )
