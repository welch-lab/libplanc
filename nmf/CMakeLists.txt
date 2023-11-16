#Copyright 2016 Ramakrishnan Kannan

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)


include(${PROJECT_SOURCE_DIR}/common/cmake_aux/NMFLIB_FindARMA.cmake)

# the settings below will be automatically configured by the rest of this script

include_directories(${NMFLIB_COMMON} ${NNLS_SRC}
  ${ARMADILLO_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${MKL_INCLUDE_DIR}
  ${OPENBLAS_INCLUDE_DIR}
  ${SUPERLU_INCLUDES})
add_compile_definitions(HAVE_OPENBLAS_CBLAS_H) #fixme
add_library(sparse_nmf OBJECT SparseNMFDriver.cpp)
add_library(dense_nmf OBJECT NMFDriver.cpp)
add_executable(plancnmf nmf_cmd.cpp)

# add_executable(${DENSE_OR_SPARSE}_inmf inmf.cpp)
# target_compile_features(${DENSE_OR_SPARSE}_inmf PRIVATE cxx_std_17)
# add_compile_options(-O0)
find_library(GETOPT_LIB getopt)
target_link_libraries(plancnmf PUBLIC ${NMFLIB_LIBS} PRIVATE $<TARGET_OBJECTS:sparse_nmf> $<TARGET_OBJECTS:dense_nmf> $<$<BOOL:${GETOPT_LIB}>:${GETOPT_LIB}>)
# target_link_libraries(${DENSE_OR_SPARSE}_inmf ${NMFLIB_LIBS})
install(TARGETS plancnmf
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )
