cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
include(${PROJECT_SOURCE_DIR}/common/cmake_aux/NMFLIB_FindARMA.cmake)

# the settings below will be automatically configured by the rest of this script
include_directories(${NMFLIB_COMMON} ${NNLS_SRC}
  ${ARMADILLO_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${MKL_INCLUDE_DIR}
  ${OPENBLAS_INCLUDE_DIR}
  ${SUPERLU_INCLUDES})

include(FetchContent)
FetchContent_Declare(
        fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
        GIT_TAG main
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)
add_executable(planc_bench planc_bench.cpp)

target_link_libraries(planc_bench ${NMFLIB_LIBS} fast_matrix_market::fast_matrix_market)

install(TARGETS planc_bench
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )
