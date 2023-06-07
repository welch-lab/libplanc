#Copyright 2016 Ramakrishnan Kannan

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
if(CMAKE_BUILD_SPARSE)
  set(DENSE_OR_SPARSE sparse)
else()
  set(DENSE_OR_SPARSE dense)
endif()
add_executable(${DENSE_OR_SPARSE}_nmf nmf.cpp)
add_executable(${DENSE_OR_SPARSE}_inmf inmf.cpp)
target_compile_features(${DENSE_OR_SPARSE}_inmf PRIVATE cxx_std_17)
add_compile_options(-O0)

target_link_libraries(${DENSE_OR_SPARSE}_nmf ${NMFLIB_LIBS})
target_link_libraries(${DENSE_OR_SPARSE}_inmf ${NMFLIB_LIBS})
install(TARGETS ${DENSE_OR_SPARSE}_nmf ${DENSE_OR_SPARSE}_inmf
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )
