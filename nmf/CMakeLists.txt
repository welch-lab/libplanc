#Copyright 2016 Ramakrishnan Kannan

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
if(SKBUILD)
    nanobind_add_module(pyplanc STABLE_ABI NB_STATIC ../python/pybind_interface.cpp)
    #target_include_directories(pyplanc PRIVATE "python")
    target_link_libraries(pyplanc PRIVATE nmflib)
    install(TARGETS pyplanc LIBRARY DESTINATION .)
    install(TARGETS nmflib RUNTIME DESTINATION . ARCHIVE DESTINATION . LIBRARY DESTINATION .)
endif()
# target_link_libraries(${DENSE_OR_SPARSE}_inmf ${NMFLIB_LIBS})
#install(TARGETS  plancnmf nmflib
#        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )
