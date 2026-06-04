cmake_minimum_required(VERSION 3.24...4.0 FATAL_ERROR)
if (SKBUILD)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(X_VCPKG_APPLOCAL_DEPS_INSTALL ON)
endif ()
mark_as_advanced(CMAKE_OSX_DEPLOYMENT_TARGET)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# the settings below will be automatically configured by the rest of this script
project(PLANC C CXX)
set(LIBPLANC_MAJOR_VERSION 1)
set(LIBPLANC_MINOR_VERSION 0)
set(LIBPLANC_PATCH_VERSION 2)
set(LIBPLANC_VERSION "${LIBPLANC_MAJOR_VERSION}.${LIBPLANC_MINOR_VERSION}.${LIBPLANC_PATCH_VERSION}")
set(NMFLIB_MAJOR_VERSION 1)
set(NMFLIB_MINOR_VERSION 0)
set(NMFLIB_PATCH_VERSION 1)
set(NMFLIB_VERSION "${NMFLIB_MAJOR_VERSION}.${NMFLIB_MINOR_VERSION}.${NMFLIB_PATCH_VERSION}")
message("This is libplanc ${LIBPLANC_VERSION} containing component nmflib ${NMFLIB_VERSION}")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CheckIncludeFile)
include(CheckCXXCompilerFlag)
include(FetchContent)
include(ExternalProject)
include(CheckSymbolExists)
include(CheckCXXSymbolExists)
include(CheckCXXSourceRuns)
include(GenerateExportHeader)
include(DetectOMP)
include(DetectBLAS)
include(DetectHighFive)
include(RcppDefines)
include(PythonDefines)
include(CompilerFlags)

# Adhere to GNU filesystem layout conventions
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_C_VISIBILITY_PRESET¶ ON)
set(CMAKE_CXX_VISIBILITY_PRESET¶ ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
separate_arguments(PASSTHROUGH_CONFIGURE_ARGS)

set(GENERAL_COMPILE_OPTIONS "")
set(GENERAL_LINK_OPTIONS "")
set(GENERAL_INCLUDE_DIRS "")
set(GENERAL_DEFINES "")

add_library(nmflib
        nmf/nmf_lib.cpp nmf/bppnmf.cpp nnls/nnls_lib.cpp
        common/parseutils.cpp common/data.cpp common/utils.cpp)
target_sources(nmflib PUBLIC FILE_SET common TYPE HEADERS BASE_DIRS common
        FILES common/config.h common/data.hpp common/inmf.hpp
              common/nmf.hpp common/parsecommandline.h common/parsecommandline.hpp
              common/parseutils.hpp common/plancopts.h common/utils.h common/utils.hpp)
target_sources(nmflib PUBLIC FILE_SET nnls TYPE HEADERS BASE_DIRS nnls
        FILES nnls/nnls.hpp nnls/nnls_lib.hpp nnls/bppnnls.hpp nnls/SortBooleanMatrix.hpp)
target_sources(nmflib PUBLIC FILE_SET nmf TYPE HEADERS BASE_DIRS nmf
        FILES nmf/nmf_lib.hpp nmf/bppinmf.hpp nmf/aoadmm.hpp nmf/bppnmf.hpp nmf/EmbeddedNMFDriver.hpp
        nmf/gnsym.hpp nmf/hals.hpp nmf/mu.hpp nmf/NMFDriver.hpp nmf/onlineinmf.hpp nmf/uinmf.hpp)
set(NNLS_SRC ${PROJECT_SOURCE_DIR}/nnls)
set(NMF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/nmf)

if (BUILD_EXECUTABLE)
    add_executable(plancnmf nmf/nmf_cmd.cpp)
    find_package(argparse REQUIRED)
    target_link_libraries(plancnmf PRIVATE nmflib argparse::argparse $<$<BOOL:${MINGW}>:detect_blas>)
    install(TARGETS plancnmf RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif ()

DetectOMP()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT MSVC)
    execute_process(COMMAND clang -v ERROR_VARIABLE CLANGID)
    if(CLANGID)
    string(REGEX MATCH "\\(.*\\)" CLANGPARSE ${CLANGID})
    if (CLANGPARSE MATCHES "\\(CLANG\\: AOCC.*")
        set(BLA_VENDOR AOCL_mt)
        list(APPEND GENERAL_COMPILE_OPTIONS -zopt -O3 -flto -fstrip-mining -freciprocal-math -finline-recursion=3
                -fremap-arrays -fnt-store=aggressive -mrecip=all -optimize-strided-mem-cost
                -fenable-aggressive-gather -favoid-fpe-causing-opt)
    endif ()
  endif ()
endif ()

DetectBLAS()
DetectHighFive()

if (BUILD_RCPP)
    RcppDefines()
elseif (SKBUILD)
    PythonDefines()
else ()
    find_package(Armadillo CONFIG REQUIRED)
    find_package(indicators CONFIG REQUIRED)
    target_sources(nmflib PRIVATE ${PROJECT_SOURCE_DIR}/common/progressWrapper.cpp)
    target_sources(nmflib PUBLIC FILE_SET progresswrapper TYPE HEADERS FILES ${PROJECT_SOURCE_DIR}/common/progressWrapper.h)
    target_include_directories(nmflib PUBLIC ${ARMADILLO_INCLUDE_DIRS})
    if(BUILD_EXECUTABLE)
        target_link_libraries(nmflib PUBLIC indicators::indicators)
    else()
        target_compile_definitions(nmflib PRIVATE NO_INDICATORS)
    endif()
endif ()

CompilerFlags()

if (NOT USING_OPENBLAS_H AND NOT USING_MKL)
    if (CBLAS_H_DIR)
        list(PREPEND GENERAL_INCLUDE_DIRS ${CBLAS_H_DIR})
    else ()
        set(CBLAS_H_DIR CACHE PATH "Directory containing cblas.h")
        check_include_file("cblas.h" CBLAS_HEADER)
        if (NOT CBLAS_HEADER)
            check_include_file("flexiblas/cblas.h" FLEXIBLAS_HEADER)
            if (FLEXIBLAS_HEADER)
                list(APPEND GENERAL_DEFINES "HAVE_FLEXIBLAS_CBLAS_H")
            else ()
                #message(FATAL_ERROR "no cblas header, please set include directory with CBLAS_H_DIR variable")
            endif ()
        endif ()
    endif ()
endif ()


add_library(hw_detect OBJECT ${PROJECT_SOURCE_DIR}/common/hw_detect.c)

find_package(HWLOC)

HandleHWLOC()

add_library(detect_blas OBJECT ${PROJECT_SOURCE_DIR}/common/detect_blas.c)
check_include_file(dlfcn.h HAS_DLFCN)
if (WIN32 AND NOT HAS_DLFCN)
    find_package(dlfcn-win32 CONFIG REQUIRED)
    set(DLFCN_LINK 1)
    target_link_libraries(detect_blas PUBLIC "$<$<BOOL:${DLFCN_LINK}>:dlfcn-win32::dl>")
endif ()



# add_executable(${DENSE_OR_SPARSE}_inmf inmf.cpp)
target_include_directories(nmflib PUBLIC ${OpenMP_CXX_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(nmflib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nmflib PUBLIC hw_detect detect_blas)
generate_export_header(nmflib)
if(BUILD_RCPP)
    if(WIN32)
        add_custom_command(TARGET nmflib POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E remove ${HighFive_SOURCE_DIR}/../highfive-subbuild/Makefile
                VERBATIM)
        add_custom_command(TARGET nmflib POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/../CMakeFiles/${CMAKE_VERSION}/CompilerIdC/CMakeCCompilerId.c
                VERBATIM)
        add_custom_command(TARGET nmflib POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/../CMakeFiles/${CMAKE_VERSION}/CompilerIdCXX/CMakeCXXCompilerId.cpp
                VERBATIM)
        add_custom_command(TARGET nmflib POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/Makefile
                VERBATIM)
        add_custom_command(TARGET nmflib POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/../Makefile
                VERBATIM)
    endif()
endif()
target_compile_definitions(nmflib PUBLIC ${GENERAL_DEFINES})
target_compile_options(nmflib PUBLIC ${GENERAL_COMPILE_OPTIONS})
target_link_options(nmflib PUBLIC ${GENERAL_LINK_OPTIONS})
target_include_directories(nmflib PUBLIC ${GENERAL_INCLUDE_DIRECTORIES})
target_link_libraries(nmflib PUBLIC "$<$<BOOL:${USING_MKL}>:MKL::MKL>"
        "$<$<BOOL:${USING_OPENBLAS}>:${OpenBLAS_LIBRARIES}>"
        "$<$<BOOL:${USING_BLAS}>:BLAS::BLAS>" "$<$<BOOL:${USING_BLAS}>:LAPACK::LAPACK>"
        "$<$<BOOL:${HWLOC_FRAMEWORKS}>:${HWLOC_FRAMEWORKS}>"
        "$<$<BOOL:${DLFCN_LINK}>:dlfcn-win32::dl>"
        "$<IF:$<BOOL:HDF5_TARGET>,HDF5::HDF5,${HDF5_LIBRARIES}>" ${R_LDFLAGS} ${RCPP_FLIBS})

if (BUILD_RCPP)
    get_directory_property(CMAKE_DEFINES COMPILE_DEFINITIONS)
    list(TRANSFORM CMAKE_DEFINES PREPEND "-D")
    get_target_property(CMAKE_TARGET_DEFINES nmflib COMPILE_DEFINITIONS)
    list(TRANSFORM CMAKE_TARGET_DEFINES PREPEND "-D")
    list(APPEND CMAKE_DEFINES CMAKE_TARGET_DEFINES)
    if(CBLAS_H_DIR)
    list(PREPEND CMAKE_DEFINES -I"${CBLAS_H_DIR}")
    endif()
    list(JOIN CMAKE_DEFINES " " CPPFLAGS)
    if (BUILD_SHARED_LIBS)
    if (APPLE)
    if (R_SHLIB_EXT MATCHES ".so")
    set_target_properties(nmflib PROPERTIES SUFFIX ${R_SHLIB_EXT})
    endif ()
    set(EXTRA_LD_ARGS "-rpath @loader_path/../libs")
    elseif (NOT WIN32)
    set(EXTRA_LD_ARGS  "'-Wl,-rpath,$$ORIGIN/../libs,--enable-new-dtags'")
    else ()
    set(EXTRA_LD_ARGS "")
    endif ()
    else ()
        if(APPLE AND ((CMAKE_HOST_SYSTEM_PROCESSOR MATCHES arm64) OR NOT OpenMP_CXX_FOUND))
            set(BLA_VENDOR "Apple")
            find_package(BLAS REQUIRED)
            cmake_path(REMOVE_FILENAME BLAS_LIBRARIES OUTPUT_VARIABLE BLAS_LIBRARIES_FRAMEWORK)
            set(BLAS_LIBRARIES_EXTRACT "-F${BLAS_LIBRARIES_FRAMEWORK} -framework Accelerate")
            unset(BLAS_LIBRARIES)
            unset(LAPACK_LIBRARIES)
        endif()
        set(EXTRA_LD_ARGS ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${pkgcfg_lib_HWLOC_hwloc} ${HWLOC_hwloc_LIBRARY} "$<$<BOOL:${HWLOC_FRAMEWORKS}>:${HWLOC_FRAMEWORKS}>" ${HDF5_LIBRARIES} ${R_LDFLAGS} ${RCPP_FLIBS} ${BLAS_LIBRARIES_EXTRACT})
    list(JOIN EXTRA_LD_ARGS " " EXTRA_LD_ARGS)
    endif ()
    install(TARGETS nmflib RUNTIME DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../" ARCHIVE DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../" LIBRARY DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../")
    configure_file(../Makevars.in ../Makevars.gen @ONLY NEWLINE_STYLE LF)
    file(GENERATE OUTPUT ../Makevars INPUT ../Makevars.gen NEWLINE_STYLE LF)
elseif (SKBUILD)
    nanobind_add_module(_lib STABLE_ABI FREE_THREADED NB_STATIC ${CMAKE_CURRENT_SOURCE_DIR}/python/pybind_interface.cpp)
    target_sources(_lib PRIVATE FILE_SET pyheaders TYPE HEADERS BASE_DIRS python FILES python/pointer_helpers.hpp python/progressWrapper.h python/dense_casters.h python/sparse_casters.h)
    target_compile_definitions(_lib PUBLIC ${GENERAL_DEFINES})
    target_compile_options(_lib PUBLIC ${GENERAL_COMPILE_OPTIONS})
    target_link_options(_lib PUBLIC ${GENERAL_LINK_OPTIONS})
    target_include_directories(_lib PUBLIC ${GENERAL_INCLUDE_DIRECTORIES})
    if(SKBUILD_FREETHREAD)
        target_compile_definitions(_lib PRIVATE Py_GIL_DISABLED)
    endif()
    if(CMAKE_BUILD_TYPE MATCHES "Release" OR CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
        target_compile_definitions(_lib PRIVATE NDEBUG)
    endif()
    find_library(NMFLIB_LIB nmflib)
    if(NMFLIB_LIB)
        find_file(NMFLIB_INC nmflib_export.h HINTS $ENV{LIBRARY_INC} $ENV{PREFIX}/include PATH_SUFFIXES libplanc/nmf REQUIRED)
        get_target_property(NMFLIB_INC nmflib INCLUDE_DIRECTORIES)
        target_include_directories(_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/nmf ${NMFLIB_INC})
        target_link_libraries(_lib PRIVATE ${NMFLIB_LIB})
    else()
        target_link_libraries(_lib PRIVATE nmflib)
        install(TARGETS nmflib RUNTIME DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../" ARCHIVE DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../" LIBRARY DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../")
    endif()

    install(TARGETS _lib LIBRARY DESTINATION .)
else()
    install(TARGETS nmflib
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    set(nmf_headers nmf/nmf_lib.hpp common/data.hpp common/config.h common/detect_blas.h ${CMAKE_CURRENT_BINARY_DIR}/nmf/nmflib_export.h)
    set(nnls_headers nnls/nnls_lib.hpp)
    install(FILES ${nmf_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libplanc/nmf)
    install(FILES ${nnls_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libplanc/nnls)
endif()
