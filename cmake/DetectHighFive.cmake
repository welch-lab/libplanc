macro(DetectHighFive)
    set(HIGHFIVE_EXAMPLES OFF CACHE INTERNAL BOOL)
    set(HIGHFIVE_BUILD_DOCS OFF CACHE INTERNAL BOOL)
    set(HIGHFIVE_USE_BOOST OFF CACHE INTERNAL BOOL)
    set(HIGHFIVE_UNIT_TESTS OFF CACHE INTERNAL BOOL)
    set(HIGHFIVE_HAS_CONCEPTS OFF CACHE INTERNAL BOOL)
    if ((WIN32 OR APPLE) AND NOT DEFINED ENV{CONDA_BUILD})
        set(HDF5_USE_STATIC_LIBRARIES ON CACHE INTERNAL BOOL)
        set(HIGHFIVE_STATIC_HDF5 ON CACHE INTERNAL BOOL)
    endif ()
    set(HIGHFIVE_FIND_HDF5 OFF)
    FetchContent_Declare(
            HighFive
            URL https://github.com/highfive-devs/HighFive/archive/refs/tags/v3.3.0.tar.gz
            URL_HASH SHA256=325cfbcf0c0296a6dd26f3b088801b7ebb8d6f109c0565c11d2d8c4af3253bff
    )
    if (BUILD_RCPP)
        find_r_module(HighFive)
        if (R_HIGHFIVE)
            set(HighFive_SOURCE_DIR "${R_HIGHFIVE}")
        else ()
            message(WARNING "missing CRAN HighFive")
            FetchContent_MakeAvailable(HighFive)
        endif ()
    else ()
        FetchContent_MakeAvailable(HighFive)
    endif ()
    find_package(HDF5 REQUIRED COMPONENTS C)
    set(HDF5_TARGET 1)
    set(CMAKE_REQUIRED_LIBRARIES ${HDF5_LIBRARIES})
    set(HighFive_Includes "${HighFive_SOURCE_DIR}/include" "${HDF5_INCLUDE_DIRS}")
    set(CMAKE_REQUIRED_INCLUDES "${HighFive_Includes}")
    set(test_hdf5_link "
#include <highfive/highfive.hpp>
int main() {
return 0;
}
")
    check_cxx_source_runs("${test_hdf5_link}" HDF5_LINKS)
    unset(CMAKE_REQUIRED_LIBRARIES)
    unset(CMAKE_REQUIRED_INCLUDES)
    if (NOT HDF5_LINKS)
        set(HDF5_TARGET 0)
        # from https://gitlab.kitware.com/cmake/cmake/-/issues/18872#note_1090297
        # Since there is no compiler line to help us, add the additional required
        # libraries manually.
        set(_additional_libs sz z dl m)
        foreach (_additional_lib IN LISTS _additional_libs)
            # If both static and shared are available, prefer static to avoid libdl
            # annoyances ("Using 'dlopen' in statically linked applications requires
            # at runtime the shared libraries from the glibc version used for
            # linking")
            if (HDF5_USE_STATIC_LIBRARIES)
                set(_libnames ${_additional_lib} lib${_additional_lib}.a)
            else ()
                set(_libnames ${_additional_lib})
            endif (HDF5_USE_STATIC_LIBRARIES)
            set(_libvar "LIB_${_additional_lib}")
            find_library(${_libvar}
                    NAMES ${_libnames}
                    HINTS ${HDF5_ROOT}
                    PATH_SUFFIXES lib Lib
                    REQUIRED)
            if (HDF5_FIND_DEBUG)
                message(STATUS "${_additional_lib} (${_libvar}): ${${_libvar}}")
            endif (HDF5_FIND_DEBUG)
            if (${${_libvar}} STREQUAL "${_libvar}-NOTFOUND")
                message(FATAL_ERROR "${_additional_lib} not found, required for HDF5 install")
            endif ()
            list(APPEND HDF5_LIBRARIES ${${_libvar}})
        endforeach ()
        list(REMOVE_DUPLICATES HDF5_LIBRARIES)
    endif ()
    target_include_directories(nmflib PUBLIC ${HighFive_Includes})
endmacro()