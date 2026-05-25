macro(PythonDefines)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module
            OPTIONAL_COMPONENTS Development.SABIModule)
    execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)
    set(HEADER_ONLY CACHE INTERNAL BOOL ON)
    find_package(Armadillo)
    if (NOT Armadillo_FOUND)
        FetchContent_Declare(
                Armadillo
                URL https://cytranet-dal.dl.sourceforge.net/project/arma/armadillo-15.2.6.tar.xz
                URL_HASH SHA256=97cb8ef708541f632e861d005a462dd0367240f81ff96f8e63ebbdd75c8ce55f
        )
        FetchContent_MakeAvailable(Armadillo)
    endif ()
    find_package(indicators CONFIG REQUIRED)
    target_sources(nmflib PRIVATE ${PROJECT_SOURCE_DIR}/common/progressWrapper.cpp)
    target_include_directories(nmflib PUBLIC "${armadillo_BINARY_DIR}/tmp/include")
    target_link_libraries(nmflib PUBLIC indicators::indicators)
endmacro()