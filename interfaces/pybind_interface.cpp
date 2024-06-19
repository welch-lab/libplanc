//
// Created by andrew on 6/19/2024.
//

#include <pybind11/pybind11.h>
#include <nmf_lib.hpp>

namespace py = pybind11;

PYBIND11_MODULE(pyplanc, m) {
    m.doc() = "A python wrapper for planc-nmflib";
}