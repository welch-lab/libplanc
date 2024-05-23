//
// Created by andrew on 5/22/2024.
//

%{
#include <../nmf/nmf_lib.hpp>
    using namespace std;
%}
%include "nmflib_export.h"
%include <../common/plancopts.h>
%include <../nmf/nmf_lib.hpp>
%include <std/std_array.i>
%template(runDenseNMF) planc::nmf<arma::mat>;
%template(runSparseNMF) planc::nmf<arma::sp_mat>;

