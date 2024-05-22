//
// Created by andrew on 5/22/2024.
//

%{
#include <../nmf/nmf_lib.hpp>
%}
%include "nmflib_export.h"
%include <../common/plancopts.h>
%include <../nmf/nmf_lib.hpp>
%template(runDenseNMF) planc::nmflib::runNMF<arma::mat>;
%template(runSparseNMF) planc::nmflib::runNMF<arma::sp_mat>;

