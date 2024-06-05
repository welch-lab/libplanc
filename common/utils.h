#pragma once
#include "config.h"
#include <unordered_map>

/* Copyright 2016 Ramakrishnan Kannan */
// utility functions

// #ifndef _VERBOSE
// #define _VERBOSE 1
// #endif

enum algotype { MU, HALS, ANLSBPP, NAIVEANLSBPP, AOADMM,
        NESTEROV, CPALS, GNSYM, R2, PGD, PGNCG };
extern std::unordered_map<std::string, algotype> algomap;

enum normtype { NONE, L2NORM, MAXNORM };
extern std::unordered_map<std::string, normtype> normmap;


enum helptype { NMF, DISTNMF, NTF, DISTNTF, JOINTNMF, DISTJOINTNMF, HIERNMF };

#include <armadillo>
#include <cmath>
#include <iostream>
#include <vector>

// using namespace std;

#ifndef ERR
#ifdef USING_R
#define ERR Rcpp::Rcerr
#else
#define ERR std::cerr
#endif
#endif

#ifndef WARN
#ifdef USING_R
#define WARN Rcpp::Rcerr
#else
#define WARN std::cerr
#endif
#endif

#ifndef INFO
#ifdef USING_R
#define INFO Rcpp::Rcout
#else
#define INFO std::cout
#endif
#endif

#ifndef OUTPUT
#ifdef USING_R
#define OUTPUT Rcpp::Rcout
#else
#define OUTPUT std::cout
#endif
#endif

constexpr auto EPSILON_1EMINUS16 = 0.00000000000000001;
constexpr auto EPSILON_1EMINUS8=0.00000001;
constexpr auto EPSILON = 0.000001;
constexpr auto EPSILON_1EMINUS12 = 1e-12;
constexpr auto NUMBEROF_DECIMAL_PLACES = 12;
constexpr auto RAND_SEED = 100;
constexpr auto RAND_SEED_SPARSE = 100;
constexpr auto WTRUE_SEED=1196089;
constexpr auto HTRUE_SEED=1230587;


#define PRINTMATINFO(A) "::" #A "::" << (A).n_rows << "x" << (A).n_cols

#define PRINTMAT(A) PRINTMATINFO((A)) << std::endl << (A)

typedef std::vector<int> STDVEC;
#ifndef ULONG
typedef uint64_t ULONG;
#endif

void absmat(const arma::fmat *X);




template <typename FVT>
inline void fillVector(const FVT value, std::vector<FVT> *a) {
  for (unsigned int ii = 0; ii < a->size(); ii++) {
    (*a)[ii] = value;
  }
}
