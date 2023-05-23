#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <memory>

namespace planc {

    template <class T>
    class INMF {
    private:

        arma::uword m, k;
        std::vector<arma::uword> ncol_E;             // vector of n_i
        std::vector<std::unique_ptr<T>> Ei;          // each of size mxn_i
        std::vector<std::unique_ptr<arma::mat>> Hi;  // each of size n_ixk
        std::vector<std::unique_ptr<arma::mat>> Vi;  // each of size mxk
        std::unique_ptr<arma::mat> W;                // mxk
        const double lambda, sqrtLambda;
        std::vector<arma::mat> C_solveH;//(2*m, k);
        const T B_solveH;//(2*m, n);
        std::vector<arma::mat> C_solveV;//(2*n, k);
        T B_solveV;//(2*n, m);

        // arma::mat HV;//(this->n, this->m); /// H(nxk) * Vt(kxm);
#ifdef CMAKE_BUILD_SPARSE
        arma::sp_mat B_solveW_i; //(this->n, this->m);
#else
        arma::mat B_solveW_i;   //(this->n, this->m); /// At(nxm) - HV(nxm);
#endif
        void updateC_solveH(int i) {
            arma::mat* Wptr = get(W);
            arma::mat* Vptr = get(Vi[i]);
            this->C_solveH[i] = arma::join_cols(&Wptr + &Vptr, sqrtLambda * &Vptr);
        }

        void updateC_solveV(int i) {
            arma::mat* Hptr = get(Hi[i]);
            this->C_solveV[i] = arma::join_cols(&Hptr, sqrtLambda * &Hptr);
        }

        T makeB_solveH(int i) { // call in constructor
            T* Eptr = get(Ei[i]);
            arma::sp_mat B_H(2 * this->m, this->ncol_E[i]);
            // Copy the values from E to the top half of B_H
            B_H.rows(0, this->m) = &Eptr;
            return B_H;
        }

        void updateB_solveV(int i) {
            T* Eptr = get(Ei[i]);
            arma::mat* Wptr = get(W);
            arma::mat* Hptr = get(Hi[i]);
            this->B_solveV = arma::zeros(2 * this->ncol_E[i], this->m);
            this->B_solveV.rows(0, this->ncol_E[i] - 1) = &Eptr.t() - &Hptr * &Wptr.t();
        }
    public:
        INMF(std::vector<std::unique_ptr<T>> Ei,
             arma::uword k) {
            this->Ei = Ei
            this->k = k;
            this->m = 0; // TODO
            //this->ncol_E = ; // TODO
            //TODO implement
        }
    };

}
