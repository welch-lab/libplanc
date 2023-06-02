#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <memory>
#include "utils.hpp"

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
        double lambda, sqrtLambda;
        std::vector<arma::mat> C_solveH;//(2*m, k);
        T B_solveH;//(2*m, n);
        std::vector<arma::mat> C_solveV;//(2*n, k);
        T B_solveV;//(2*n, m);

        // arma::mat HV;//(this->n, this->m); /// H(nxk) * Vt(kxm);
#ifdef CMAKE_BUILD_SPARSE
        arma::sp_mat B_solveW_i; //(this->n, this->m);
#else
        arma::mat B_solveW_i;   //(this->n, this->m); /// At(nxm) - HV(nxm);
#endif
        void updateC_solveH(int i) {
            arma::mat* Wptr = W.get();
            arma::mat* Vptr = Vi[i].get();
            this->C_solveH[i] = arma::join_cols(*Wptr + *Vptr, sqrtLambda * *Vptr);
        }

        void updateC_solveV(int i) {
            arma::mat* Hptr = Hi[i].get();
            this->C_solveV[i] = arma::join_cols(*Hptr, sqrtLambda * *Hptr);
        }

        T makeB_solveH(int i) { // call in constructor
            T* Eptr = Ei[i].get();
            T B_H(2 * this->m, this->ncol_E[i]);
            // Copy the values from E to the top half of B_H
            B_H.rows(0, this->m) = *Eptr;
            return B_H;
        }

        void updateB_solveV(int i) {
            T* Eptr = Ei[i].get();
            arma::mat* Wptr = W.get();
            arma::mat* Hptr = Hi[i].get();
            this->B_solveV = arma::zeros(2 * this->ncol_E[i], this->m);
            this->B_solveV.rows(0, this->ncol_E[i] - 1) = Eptr->t() - *Hptr * Wptr->t();
        }
    public:
        INMF(std::vector<std::unique_ptr<T>> &Ei,
             arma::uword k, double lambda) {
            Ei.swap(this->Ei);
            this->k = k;
            this->m = Ei[0]->n_rows;
            for (typename std::vector<std::unique_ptr<T>>::iterator it = Ei.begin(); it != Ei.end(); ++it)
            {
                arma::mat* E = it->get();
                this->ncol_E.push_back(E->n_cols);
            };
            this->lambda = lambda;
            this->sqrtLambda = sqrt(lambda); //TODO
            //TODO implement common tasks i.e. norm, reg, etc
        }
    };

}
