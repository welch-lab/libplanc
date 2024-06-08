//
// Created by andrew on 5/24/2024.
//

#ifndef PLANC_EMBEDDEDNMFDRIVER_H
#define PLANC_EMBEDDEDNMFDRIVER_H
#include "NMFDriver.hpp"

namespace planc {

    template <typename T>
    class EmbeddedNMFDriver : public NMFDriver<T> {
    protected:
        T Winit;
        T Hinit;

        void parseParams(const internalParams<T>& pc) {
            this->A = pc.getMAMat();
            this->Winit = pc.getMWInitMat();
            this->Hinit = pc.getMHInitMat();
            this->commonParams(pc);
        }

    private:
        void loadWHInit(arma::mat W, arma::mat H) override {
            if (!Winit.is_empty()) {
                W = this->Winit;
                if (W.n_rows != this->m_m || W.n_cols != this->m_k) {
                    std::throw_with_nested("Winit must be of size " +
                                           std::to_string(this->m_m) + " x " + std::to_string(this->m_k));
                }
            } else {
                W = arma::randu<arma::mat>(this->m_m, this->m_k);
            }
            if (!Hinit.is_empty()) {
                H = this->Hinit;
                if (H.n_rows != this->m_n || H.n_cols != this->m_k) {
                    std::throw_with_nested("Hinit must be of size " +
                                           std::to_string(this->m_n) + " x " + std::to_string(this->m_k));
                }
            } else {
                H = arma::randu<arma::mat>(this->m_n, this->m_k);
            }
        }

        void loadMat(double t2) override {
            this->m_m = this->A.n_rows;
            this->m_n = this->A.n_cols;
        }

        template<class NMFTYPE>
        void outRes(NMFTYPE nmfA) {}

    public:
        explicit EmbeddedNMFDriver<T>(internalParams<T> pc) : NMFDriver<T>(pc)
        {
            this->parseParams(pc);
        }
    };



}
#endif //PLANC_EMBEDDEDNMFDRIVER_H
