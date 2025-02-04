//
// Created by andrew on 7/26/2024.
//

#ifndef CASTERS_H
#define CASTERS_H
#include <config.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;


NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)


/* This type alias constructs an ndarray of relevant dtype and dimension for armadillo vec/mat and populates auxillary
 * memory with its values using make_caster<ndarray>. Should probably check continuity and if the size is fixed.
 * This code is derived from https://github.com/wjakob/nanobind/blob/b0136fe6ac1967cb2399456adc346a1af06a3b88/include/nanobind/eigen/dense.h
 * Copyright (c) 2023 Wenzel Jakob.  <wenzel.jakob@epfl.ch>, All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
*/

/// Arma vec
template <typename T, typename eT = typename T::elem_type> constexpr bool is_arma_v = T::is_row || T::is_col || T::is_xvec;

/// Detects Dense
template <typename T, typename eT = typename T::elem_type> constexpr bool is_arma_dense_v = std::is_base_of_v<arma::Mat<eT>, T>;

/// Detect Sparse
template <typename T, typename eT = typename T::elem_type> constexpr bool is_arma_sparse_v = std::is_base_of_v<arma::SpMat<eT>, T>;

/// Determine the number of dimensions of the given Eigen type
template <typename T>
constexpr int ndim_v = bool(is_arma_v<T>) ? 1 : 2;

template <typename T, typename eT = typename T::elem_type, const int Dim = ndim_v<T>>
using array_for_arma_t = ndarray<
    eT,
    numpy,
    ndim<Dim>,
    f_contig>;


template <typename T>
struct type_caster<T, enable_if_t<is_arma_dense_v<T> &&
                                  is_ndarray_scalar_v<typename T::elem_type>>> {
    using eT = typename T::elem_type;
    using NDArray = array_for_arma_t<T>;
    using NDArrayCaster = make_caster<NDArray>;

    NB_TYPE_CASTER(T, NDArrayCaster::Name)

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        // We're in any case making a copy, so non-writable inputs area also okay
        using NDArrayConst = array_for_arma_t<T, const typename T::elem_type>;
        make_caster<NDArrayConst> caster;
        if (!caster.from_python(src, flags, cleanup))
            return false;

        const NDArrayConst &array = caster.value;
        if constexpr (ndim_v<T> == 1)
            value.set_size(array.shape(0));
        else
            value.set_size(array.shape(0), array.shape(1));

        // The layout is contiguous & compatible thanks to array_for_arma_t<T>
        memcpy(value.memptr(), array.data(), array.size() * sizeof(eT));

        return true;
    }

    static handle from_cpp(eT &&v, rv_policy policy, cleanup_list *cleanup) noexcept {
        if (policy == rv_policy::automatic ||
            policy == rv_policy::automatic_reference)
            policy = rv_policy::move;

        return from_cpp((const T &) v, policy, cleanup);
    }

    static handle from_cpp(const T &v, rv_policy policy, cleanup_list *cleanup) noexcept {
        size_t shape[ndim_v<T>];
        int64_t strides[ndim_v<T>];

        if constexpr (ndim_v<T> == 1) {
            shape[0] = v.size();
            strides[0] = 1;
        } else {
            shape[0] = v.n_rows;
            shape[1] = v.n_cols;
            strides[0] = 1;
            strides[1] = v.n_rows;
        }

        void *ptr = (void *) v.memptr();

        switch (policy) {
            case rv_policy::automatic:
                policy = rv_policy::copy;
                break;

            case rv_policy::automatic_reference:
                policy = rv_policy::reference;
                break;

            default: // leave policy unchanged
                break;
        }

        object owner;
        if (policy == rv_policy::move) {
            T *temp = new T(std::move(v));
            owner = capsule(temp, [](void *p) noexcept { delete (T *) p; });
            ptr = temp->memptr();
            policy = rv_policy::reference;
        } else if (policy == rv_policy::reference_internal && cleanup->self()) {
            owner = borrow(cleanup->self());
            policy = rv_policy::reference;
        }

        object o = steal(NDArrayCaster::from_cpp(
            NDArray(ptr, ndim_v<T>, shape, owner, strides),
            policy, cleanup));

        return o.release();
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

#endif //CASTERS_H
