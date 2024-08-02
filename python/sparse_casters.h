//
// Created by andrew on 8/2/2024.
//

#ifndef SPARSE_CASTERS_H
#define SPARSE_CASTERS_H

/*
    nanobind/eigen/sparse.h: type casters for sparse Eigen matrices

 * This type alias constructs an ndarray of relevant dtype and dimension for armadillo SpMat and batch initializes
 * the CSC array with its values using make_caster<ndarray> * 3.
 * This code is derived from https://github.com/wjakob/nanobind/blob/b0136fe6ac1967cb2399456adc346a1af06a3b88/include/nanobind/eigen/sparse.h
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

#include <nanobind/ndarray.h>
#include "dense_casters.h"

#include <memory>
#include <type_traits>
#include <utility>

NAMESPACE_BEGIN(NB_NAMESPACE)

NAMESPACE_BEGIN(detail)

/// Detect Arma::SpMat
template <typename T, typename eT = typename T::elem_type> constexpr bool is_arma_sparse_matrix_v =
    is_arma_sparse_v<T>;


/// Caster for Arma::SpMat
template <typename T> struct type_caster<T, enable_if_t<is_arma_sparse_matrix_v<T> &&  is_ndarray_scalar_v<typename T::elem_type>>> {
    using eT = typename T::elem_type;
    using StorageIndex = arma::uword;

    using NDArray = ndarray<numpy, eT, shape<-1>>;
    using StorageIndexNDArray = ndarray<numpy, StorageIndex, shape<-1>>;

    using Caster = make_caster<NDArray>;
    using StorageIndexCaster = make_caster<StorageIndexNDArray>;

    NB_TYPE_CASTER(T, const_name("scipy.sparse.csc_matrix[")
                   + make_caster<eT>::Name + const_name("]"))

    Caster data_caster;
    StorageIndexCaster indices_caster, indptr_caster;

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        object obj = borrow(src);
        try {
            object matrix_type = module_::import_("scipy.sparse").attr("csc_matrix");
            if (!obj.type().is(matrix_type))
                obj = matrix_type(obj);
        } catch (const python_error &) {
            return false;
        }

        if (object data_o = obj.attr("data"); !data_caster.from_python(data_o, flags, cleanup))
            return false;
        NDArray& values = data_caster.value;

        if (object indices_o = obj.attr("indices"); !indices_caster.from_python(indices_o, flags, cleanup))
            return false;
        StorageIndexNDArray& inner_indices = indices_caster.value;

        if (object indptr_o = obj.attr("indptr"); !indptr_caster.from_python(indptr_o, flags, cleanup))
            return false;
        StorageIndexNDArray& outer_indices = indptr_caster.value;

        object shape_o = obj.attr("shape"), nnz_o = obj.attr("nnz");
        StorageIndex rows, cols, nnz;
        try {
            if (len(shape_o) != 2)
                return false;
            rows = cast<StorageIndex>(shape_o[0]);
            cols = cast<StorageIndex>(shape_o[1]);
            nnz = cast<StorageIndex>(nnz_o);
        } catch (const python_error &) {
            return false;
        }

        value = arma::SpMat<eT>(arma::uvec(inner_indices.data(), nnz), arma::uvec(outer_indices.data(), cols + 1), arma::Col<eT>(values.data(), nnz), rows, cols);

        return true;
    }

    static handle from_cpp(T &&v, rv_policy policy, cleanup_list *cleanup) noexcept {
        if (policy == rv_policy::automatic ||
            policy == rv_policy::automatic_reference)
            policy = rv_policy::move;

        return from_cpp((const T &) v, policy, cleanup);
    }

    static handle from_cpp(const T &v, rv_policy policy, cleanup_list *) noexcept {

        object matrix_type;
        try {
            matrix_type = module_::import_("scipy.sparse").attr("csc_matrix");
        } catch (python_error &e) {
            e.restore();
            return handle();
        }

        const StorageIndex rows = v.n_rows, cols = v.n_cols;
        const size_t data_shape[] = { (size_t) v.n_nonzero };
        const size_t outer_indices_shape[] = { (size_t) (cols + 1) };

        T *src = std::addressof(const_cast<T &>(v));
        object owner;
        if (policy == rv_policy::move) {
            src = new T(std::move(v));
            owner = capsule(src, [](void *p) noexcept { delete (T *) p; });
        }

        NDArray data(src->values, 1, data_shape, owner);
        StorageIndexNDArray outer_indices(src->row_indices, 1, outer_indices_shape, owner);
        StorageIndexNDArray inner_indices(src->colptrs(), 1, data_shape, owner);

        try {
            return matrix_type(nanobind::make_tuple(
                                   std::move(data), std::move(inner_indices), std::move(outer_indices)),
                               nanobind::make_tuple(rows, cols))
                .release();
        } catch (python_error &e) {
            e.restore();
            return handle();
        }
    }
};

NAMESPACE_END(detail)

NAMESPACE_END(NB_NAMESPACE)


#endif //SPARSE_CASTERS_H
