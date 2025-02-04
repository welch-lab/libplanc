import math
from pickle import TRUE
from random import sample
from typing import NamedTuple
import pytest
import numpy as np
import numpy.typing
import pyplanc
import scipy.sparse

class nmfparams(NamedTuple):
    m: int
    n: int

@pytest.fixture(scope="class")
def nmf_params() -> nmfparams:
    m = int(100)
    n = int(300)
    return nmfparams(m, n)

@pytest.fixture(scope="class")
def nmf_dense_mat(nmf_params: nmfparams) -> numpy.typing.NDArray[np.float64]:
    rs = np.random.default_rng(1)
    mat = np.empty((nmf_params[0], nmf_params[1]), np.float64, order="F")
    rs.random((nmf_params[0], nmf_params[1]), np.float64, mat)
    return mat

@pytest.fixture(scope="module", params=["anlsbpp", "admm", "hals", "mu"])
def algoarg(request):
    param = request.param
    yield param


class TestNMFDense:
    @pytest.fixture(scope="class", autouse=True)
    def run_nmf(self, nmf_dense_mat: numpy.typing.NDArray[np.float64], algoarg) -> pyplanc.nmfOutput:
        res = pyplanc.nmf(x=nmf_dense_mat, k=10, niter=100, algo=algoarg)
        return res

    def test_nmf_prefill(self, nmf_dense_mat, run_nmf, algoarg):
        newres = pyplanc.nmf(x=nmf_dense_mat, k=10, niter=100, algo=algoarg, Winit=run_nmf.W, Hinit=run_nmf.H)
        assert 1850 <= newres.objErr <= 2650

    def test_nmf_type(self, run_nmf: pyplanc.nmfOutput):
        assert isinstance(run_nmf, pyplanc.nmfOutput)

    def test_nmf_shape(self, run_nmf: pyplanc.nmfOutput, nmf_params: nmfparams):
        assert run_nmf.W.shape == (nmf_params.m, 10)
        assert run_nmf.H.shape == (nmf_params.n, 10)

    def test_nmf_objerr(self, run_nmf):
        assert 1850 <= run_nmf.objErr <= 2650


class TestNMFDenseFail:
    @pytest.fixture(scope="class", params=["hello"])
    def algoarg(self, request):
        param = request.param
        yield param

    def test_bad_dense_algo_arg(self, nmf_dense_mat: numpy.typing.NDArray[np.float64], algoarg) -> pyplanc.nmfOutput:
        with pytest.raises(RuntimeError) as e:
            pyplanc.nmf(x=nmf_dense_mat, k=10, niter=100, algo=algoarg)
        assert 'Please choose `algo` from' in str(e.value)

@pytest.fixture(params=["mat", "array"])
def outtype(request):
    yield request.param

@pytest.fixture()
def nmf_sparse_mat(nmf_dense_mat, outtype):
    rs = np.random.default_rng(1)
    sparsity = .9
    matlen = nmf_dense_mat.size
    regenerate = True
    matsp = None
    while regenerate:
        zeroidx = rs.choice(
            matlen, round(sparsity * matlen), replace=False, shuffle=False
        )
        matsp = nmf_dense_mat.copy()
        matsp.flat[zeroidx] = 0
        if outtype == "mat":
            matsp = scipy.sparse.csc_matrix(matsp)
        elif outtype == "array":
            matsp = scipy.sparse.csc_array(matsp)
        if np.sum(matsp.sum(axis=0) == 0) == 0 and np.sum(matsp.sum(axis=1) == 0) == 0:
            regenerate = False
    return matsp
''
class TestNMFSparse(TestNMFDense):
    @pytest.fixture(autouse=True)
    def run_nmf(self, nmf_sparse_mat, algoarg) -> pyplanc.nmfOutput:
        res = pyplanc.nmf(x=nmf_sparse_mat, k=10, niter=100, algo=algoarg)
        return res

    def test_nmf_objerr(self, run_nmf):
        assert 700 < run_nmf.objErr <= 875
    def test_sparsity(self, run_nmf):
        sparsity = np.sum(np.isclose(run_nmf.H, 0.0)) / np.size(run_nmf.H)
        assert sparsity >= .35
    def test_nmf_prefill(self, nmf_sparse_mat, run_nmf, algoarg):
        newres = pyplanc.nmf(x=nmf_sparse_mat, k=10, niter=100, algo=algoarg, Winit=run_nmf.W, Hinit=run_nmf.H)
        assert 700 <= newres.objErr <= 875
    def test_consistency(self, nmf_sparse_mat, run_nmf, algoarg):
        newmat = nmf_sparse_mat.toarray()
        newres = pyplanc.nmf(x=newmat, k=10, niter=100, algo=algoarg)
        assert numpy.isclose(run_nmf.W.all(), newres.W.all())
        assert numpy.isclose(run_nmf.H.all(), newres.H.all())
        assert math.isclose(run_nmf.objErr, newres.objErr, abs_tol=5)

#   expect_error({
#     nmf(mat, k, algo = "hello")
#   }, "Please choose `algo` from")
# })



# symmat <- t(mat) %*% mat
# lambda <- 5

# test_that("dense, symNMF, anlsbpp", {
#     res <- symNMF(symmat, k, niter = 100, lambda = lambda, algo = "anlsbpp")
#     expect_lte(res$objErr, 4e7)

#     res <- symNMF(symmat, k, niter = 100, lambda = lambda, algo = "anlsbpp",
#                   Hinit = res$H)
#     expect_lte(res$objErr, 4e7)

#     expect_error({
#       symNMF(mat, k, 100)
#     }, "Input `x` is not square.")

#     expect_error({
#       symNMF(symmat, 1e4, 100)
#     })

#     expect_error({
#       symNMF(symmat, k, 100, Hinit = t(res$H))
#     }, "Hinit must be of size ")

#     expect_error({
#       symNMF(symmat, k, 100, algo = "hello")
#     }, "Please choose `algo` from")
# })

# test_that("dense, symNMF, gnsym", {
#   res <- symNMF(symmat, k, niter = 100, algo = "gnsym")
#   expect_lte(res$objErr, 5.8e4)
# })


# symmat.sp <- Matrix::t(mat.sp) %*% mat.sp

# test_that("sparse, symNMF, anlsbpp", {
#   res <- symNMF(symmat.sp, k, niter = 100, lambda = lambda, algo = "anlsbpp")
#   expect_lte(res$objErr, 4e5)
# })

# test_that("sparse, symNMF, gnsym", {
#   res <- symNMF(symmat.sp, k, niter = 100, algo = "gnsym")
#   expect_lte(res$objErr, 1e4)
# })
