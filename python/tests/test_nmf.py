from typing import NamedTuple
import pytest
import numpy as np
import numpy.typing
import pyplanc


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
    mat = rs.random((nmf_params[0], nmf_params[1]), np.float64)
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

    def test_dense_nmf_prefill(self, nmf_dense_mat, run_nmf, algoarg):
        newres = pyplanc.nmf(x=nmf_dense_mat, k=10, niter=100, algo=algoarg, Winit=run_nmf.W, Hinit=run_nmf.H)
        assert 1850 <= newres.objErr and newres.objErr <= 2650

    def test_anlsbpp_type(self, run_nmf: pyplanc.nmfOutput):
        assert isinstance(run_nmf, pyplanc.nmfOutput)

    def test_dense_nmf_shape(self, run_nmf: pyplanc.nmfOutput, nmf_params: nmfparams):
        assert run_nmf.W.shape == (nmf_params.m, 10)
        assert run_nmf.H.shape == (nmf_params.n, 10)

    def test_dense_nmf_objerr(self, run_nmf):
        assert 1850 <= run_nmf.objErr and run_nmf.objErr <= 2650


class TestNMFDenseFail:
    @pytest.fixture(scope="class", params=["hello"])
    def algoarg(self, request):
        param = request.param
        yield param

    def test_bad_dense_algo_arg(self, nmf_dense_mat: numpy.typing.NDArray[np.float64], algoarg) -> pyplanc.nmfOutput:
        with pytest.raises(RuntimeError) as e:
            pyplanc.nmf(x=nmf_dense_mat, k=10, niter=100, algo=algoarg)
        assert 'Please choose `algo` from' in str(e.value)
#     # Failing use
#     expect_error({
#       nmf(mat, k, algo = "hello")
#     }, "Please choose `algo` from")

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

# library(Matrix)
# # Sparsen the `mat`
# sparsity <- .9
# regenerate <- TRUE
# while (regenerate) {
#     zero.idx <- sample(length(mat), round(sparsity * length(mat)))
#     mat.sp <- mat
#     mat.sp[zero.idx] <- 0
#     mat.sp <- as(mat.sp, "CsparseMatrix")
#     # Make sure there is no col/row that has all zero
#     if (sum(Matrix::colSums(mat.sp) == 0) == 0 &&
#         sum(Matrix::rowSums(mat.sp) == 0) == 0) {
#       regenerate <- FALSE
#     }
# }

# test_that("sparse, nmf, anlsbpp", {
#   set.seed(1)
#   res1 <- nmf(mat.sp, k, niter = 100)
#   expect_type(res1, "list")
#   expect_equal(nrow(res1$W), m)
#   expect_equal(ncol(res1$W), k)
#   expect_equal(nrow(res1$H), n)
#   expect_equal(ncol(res1$H), k)
#   expect_lte(res1$objErr, 800)
#   set.seed(1)
#   res2 <- nmf(as.matrix(mat.sp), k, niter = 100)
#   expect_true(all.equal(res1, res2))
#   # Using init W and H
#   res <- nmf(mat.sp, k, niter = 100, Winit = res1$W, Hinit = res1$H)
#   # Expected max objective error
#   expect_lte(res$objErr, 800)
#   # Expected min sparsity of W
#   W.sparsity <- sum(res$W == 0) / length(res$W)
#   cat("\nW sparsity:",W.sparsity,"\n")
#   expect_gte(W.sparsity, .4)

#   expect_error({
#     nmf(mat, k, algo = "hello")
#   }, "Please choose `algo` from")
# })

# test_that("sparse, nmf, admm", {
#   res <- nmf(mat.sp, k, niter = 100, algo = "admm")
#   expect_lte(res$objErr, 800)
# })

# test_that("sparse, nmf, hals", {
#   res <- nmf(mat.sp, k, niter = 100, algo = "hals")
#   expect_lte(res$objErr, 800)
# })

# test_that("sparse, nmf, mu", {
#   res <- nmf(mat.sp, k, niter = 100, algo = "mu")
#   expect_lte(res$objErr, 950)
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
