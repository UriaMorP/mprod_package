import pytest
import numpy as np

from mprod import m_prod, x_m3, tensor_mtranspose
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_allclose, suppress_warnings,
    assert_raises_regex, HAS_LAPACK64,
)


def gen_m_transpose(mpair):
    mfun, minv = mpair

    def _do(a):
        return tensor_mtranspose(a, mfun, minv)

    return _do


def gen_m_product(mpair):
    mfun, minv = mpair

    def _do(a, b):
        return m_prod(a, b, mfun, minv)

    return _do


def assert_identity(J, tensor, mproduct):
    tensor2 = mproduct(J, tensor)
    assert_almost_equal(tensor, tensor2)


def assert_m_orth(tensor, mfun, minv):
    m, p, n = tensor.shape

    _t = gen_m_transpose((mfun, minv))
    _m = gen_m_product((mfun, minv))

    if m <= p:
        J = _m(tensor, _t(tensor))
    else:
        J = _m(_t(tensor), tensor)

    TENSOR_CASES = []
    for mode2_size in range(1, 10, 100):
        for i in range(10):
            rng = np.random.default_rng(seed=i + int(np.log10(mode2_size)))
            TENSOR_CASES.append(rng.random((J.shape[1], mode2_size, n)))

    @pytest.mark.parametrize('tens', TENSOR_CASES)
    def _assert_id(tens):
        assert_identity(J, tens, _m)

