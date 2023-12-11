""" Test functions for decompositions module

"""

import pytest
import numpy as np

from itertools import product

from mprod.decompositions import svdm, tqrm
from mprod.tests._utils import (_make_mprod_op_cases, _make_tensor_cases, gen_m_product, gen_m_transpose, assert_m_orth)

from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_allclose, suppress_warnings,
    assert_raises_regex, HAS_LAPACK64,
)

M_FUN_CASES = _make_mprod_op_cases()
TENSOR_CASES = _make_tensor_cases()


@pytest.mark.parametrize('tensor, m_pair', product(TENSOR_CASES, M_FUN_CASES))
def test_tsvdm(tensor, m_pair):
    mfun, minv = m_pair
    # _m = gen_m_product(m_pair)
    # _t = gen_m_transpose(m_pair)

    u, s, v = svdm(tensor, mfun, minv)
    m, p, n = tensor.shape
    rk = min(m, p)

    assert s.shape[0] == rk, f"expected shape[0] of s to be {rk}, got {s.shape[0]}"
    assert s.shape[1] == tensor.shape[-1], f"expected shape[1] of s to be {tensor.shape[-1]}, got {s.shape[1]}"


    # tensor2 = _m(_m(u, s), _t(v))
    shat =  mfun(s)
    us = mfun(u).transpose(2, 0, 1) * shat.T.reshape(n, 1, m)
    usv = np.matmul(us, mfun(v).transpose(2, 1, 0))
    usv = usv.transpose(1, 2, 0)
    tensor2 = minv(usv)
    assert_almost_equal(tensor, tensor2)

    assert_m_orth(u, *m_pair)
    assert_m_orth(v, *m_pair)


@pytest.mark.parametrize('tensor, m_pair', product(TENSOR_CASES, M_FUN_CASES))
def test_tqrm(tensor, m_pair):
    mfun, minv = m_pair

    _m = gen_m_product(m_pair)
    _t = gen_m_transpose(m_pair)

    Q, R = tqrm(tensor, mfun, minv)

    tensor2 = _m(Q, R)
    assert_almost_equal(tensor, tensor2)

    assert_m_orth(Q, *m_pair)
