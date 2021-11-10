import pytest
import numpy as np

from itertools import product

from mprod.dimensionality_reduction import TCAM
from mprod import MeanDeviationForm
from mprod.tests._utils import (_make_mprod_op_cases, _make_tensor_cases, gen_m_product, gen_m_transpose, assert_m_orth,
                                m, n, p)

from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_allclose, suppress_warnings,
    assert_raises_regex, HAS_LAPACK64,
)

M_FUN_CASES = _make_mprod_op_cases()[:1]
TENSOR_CASES = _make_tensor_cases()[:1]

@pytest.mark.parametrize('X', TENSOR_CASES)
@pytest.mark.parametrize('n_components', np.linspace(1, min(m, p) * n - 1, 3, dtype=int))
@pytest.mark.parametrize('mpair', M_FUN_CASES + [None])
def test_tcam_fit_transform(X, n_components, mpair):
    print(min(m, p) * n - 1)
    if mpair is None:
        tca = TCAM(n_components=n_components)
    else:
        mfun, minv = mpair
        tca = TCAM(fun_m=mfun, inv_m=minv, n_components=n_components)

    X_r = tca.fit(X).transform(X)
    assert X_r.shape[1] == n_components

    # check the equivalence of fit.transform and fit_transform
    X_r2 = tca.fit_transform(X)
    assert_allclose(X_r, X_r2)
    # X_r = tca.transform(X)
    assert_allclose(X_r, X_r2)


@pytest.mark.parametrize('X', TENSOR_CASES)
@pytest.mark.parametrize('n_components', np.linspace(.1, 1., 3, dtype=float))
@pytest.mark.parametrize('mpair', M_FUN_CASES + [None])
def test_tcam_reconstruction_err(X, n_components, mpair):
    print(min(m, p) * n - 1)
    if mpair is None:
        tca = TCAM(n_components=n_components)

    else:
        mfun, minv = mpair
        tca = TCAM(fun_m=mfun, inv_m=minv, n_components=n_components)
    # check the shape of fit.transform
    Y = tca.fit(X).transform(X)
    X2 = tca.inverse_transform(Y)

    assert np.round(1 - ((X2 - X) ** 2).sum() / (X ** 2).sum(), 20) >= n_components


@pytest.mark.parametrize('X', TENSOR_CASES)
@pytest.mark.parametrize('n_components', range(1, min(m, p) * n - 1, 200))
@pytest.mark.parametrize('mpair', M_FUN_CASES + [None])
def test_tcam_residue_m_orth(X, n_components, mpair):
    print(min(m, p) * n - 1)
    if mpair is None:
        tca = TCAM(n_components=n_components)

    else:
        mfun, minv = mpair
        tca = TCAM(fun_m=mfun, inv_m=minv, n_components=n_components)

    Y = tca.fit(X).transform(X)
    X2 = tca.inverse_transform(Y)
    _t = gen_m_transpose((tca.fun_m, tca.inv_m))

    res_prod_norm = (tca._mprod(_t(X - X2), X2) ** 2).sum()
    assert_almost_equal(res_prod_norm, 0, err_msg=f"got {res_prod_norm} instead of 0", verbose=True, )



