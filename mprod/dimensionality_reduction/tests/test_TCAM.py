import pytest
import numpy as np

from itertools import product

from mprod.dimensionality_reduction import TCAM
from mprod import MeanDeviationForm, table2tensor
from mprod.tests._utils import (_make_mprod_op_cases, _make_tensor_cases, gen_m_product, gen_m_transpose,
                                assert_m_orth, _make_mprod_op_cases_tube, _make_gapped_table,
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


def test_tcam_scalar_type_tables():
    # regression: numpy 2.0 dropped the single-char keys of np.sctypeDict, so
    # `from mprod.dimensionality_reduction import TCAM` raised KeyError: 'e'.
    from mprod.dimensionality_reduction import _tcam
    assert_equal(_tcam._float_types,
                 [np.float16, np.float32, np.float64, np.longdouble, float])
    assert_equal(_tcam._int_types,
                 [np.int8, np.int16, np.int32, np.intp, int])


GAPPED_TUBE = 5
GAPPED_M_FUN_CASES = _make_mprod_op_cases_tube(GAPPED_TUBE)


@pytest.mark.parametrize('n_components', [2, 0.5])
@pytest.mark.parametrize('mpair', GAPPED_M_FUN_CASES + [None])
def test_tcam_fit_transform_incomplete_table(n_components, mpair):
    # regression: with an x_m3-based fun_m, MeanDeviationForm.transform used to
    # return a memoryview and x_m3's 3-d branch raised AttributeError.
    table = _make_gapped_table(nn=GAPPED_TUBE, seed=1, n_drop=4)
    tensor, mode1_map, mode3_map = table2tensor(table, missing_flag=True)
    assert_(isinstance(tensor, np.ma.MaskedArray))

    if mpair is None:
        tca = TCAM(n_components=n_components)
    else:
        mfun, minv = mpair
        tca = TCAM(fun_m=mfun, inv_m=minv, n_components=n_components)

    Y = tca.fit_transform(tensor)
    assert_(isinstance(Y, np.ndarray), msg=f"fit_transform returned {type(Y)}")
    assert_equal(Y.shape[0], tensor.shape[0])
    assert_allclose(Y, tca.fit(tensor).transform(tensor))
    assert_equal(tca.inverse_transform(Y).shape, tensor.shape)
