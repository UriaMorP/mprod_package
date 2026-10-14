""" Test functions for the _ml_helpers module
"""

import pytest
import numpy as np

from mprod import MeanDeviationForm, table2tensor
from mprod.tests._utils import _make_masked_tensor_cases, _make_gapped_table

from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_allclose,
)

MASKED_TENSOR_CASES = _make_masked_tensor_cases()


@pytest.mark.parametrize('X', MASKED_TENSOR_CASES)
def test_mdf_masked_transform_returns_ndarray(X):
    # regression: np.ma.MaskedArray.filled() already returns an ndarray, so the
    # old `.filled().data` handed back a memoryview.
    mdf = MeanDeviationForm()
    Xt = mdf.fit_transform(X)

    assert_(isinstance(Xt, np.ndarray), msg=f"transform returned {type(Xt)}")
    assert_(not isinstance(Xt, np.ma.MaskedArray))
    assert_equal(Xt.shape, X.shape)
    assert_(hasattr(Xt, 'T') and hasattr(Xt, 'transpose'))

    X2 = mdf.inverse_transform(np.ma.array(Xt, mask=X.mask, fill_value=0))
    assert_(isinstance(X2, np.ndarray), msg=f"inverse_transform returned {type(X2)}")
    assert_(not isinstance(X2, np.ma.MaskedArray))
    assert_allclose(X2[~X.mask], X.data[~X.mask])


@pytest.mark.parametrize('X', MASKED_TENSOR_CASES)
def test_mdf_dense_transform_roundtrip(X):
    mdf = MeanDeviationForm()
    Xd = np.asarray(X.filled())
    Xt = mdf.fit_transform(Xd)
    assert_(isinstance(Xt, np.ndarray))
    assert_allclose(mdf.inverse_transform(Xt), Xd)


def test_table2tensor_incomplete_grid_is_masked():
    table = _make_gapped_table(n_drop=4)
    tensor, mode1_map, mode3_map = table2tensor(table, missing_flag=True)
    assert_(isinstance(tensor, np.ma.MaskedArray))
    assert_equal(tensor.shape, (12, 7, 5))
    assert_equal(int(tensor.mask[:, 0, :].sum()), 4)


def test_table2tensor_incomplete_grid_raises_by_default():
    table = _make_gapped_table(n_drop=4)
    assert_raises(KeyError, table2tensor, table, False)
