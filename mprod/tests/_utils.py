import pytest
import numpy as np

from mprod import m_prod, x_m3, tensor_mtranspose, generate_dct, generate_haar
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_allclose, suppress_warnings,
    assert_raises_regex, HAS_LAPACK64,
)


m, p, n = 100, 300, 10


def _make_mprod_op_cases():
    mprod_cases = []
    for i in range(2):
        rng = np.random.default_rng(seed=i)
        mat = rng.random((n, n))
        mat_m = np.linalg.qr(mat)[0]
        mfun = x_m3(mat_m)
        minv = x_m3(mat_m.T)
        mprod_cases.append((mfun,minv))
        
    # add dct based transforms
    mfun, minv = generate_dct(n)
    mprod_cases.append((mfun,minv))
    return mprod_cases


def _make_tensor_cases():
    tensor_cases = []

    for i in range(2):
        rng = np.random.default_rng(seed=i)
        X = rng.random((m, p, n))
        tensor_cases.append(X)
    return tensor_cases


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
        for i in range(2):
            rng = np.random.default_rng(seed=i + int(np.log10(mode2_size)))
            TENSOR_CASES.append(rng.random((J.shape[1], mode2_size, n)))

    @pytest.mark.parametrize('tens', TENSOR_CASES)
    def _assert_id(tens):
        assert_identity(J, tens, _m)


def _make_mprod_op_cases_tube(tube_size):
    """`_make_mprod_op_cases` for an arbitrary tube size."""
    cases = []
    for i in range(2):
        rng = np.random.default_rng(seed=i)
        mat_m = np.linalg.qr(rng.random((tube_size, tube_size)))[0]
        cases.append((x_m3(mat_m), x_m3(mat_m.T)))
    cases.append(generate_haar(tube_size, random_state=0))
    cases.append(generate_dct(tube_size))
    return cases


def _make_masked_tensor_cases(mm=12, pp=7, nn=5):
    cases = []
    for i in range(2):
        rng = np.random.default_rng(seed=i)
        X = rng.random((mm, pp, nn))
        mask = np.zeros((mm, pp, nn), dtype=bool)
        mask[0, :, -1] = True
        mask[3, :, 0] = True
        cases.append(np.ma.array(X, mask=mask, fill_value=0))
    return cases


def _make_gapped_table(mm=12, pp=7, nn=5, seed=0, n_drop=4):
    """A (subject x rep) x feature table whose id x mode3 grid is incomplete."""
    import pandas as pd
    rng = np.random.default_rng(seed=seed)
    idx = pd.MultiIndex.from_product(
        [[f"s{i}" for i in range(mm)], [f"t{j}" for j in range(nn)]],
        names=["SubjectID", "rep"])
    table = pd.DataFrame(rng.random((len(idx), pp)), index=idx,
                         columns=[f"f{k}" for k in range(pp)])
    return table.drop(index=table.index[rng.choice(len(idx), size=n_drop, replace=False)])
