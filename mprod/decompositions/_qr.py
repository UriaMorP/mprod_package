import numpy as np
from typing import Tuple, Dict

from mprod._base import NumpynDArray, MatrixTensorProduct


def tqrm(tens_a: np.ndarray, fun_m: MatrixTensorProduct, inv_m: MatrixTensorProduct, hats: bool = False) \
        -> Tuple[NumpynDArray, NumpynDArray]:
    """
    The ``tqrm`` implements tensor-QR decomposition:
    ``Q,R = tqrm(tensor_a, m, inv_m)`` where ``Q`` is M-orthogonal tensor of shape ``(m,m,n)`` and ``R`` is f-upper
    triangular tensor of shape ``(m,p,n)``

    Parameters
    ----------
    tens_a: np.ndarray
        Tensor of shape ``(m,p,n)``
    fun_m: MatrixTensorProduct
        Invertible mat-vec operation for transforming ``tens_a`` tube fibers
    inv_m: MatrixTensorProduct
        Invertible mat-vec operation for transforming ``tens_a`` tube fibers. This operation is the inverse of ``fun_m``
    hats: bool
        Setting this to ``True`` will cause the function to return the tqrm factors in the tensor domain transform.

    Returns
    -------
    tens_q: np.ndarray
        M-orthogonal tensor of shape ``(m,m,n)``
    tens_r: np.ndarray
        f-upper triangular tensor of shape ``(m,p,n)``

    """

    m, p, n = tens_a.shape
    a_hat = fun_m(tens_a)

    q_hat = np.zeros((m, m, n))
    r_hat = np.zeros((m, p, n))
    k = 0

    for i in range(n):
        qq, rr = np.linalg.qr(a_hat[:, :, i])

        qs1, qs2 = qq.shape
        rs1, rs2 = rr.shape

        q_hat[:qs1, :qs2, i] = np.copy(qq)
        r_hat[:rs1, :rs2, i] = np.copy(rr)

        k = max(k, max(qs2, rs1))

    # truncate sizes
    q_hat = q_hat[:, :k, :]
    r_hat = r_hat[:k, :, :]

    if hats:
        return q_hat, r_hat

    tens_q = inv_m(q_hat)
    tens_r = inv_m(r_hat)

    return tens_q, tens_r
