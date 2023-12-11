import numpy as np
from typing import Tuple, Dict

from mprod._base import NumpynDArray, MatrixTensorProduct


def svdm(tens_a: np.ndarray, fun_m: MatrixTensorProduct, inv_m: MatrixTensorProduct
         , hats: bool = False) \
        -> Tuple[NumpynDArray, NumpynDArray, NumpynDArray]:
    """
    The svdm function is a helper function for computing the tsvdmII.
    This function does the **THIN** tsvdm: 
    ``u,s,b = tsvdm(tensor_a, m, inv_m)`` where ``u,v`` are
    ``(m,k,n)`` and ``(p,k,n)`` M-orthogonal tensors and ``s``
    is an f-diagonal tensor of shape ``(k,k,n)`` and ``k=min(p,m)``

    Parameters
    ----------
    tens_a: np.ndarray
        Tensor of shape ``(m,p,n)``
    fun_m: MatrixTensorProduct
        Invertible mat-vec operation for transforming ``tens_a`` tube fibers
    inv_m: MatrixTensorProduct
        Invertible mat-vec operation for transforming ``tens_a`` tube fibers.
        This operation is the inverse of ``fun_m``
    hats: bool
        Setting this to ``True`` will cause the function to return the tsvdm
        factors in the tensor domain transform.

    Returns
    -------
    tens_u: np.ndarray
        M-orthogonal tensor of shape ``(m,k,n)``
    tens_s: np.ndarray
        A ``(k,n)`` matrix representation of the f-diagonal tensor of
        shape ``(k,k,n)``
    tens_v: np.ndarray
        M-orthogonal Tensor of shape ``(p,k,n)``

    """
    m, p, n = tens_a.shape
    a_hat = fun_m(tens_a)

    # The code bellow is a super efficient numpy trick for performing the following
    #
    # u_hat = np.zeros((m, m, n))
    # s_hat = np.zeros((m, p, n))
    # v_hat = np.zeros((p, p, n))
    #
    # for i in range(n):
    #     uu, ss, vt = np.linalg.svd(a_hat[:, :, i], full_matrices=False)
    #
    #     us1, us2 = uu.shape
    #     vs1, vs2 = vt.shape
    #
    #     ssize = ss.size
    #     s_hat[:ssize, :ssize, i] = np.diag(ss)
    #     u_hat[:us1, :us2, i] = uu.copy()
    #     v_hat[:vs2, :vs1, i] = vt.T.copy()

    u_hat, s_hat, v_hat = np.linalg.svd(a_hat.transpose(2, 0, 1), full_matrices=False)
    u_hat, s_hat, v_hat = u_hat.transpose(1, 2, 0), s_hat.transpose(), v_hat.transpose(2, 1, 0)

    # sreshape = s_hat.copy().reshape(1, m, n)
    # sreshape = sreshape.transpose(1, 0, 2)
    # idreshape = np.eye(m, p).reshape(m, p, 1)

    # s_hat = idreshape @ sreshape

    if hats:
        return u_hat, s_hat, v_hat

    u = inv_m(u_hat)
    v = inv_m(v_hat)
    s = inv_m(s_hat)

    return u, s, v


def tsvdmii(tens_a: NumpynDArray,
            fun_m: MatrixTensorProduct,
            inv_m: MatrixTensorProduct,
            gamma: float = 1,
            n_components: int = None) -> \
        Tuple[Dict[int, NumpynDArray], Dict[int, NumpynDArray], Dict[int, NumpynDArray], float, Dict[int, int], int]:
    assert not ((gamma is not None) and (
            n_components is not None)), "Arguments gamma and n_components are mutually exclusive"
    assert (gamma is not None) or (
            n_components is not None), "Exactely one of arguments gamma, n_components must be defined"

    m, p, n = tens_a.shape

    # execute full decomposition
    u_hat, s_hat, v_hat = svdm(tens_a, fun_m, inv_m, hats=True)

    # compute variation in the decomposition
    #   var is the sorted (hat) squared singular values
    #   cumm_var is scre
    #   w_idx is an array of indices for `cumm_var` and `var`
    #   total_var is the (float) sum of squared singular values `var`
    var = np.concatenate([np.diagonal(s_hat[:, :, i]) for i in range(n)]) ** 2
    var = np.sort(var.reshape(-1))[::-1]
    cumm_var = var.cumsum(axis=0)
    w_idx = np.arange(0, cumm_var.size, dtype=int)
    total_variance = var.sum()

    # Find truncation threshold according to
    if gamma is not None:
        reduced_ind = w_idx[(cumm_var / total_variance) > gamma]
        if reduced_ind.size == 0:
            j = 0
        else:
            j = reduced_ind.min()
    else:
        j = n_components

    tau = np.sqrt(var[j - 1])
    rho = {}

    u_hat_rho_dict = {}
    s_hat_rho_dict = {}
    v_hat_rho_dict = {}

    max_rho = 0
    r = 0
    for i in range(n):
        diag_shat_i = np.diagonal(s_hat[:, :, i])
        tau_mask = (diag_shat_i >= tau)
        rho_i = tau_mask.sum()
        if rho_i > 0:
            u_hat_rho_dict[i] = u_hat[:, :rho_i, i].copy()
            s_hat_rho_dict[i] = s_hat[:rho_i, :rho_i, i].copy()
            v_hat_rho_dict[i] = v_hat[:, :rho_i, i].copy()
            rho[i] = rho_i

            if rho_i > max_rho:
                max_rho = rho_i
            r += rho_i

    if n_components is not None:
        assert r == n_components, f"expected multirank {n_components} got {r}"

    return u_hat_rho_dict, s_hat_rho_dict, v_hat_rho_dict, total_variance, rho, r
