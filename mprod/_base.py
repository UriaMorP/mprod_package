import numpy as np
from typing import Callable, Tuple, Dict, List

import scipy.fft
from scipy.fft import dct, idct, rfft, irfft
from scipy.stats import ortho_group

NumpynDArray = np.ndarray
MatrixTensorProduct = Callable[[NumpynDArray], NumpynDArray]


def _default_transform(tube_size: int) -> Tuple[MatrixTensorProduct, MatrixTensorProduct]:
    def fun_m(x):
        return dct(x, type=2, n=tube_size, axis=-1, norm='ortho')

    def inv_m(x):
        return idct(x, type=2, n=tube_size, axis=-1, norm='ortho')

    return fun_m, inv_m


def generate_dct(tube_size: int, dct_type: int = 2) -> Tuple[MatrixTensorProduct, MatrixTensorProduct]:
    """Generates a DCT based tensor-matrix operation (forward and inverse)

    Parameters
    ----------

    tube_size: int
        the fiber-tube size of the tensors of interest

    dct_type: int, default = 2
        The choice of dct type, see scipy.fft.dct.__doc__ for details

    Returns
    -------

    fun_m: MatrixTensorProduct
        A tensor transform

    inv_m: MatrixTensorProduct
        A tensor transform (the inverse of `fun_m`)

    """

    def fun_m(x):
        return dct(x, type=dct_type, n=tube_size, axis=-1, norm='ortho')

    def inv_m(x):
        return idct(x, type=dct_type, n=tube_size, axis=-1, norm='ortho')

    return fun_m, inv_m


# noinspection PyPep8Naming
def _mod3prod(A: NumpynDArray, funM: MatrixTensorProduct) -> NumpynDArray:
    """Maps a tensor `A` to the tensor domain transform defined by the operation of a mapping `funM` on
    the tube fibers of `A`

    Parameters
    ----------

    A: NumpynDArray
        Tensor with `A.shape[2] == n`

    funM: MatrixTensorProduct
        Picklable mapping that operates on (n dimensional) tube fibers of a tensor

    Returns
    -------

    hatA: MatrixTensorProduct
        Returns domain transform of `A` defined by the operation of `funM`
    """
    m, p, n = A.shape
    return funM(A.transpose((2, 1, 0)).reshape(n, m * p)).reshape((n, p, m)).transpose((2, 1, 0))


def x_m3(M: NumpynDArray) -> MatrixTensorProduct:
    """
    Creates a picklable tensor transformation forming the mod3 tensor-matrix multiplication required in the M product
    definition.

    Parameters
    ----------
    M: np.ndarray
        A matrix of shape `(n,n)`

    Returns
    -------
    fun:  Callable[[NumpynDArray], NumpynDArray]
        Picklable mapping that operates on (n dimensional) tube fibers of a tensor

    """
    assert len(M.shape) == 2, "M must be a 2 dimensional matrix"
    assert M.shape[0] == M.shape[1], "M must be a square matrix"
    
    tube_size = M.shape[0]
    def fun(A: NumpynDArray) -> NumpynDArray:
        assert A.shape[-1] == tube_size, "The last dimension of A must be the same as the tube size "
        if len(A.shape) == 2:
            # the case where A is a matrix representation of f-diagonal tensor
            return  A @ M.T
        elif len(A.shape) == 3:
            m, p, n = A.shape
            return (M @ A.transpose((2, 1, 0)).reshape(n, m * p)).reshape((n, p, m)).transpose((2, 1, 0))
        else:
            raise NotImplementedError("We only work with 3d tensors for now!")
    return fun


def generate_haar(tube_size: int, random_state = None) -> Tuple[MatrixTensorProduct, MatrixTensorProduct]:
    """Generates a tensor-matrix transformation based on random sampling of unitary matrix
    (according to the Haar distribution on O_n See scipy.stats.)

    Parameters
    ----------

    tube_size: int
        the fiber-tube size of the tensors of interest

    Returns
    -------

    fun_m: MatrixTensorProduct
        A tensor transform

    inv_m: MatrixTensorProduct
        A tensor transform (the inverse of `fun_m`)

    """

    M = ortho_group.rvs(tube_size, random_state=random_state)

    fun_m = x_m3(M)
    inv_m = x_m3(M.T)

    return fun_m, inv_m


def m_prod(tens_a: NumpynDArray,
           tens_b: NumpynDArray,
           fun_m: MatrixTensorProduct,
           inv_m: MatrixTensorProduct) -> NumpynDArray:
    """
    Returns the :math:`\\star_{\\mathbf{M}}` product of tensors `A` and `B`
    where ``A.shape == (m,p,n)`` and ``B.shape == (p,r,n)``.

    Parameters
    ----------
    tens_a: array-like
        3'rd order tensor with shape `m x p x n`

    tens_b: array-like
        3'rd order tensor with shape `p x r x n`

    fun_m: MatrixTensorProduct, Callable[[NumpynDArray], NumpynDArray]
        Invertible linear mapping from `R^n` to `R^n`

    inv_m: MatrixTensorProduct, Callable[[NumpynDArray], NumpynDArray]
        Invertible linear mapping from R^n to R^n ( `fun_m(inv_m(x)) = inv_m(fun_m(x)) = x` )

    Returns
    -------
    tensor: array-like
        3'rd order tensor of shape `m x r x n` that is the star :math:`\\star_{\\mathbf{M}}`
        product of `A` and `B`
    """

    assert tens_a.shape[1] == tens_b.shape[0]
    assert tens_a.shape[-1] == tens_b.shape[-1]

    a_hat = fun_m(tens_a)
    b_hat = fun_m(tens_b)

    c_hat = np.einsum('mpi,pli->mli', a_hat, b_hat)
    return inv_m(c_hat)


# copied version from transformers.py
# def m_prod(A: NumpynDArray, B: NumpynDArray, funM: MatrixTensorProduct, invM: MatrixTensorProduct) -> NumpynDArray:
#     # assert A.shape[1] == B.shape[0]
#     # assert A.shape[-1] == B.shape[-1]
#     A_hat = funM(A)
#     B_hat = funM(B)
#
#     calE_hat = np.einsum('mpi,pli->mli', A_hat, B_hat)
#     return invM(calE_hat)

def tensor_mtranspose(tensor, mfun, minv):
    tensor_hat = mfun(tensor)
    tensor_hat_t = tensor_hat.transpose((1, 0, 2))
    tensor_t = minv(tensor_hat_t)
    return tensor_t


def _t_pinv_fdiag(F, Mfun, Minv) -> NumpynDArray:
    m, p, n = F.shape
    hat_f = Mfun(F)

    pinv_hat_f = np.zeros_like(hat_f)
    for i in range(n):
        fi_diag = np.diagonal(hat_f[:, :, i]).copy()
        fi_diag[(fi_diag ** 2) > 1e-6] = 1 / fi_diag[(fi_diag ** 2) > 1e-6]

        pinv_hat_f[:fi_diag.size, :fi_diag.size, i] = np.diag(fi_diag)

    pinv_f = Minv(pinv_hat_f)

    return tensor_mtranspose(pinv_f, Mfun, Minv)

# # TODO: Is TensorArray needed ?
# # noinspection PyPep8Naming
# class TensorArray(np.ndarray):
#     def __new__(cls, input_array):
#         # Input array is an already formed ndarray instance
#         # We first cast to be our class type
#         obj = np.asarray(input_array).view(cls)
#         # add the new attribute to the created instance
#         # Finally, we must return the newly created object:
#         return obj
#
#     @property
#     def TT(self):
#         return self.transpose((1, 0, 2))
#
#     def __array_finalize__(self, obj):
#         # see InfoArray.__array_finalize__ for comments
#         if obj is None: return
#         self.info = getattr(obj, 'info', None)
