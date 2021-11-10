from typing import Iterable, Tuple
from numpy import ndarray


def _assert_order(tensor: ndarray, tensor_varname: str, order: int):
    got_order = len(tensor.shape)
    assert got_order == order, f"{tensor_varname} must be a order {order} tensor, found order {got_order}"


def _assert_size(tensor: ndarray, tensor_varname: str, axis: int, dim: int):
    got_dim = tensor.shape[axis]
    assert got_dim == dim, f"Dimension {axis} of {tensor_varname} must equal {dim}, found {got_dim}"


def _assert_order_and_mdim(tensor: ndarray,
                          tensor_varname: str,
                          order: int,
                          dim_inspection_list: Iterable[Tuple[int, int]]):
    """

    Parameters
    ----------
    tensor: np.ndarray
        The tensor for inpection
    tensor_varname: str
        The variable name of the tensor as it appears in the code
    order: int
        The intended order of `tensor`
    dim_inspection_list


    """
    _assert_order(tensor, tensor_varname, order)
    for ax, dim in dim_inspection_list:
        assert ax < order, f"Trying to assert the dimension of mode {ax} of a {order} order tensor {tensor_varname}"
        _assert_size(tensor, tensor_varname, ax, dim)

