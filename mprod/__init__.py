"""

"""
# import dimensionality_reduction
# from .dimensionality_reduction._tcam import TCAM
# import decompositions as decompositions
# import dimensionality_reduction
from ._ml_helpers import MeanDeviationForm, table2tensor
from ._base import m_prod, tensor_mtranspose, x_m3

from mprod._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
#
__all__ = [
    "m_prod",
    "tensor_mtranspose",
    "x_m3",
    "MeanDeviationForm",
    "table2tensor",
    "dimensionality_reduction",
    "decompositions"
]
