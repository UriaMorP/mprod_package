"""TCAM
"""
import numpy as np
from dataclasses import dataclass
from sklearn.base import TransformerMixin, BaseEstimator

from .._base import m_prod, tensor_mtranspose, _default_transform, _t_pinv_fdiag
from .._base import MatrixTensorProduct, NumpynDArray
from ..decompositions import svdm
from .._misc import _assert_order_and_mdim
from .._ml_helpers import MeanDeviationForm

_float_types = [np.sctypeDict[c] for c in 'efdg'] + [float]
_int_types = [np.sctypeDict[c] for c in 'bhip'] + [int]


def _pinv_diag(diag_tensor):
    sinv = diag_tensor.copy()
    sinv += ((diag_tensor ** 2) <= 1e-6) * 1e+20
    sinv = (((diag_tensor ** 2) > 1e-6) * (1 / sinv))
    return sinv


@dataclass
class TensorSVDResults:
    u: np.ndarray
    s: np.ndarray
    v: np.ndarray

    def astuple(self):
        return self.u.copy(), self.s.copy(), self.v.copy()


# noinspection PyPep8Naming
class TCAM(TransformerMixin, BaseEstimator):
    """tsvdm based tensor component analysis (TCAM).
    Linear dimensionality reduction using tensor Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the tSVDM (using :mod:`mprod.MeanDeviationForm` ) .
    It uses the :mod:`mprod.decompositions.svdm` function as basis for the ``TSVDMII`` algorithm from Kilmer et. al.
    (https://doi.org/10.1073/pnas.2015851118) then offers a CP like transformations of the data accordingly.
    See https://arxiv.org/abs/2111.14159 for theoretical results and case studies, and the :ref:`Tutorials <TCAM>`
    for elaborated examples

    Parameters
    ----------
    n_components : int, float, default=None
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(m_samples, p_features) * n_reps - 1

        If ``0 < n_components < 1`` , select the number of components such that the
        amount of variance that needs to be explained is greater than the percentage specified
        by n_components. In case ``n_components >= 1`` is an integer then the estimated number
        of components will be::

            n_components_ == min(n_components, min(m_samples, p_features) * n_reps - 1)


    Attributes
    ----------
    n_components_ : int
        The estimated number of components. When n_components is set
        to a number between 0 and 1. this number is estimated from input data.
        Otherwise it equals the parameter n_components,
        or `min(m_samples, p_features) * n_reps -1` if n_components is None.

    explained_variance_ratio_ : ndarray of shape (`n_components_`,)
        The amount of variance explained by each of the selected components.

    mode2_loadings : ndarray (float) of shape (`n_components_`, `n_features` )
        A matrix representing the contribution (coefficient) of each feature in the orinial
        features space (2'nd mode of the tensor) to each of the TCAM factors.


    Methods
    -------
    fit:
        Compute the TCAM transformation for a given dataset
    transform:
        Transform a given dataset using a fitted TCAM
    fit_transform:
        Fit a TCAM to a dataset then return its TCAM transformation
    inverse_transform:
        Given points in the reduced TCAM space, compute the points pre-image in the original features space.


    """

    def __init__(self, fun_m: MatrixTensorProduct = None,
                 inv_m: MatrixTensorProduct = None,
                 n_components=None):
        assert (type(n_components) in _int_types and (n_components >= 1)) or \
               ((type(n_components) in _float_types) and (0 < n_components <= 1)) \
               or (n_components is None), f"`n_components` must be positive integer or a float between 0 and 1" \
                                          f" or `None`, got {n_components} of type {type(n_components)}"

        assert (fun_m is None) == (inv_m is None), "Only one of fun_m,inv_m is None. " \
                                                   "Both must be defined (or both None)"

        self.n_components = n_components

        self.fun_m = fun_m
        self.inv_m = inv_m
        self._mdf = MeanDeviationForm()

    def _mprod(self, a, b) -> NumpynDArray:
        return m_prod(a, b, self.fun_m, self.inv_m)

    def _fit(self, X: np.ndarray):
        max_rank = self._n * min(self._m, self._p) - 1

        self._hat_svdm = TensorSVDResults(*svdm(X, self.fun_m, self.inv_m, hats=True))

        # get factors order
        diagonals = self._hat_svdm.s.transpose().copy()
        self._factors_order = np.unravel_index(np.argsort(- (diagonals ** 2), axis=None), diagonals.shape)
        self._sorted_singular_vals = diagonals[self._factors_order]
        self._total_variation = (self._sorted_singular_vals ** 2).sum()
        self.explained_variance_ratio_ = ((self._sorted_singular_vals ** 2) / self._total_variation)

        # populate n_components if not given
        if self.n_components is None:
            self.n_components_ = max_rank
        elif type(self.n_components) in _int_types and self.n_components > 0:
            self.n_components_ = min(max_rank, self.n_components)
        elif type(self.n_components) in _float_types and self.n_components == 1.:
            self.n_components_ = max_rank
        elif 0 < self.n_components < 1 and type(self.n_components) in _float_types:
            var_cumsum = (self._sorted_singular_vals ** 2).cumsum()  # w in the paper
            w_idx = np.arange(0, var_cumsum.size, dtype=int)  # w index
            self.n_components_ = min(max_rank,
                                     w_idx[(var_cumsum / self._total_variation) > self.n_components].min() + 1)
        else:
            raise ValueError("Unexpected edge case for the value of `n_components`")

        self.n_components_ = max(1, self.n_components_)

        self._n_factors_order = tuple([self._factors_order[0][:self.n_components_].copy(),
                                       self._factors_order[1][:self.n_components_].copy()])

        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components_]
        self._rrho = np.array([0 for _ in range(self._n)])
        for nn, rr in zip(*self._n_factors_order):
            self._rrho[nn] = max(self._rrho[nn], rr + 1)
        # self._rrho += 1
        # populate truncations
        # _tau = self._sorted_singular_vals[self.n_components_ + 1]
        # self._rrho = (diagonals > _tau).sum(axis=1)
        self._truncated_hat_svdm = TensorSVDResults(*self._hat_svdm.astuple())

        self._truncated_hat_svdm.u = self._truncated_hat_svdm.u[:, :self._rrho.max(), :]
        self._truncated_hat_svdm.s = self._truncated_hat_svdm.s[:self._rrho.max(), :]
        self._truncated_hat_svdm.v = self._truncated_hat_svdm.v[:, :self._rrho.max(), :]

        for i, rho_i in enumerate(self._rrho):
            self._truncated_hat_svdm.u[:, rho_i:, i] = 0
            self._truncated_hat_svdm.s[rho_i:, i] = 0
            self._truncated_hat_svdm.v[:, rho_i:, i] = 0

        self._truncated_svdm = TensorSVDResults(self.inv_m(self._truncated_hat_svdm.u),
                                                self.inv_m(self._truncated_hat_svdm.s),
                                                self.inv_m(self._truncated_hat_svdm.v))

        self._truncS_pinv = self._truncated_svdm.s.copy()
        self._truncS_pinv[(self._truncS_pinv ** 2) <= 1e-6] = 0
        self._truncS_pinv[(self._truncS_pinv ** 2) > 1e-6] = 1 / self._truncS_pinv[(self._truncS_pinv ** 2) > 1e-6]

        return self

    # noinspection PyUnusedLocal
    def fit(self, X, y=None, **fit_params):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (m_samples, p_features, n_modes)
            Training data, where m_samples is the number of samples,
            p_features is the number of features and n_modes is the
            number of modes (timepoints/locations etc...)

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.


        Examples
        --------
        >>> from mprod.dimensionality_reduction import TCAM
        >>> import numpy as np
        >>> X = np.random.randn(10,20,4)
        >>> tca = TCAM()
        >>> mdf = tca.fit(X)


        """

        assert len(X.shape) == 3, "X must be a 3'rd order tensor"
        self._m, self._p, self._n = X.shape

        if self.fun_m is None:
            self.fun_m, self.inv_m = _default_transform(self._n)
        _X = self._mdf.fit_transform(X)

        return self._fit(_X)

    def _mode0_reduce(self, tU):
        return np.concatenate(
            [self._sorted_singular_vals[e] * tU[:, [fj], [fi]] for e, (fi, fj) in
             enumerate(zip(*self._n_factors_order))],
            axis=1)

    def _mode1_reduce(self, tV):
        return np.concatenate(
            [self._sorted_singular_vals[e] * tV[:, [fj], [fi]] for e, (fi, fj) in
             enumerate(zip(*self._n_factors_order))],
            axis=1)

    def _mode0_projector(self, X):

        trunc_U, trunc_S, trunc_V = self._truncated_hat_svdm.astuple()
        # trunc_Spinv = _t_pinv_fdiag(trunc_S, self.fun_m, self.inv_m)
        # XV = self._mprod(X, trunc_V)
        # XVS = self._mprod(XV, trunc_Spinv)
        # XVS_hat = self.fun_m(XVS)

        XV_hat = np.matmul(self.fun_m(X).transpose(2, 0, 1), trunc_V.transpose(2, 0, 1)).transpose(1, 2, 0)
        Y = XV_hat[:, self._n_factors_order[1], self._n_factors_order[0]].copy()

        # XV_hat = np.matmul(self.fun_m(X).transpose(2, 0, 1), trunc_V.transpose(2, 0, 1))
        # XVS_hat = XV_hat * _pinv_diag(trunc_S).transpose().reshape(self._n, 1, self._rrho.max())
        # XVS_hat = XVS_hat.transpose(1, 2, 0)
        # Y = XVS_hat[:, self._n_factors_order[1], self._n_factors_order[0]].copy()

        # X_transformed_0 = self._mprod(X, self._truncated_svdm.v)
        # X_transformed_0 = self._mprod(X_transformed_0, self._truncS_pinv)
        # X_transformed = self.fun_m(X_transformed_0)
        return Y

    # def _mode1_projector(self, X):
    #     truncU_mtranspose = tensor_mtranspose(self._truncated_svdm.u, self.fun_m, self.inv_m)
    #     X_transformed_0 = self._mprod(truncU_mtranspose, X)
    #     X_transformed_0 = tensor_mtranspose(self._mprod(self._truncS_pinv, X_transformed_0), self.fun_m, self.inv_m)
    #     X_transformed = self.fun_m(X_transformed_0)
    #     return self._mode1_reduce(X_transformed)

    def transform(self, X):
        """Apply mode-1 dimensionality reduction to X.

        X is projected on the first mode-1 tensor components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like of shape (m_samples, p_features, n_modes)
            Training data, where m_samples is the number of samples,
            p_features is the number of features and n_modes is the
            number of modes (timepoints/locations etc...)

        Returns
        -------
        X_new : array-like of shape (m_samples, `n_components_`)
            Projection of X in the first principal components, where m_samples
            is the number of samples and n_components is the number of the components.

        """
        _assert_order_and_mdim(X, 'X', 3, [(1, self._p), (2, self._n)])
        return self._mode0_projector(self._mdf.transform(X))

    @property
    def mode2_loadings(self):
        """ The weights driving the variation in each of the obtained factors with respect to
        each feature
        """

        return self._truncated_hat_svdm.v[:,self._n_factors_order[1], self._n_factors_order[0]].copy()

    def fit_transform(self, X: np.ndarray, y=None, **fit_params):

        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like of shape (m_samples, p_features, n_modes)
            Training data, where m_samples is the number of samples,
            p_features is the number of features and n_modes is the
            number of modes (timepoints/locations etc...)

        y : Ignored
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (m_samples, `n_components_`)
            Transformed values.

        """

        self.fit(X)
        return self.transform(X)

    # noinspection PyPep8Naming
    def inverse_transform(self, Y: NumpynDArray):
        """
        Inverts TCAM scores back to the original features space

        Parameters
        ----------
        Y: np.ndarray
            2d array with shape (k, `n_components_`)

        Returns
        -------
        Y_inv: NumpynDArray
            3rd order tensor that is the inverse transform of Y to the original features space

        """

        trunc_U, trunc_S, trunc_V = self._truncated_hat_svdm.astuple()

        # Suppose YY = X * V * pinv(S)
        # and the matrix Y is an ordering of YYs columns according to the factors order

        YY_hat = np.zeros((Y.shape[0], self._rrho.max(), self._n))
        YY_hat[:, self._n_factors_order[1], self._n_factors_order[0]] = Y.copy()
        # YYS_hat = YY_hat.transpose(2, 0, 1) * trunc_S.transpose().reshape(self._n, 1, self._rrho.max())
        X_hat = np.matmul(YY_hat.transpose(2, 0, 1), trunc_V.transpose(2, 1, 0)).transpose(1, 2, 0)
        XX = self.inv_m(X_hat)

        # Note that
        # YY*S*V' = X * V * pinv(S) * S * V'
        #         = X * V * (JJ) * V'
        #         = X * (V * JJ) * V'
        #         = X * (VV) * V'
        #         = X * (JJ) \approx X
        #
        # where JJ is "almost" the identity tensor


        # #################################### OLD CODE #################################################
        # YY_hat = np.zeros((trunc_U.shape[0], trunc_U.shape[1], trunc_U.shape[-1]))                    #
        # YY_hat[:, self._n_factors_order[1], self._n_factors_order[0]] = Y.copy()                      #
        # YY = self.inv_m(YY_hat)  # get YY from YY_hat                                                 #
        # YYs = self._mprod(YY, trunc_S)  # YY*S                                                        #
        # Yinv = self._mprod(YYs, tensor_mtranspose(trunc_V, self.fun_m, self.inv_m))  # YY*S*V'        #
        # # return self._mdf.inverse_transform(Yinv)                                                    #
        # ###############################################################################################

        return self._mdf.inverse_transform(XX)


