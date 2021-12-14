import numpy as np
from typing import List, Tuple, Dict, Mapping
from ._base import NumpynDArray
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from itertools import product


def table2tensor(table: pd.DataFrame, missing_flag: bool = False) -> Tuple[np.ma.core.MaskedArray, Mapping, Mapping]:
    """
    Reshapes a  `nm x p` (`(samples x reps) x features`) multi-indexed datafram to form a `m x p x n` tensor
    `(subjects, features, reps)`

    Parameters
    ----------
    table: pd.DataFrame
        a `nm x p` table of sampels x features

    missing_flag: `bool`, default = False
        When set to `False` (default), the function will raise an error in case there are missing samples.
        Setting to `True` will result in a tensor with masked entries.

    Returns
    -------
    tensor: ndarray, np.ma.array
        3'rd order tensor `m x p x n` (subjects, features, reps)

    mode1_mapping : dict
        The mapping of each mode1 (frontal) slice  index of the tensor to the table's original subject name

    mode3_mapping : dict
        The mapping of each mode3 (lateral) slice index of the tensor to the table's original rep id


    Examples
    --------
    Suppose that ``table_data`` is a dataframe with no missing values.

    >>> from mprod import table2tensor
    >>> import pandas as pd
    >>> np.random.seed(0)
    >>> table_data.iloc[:5,:4]
                        f1        f2        f3        f4
    SubjetID rep
    a        t1   0.251259  0.744838  -0.45889 -0.208525
             t10   2.39831  0.248772   0.65873   1.36994
             t2  -0.303154 -0.337603 -0.568608   -1.0239
             t3    1.36369  0.978895  0.161972 -0.804368
             t4     1.8548   1.52954   0.78576  0.538041
    >>> msk_tensor, mode1_mapping, mode3_mapping = table2tensor(table_data, missing_flag=False)
    >>> msk_tensor[:3,:3,:2]
    [[[0.25125853442243695 2.398308745102709]
      [0.7448378210349296 0.2487716728987871]
      [-0.4588901621837434 0.6587302072601999]]
     [[-0.5689263433318329 -0.06564253839123065]
      [1.0017636851038796 -0.49265853128383713]
      [0.45266517056628647 -1.4812390563653883]]
     [[0.7690616486878629 0.49302719962677855]
      [0.3186320585255899 1.469576084933633]
      [0.9609169837347897 -0.19564077520234632]]]
    >>> mode1_mapping
    {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    >>> mode3_mapping
    {'t1': 0,
     't10': 1,
     't2': 2,
     't3': 3,
     't4': 4,
     't5': 5,
     't6': 6,
     't7': 7,
     't8': 8,
     't9': 9}

    **missing values**

    >>> msk_tensor, mode1_mapping, mode3_mapping = table2tensor(table_data.sample(40)
    ...                                                          , missing_flag=True)
    >>> msk_tensor[:3,:3,:2]
    masked_array(
      data=[[[0.07664420134210018, --],
             [-0.7358062254334045, --],
             [0.5562074188402509, --]],
            [[2.088982483926928, -0.06564253839123065],
             [0.7697757466063808, -0.49265853128383713],
             [0.4147812728859107, -1.4812390563653883]],
            [[-0.004794963866429985, 1.2262908375944879],
             [-0.15033350807209261, -0.3068131758163276],
             [0.6461670563178799, 0.1769508046682527]]],
      mask=[[[False,  True],
             [False,  True],
             [False,  True]],
            [[False, False],
             [False, False],
             [False, False]],
            [[False, False],
             [False, False],
             [False, False]]], fill_value=0.0)
    >>> mode1_mapping
    {'a': 3, 'b': 1, 'c': 0, 'd': 4, 'e': 2}
    >>> mode3_mapping
    {'t1': 2,
     't10': 1,
     't2': 3,
     't3': 6,
     't4': 5,
     't5': 7,
     't6': 8,
     't7': 4,
     't8': 0,
     't9': 9}
    """

    samples_map, usamples = table.index.get_level_values(0).factorize()
    reps_map, ureps = table.index.get_level_values(1).factorize()

    m, p, n = usamples.size, table.shape[1], ureps.size

    samples_map_dict = pd.Series(np.unique(samples_map), usamples).to_dict()
    reps_map_dict = pd.Series(np.unique(reps_map), ureps).to_dict()

    if missing_flag:
        tensor = np.ma.array(np.zeros((m, p, n)), mask=np.ones((m, p, n)), fill_value=0)
        index_iterator = table.iterrows()
    else:
        tensor = np.zeros((m, p, n))
        index_iterator = (((i, j), table.loc[(i, j)].copy()) for i, j in product(usamples, ureps))

    try:
        for (m1, m3), val in index_iterator:
            tensor[samples_map_dict[m1], :, reps_map_dict[m3]] = val.values
    except KeyError as ke:
        raise KeyError("Discovered missing data in the table, which is not allowed by default. "
                       "To work with missing data and have a masked array returned, set missing_flag to True")

    return tensor, samples_map_dict, reps_map_dict


# noinspection PyPep8Naming
# noinspection PyUnusedLocal
class MeanDeviationForm(TransformerMixin, BaseEstimator):
    """Standardize the data by subtracting the mean (or empiric mean) sample
    The mean deviation form of a tensor :math:`X \\in \mathbb{R}^{m \\times p \\times n}` is calculated as:

            Z = X - U

    where `U` is the mean sample of `X` , calculated as follows:

    .. math::
        U = \\frac{1}{m} \\sum_{i=1}^{m} X[i,:,:]

    and for the empiric mean deviation form:

    .. math::
        U = \\frac{1}{m-1} \\sum_{i=1}^{m} X[i,:,:]


    Attributes
    ----------
    _mean_sample : ndarray of shape (p_features, n_repeats), or `None`
        The mean sample of the dataset


    Methods
    -------
    fit:
        Fits a MeanDeviationForm transformer by computing the mean sample of a training dataset
    transform:
        Shift dataset by fitted sample mean
    fit_transform:
        Compute the mean sample of a dataset and transform it to its mean deviation form
    inverse_transform:
        Add precomputed mean sample to a dataset




    """

    def __init__(self):
        # super(MeanDeviationForm, self).__init__()

        self._mean_sample = None

    def _fit(self, X):
        denum = X.shape[0]
        self._mean_sample = np.nansum(X, axis=0, keepdims=True) / denum

    def fit(self, X, y=None, **fit_param):
        """Compute the mean (or empiric mean) sample of a tensor

        Parameters
        ----------
        X : {array-like} of shape (m_samples, p_features, n_repeats)
            The data used to compute the mean sample
            used for later cantering along the features-repeats axes.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted MeanDeviationForm object

        Examples
        --------
        >>> from mprod import MeanDeviationForm
        >>> import numpy as np
        >>> X = np.random.randn(10,20,4)
        >>> mdf = MeanDeviationForm()
        >>> mdf = mdf.fit(X)
        """
        self._fit(X)
        return self

    def transform(self, X, y=None):
        """Perform standardization by centering.

        Parameters
        ----------
        X : array-like of shape (k_samples, p_features, n_repeats)
            The data used to center along the features-repeats axes.

        Returns
        -------
        X_tr : ndarray of shape (k_samples, p_features, n_repeats)
            Transformed tensor.

        Examples
        --------
        >>> from mprod import MeanDeviationForm
        >>> import numpy as np
        >>> X = np.random.randn(10,20,4)
        >>> y = np.random.randn(50,20,4)
        >>> mdf = MeanDeviationForm()
        >>> mdf_fit = mdf.fit(X)
        >>> yt = mdf.transform(yt)

        """

        X_transform = X - self._mean_sample
        if type(X_transform) == np.ma.core.MaskedArray:
            return X_transform.filled().data
        else:
            return X_transform

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def inverse_transform(self, Y):
        """Undo the centering of X according to mean sample.

        Parameters
        ----------
        X : array-like of shape (m_samples, p_features, n_repeats)
            Input data that will be transformed.

        Returns
        -------
        Xt : ndarray of shape (m_samples, p_features, n_repeats)
            Transformed data.

        Examples
        --------
        >>> from mprod import MeanDeviationForm
        >>> import numpy as np
        >>> X = np.random.randn(10,20,4)
        >>> mdf = MeanDeviationForm()
        >>> Xt = mdf.fit_transform(X)
        >>> mdf.inverse_transform(Xt) - X

        """
        Y_transform = Y + self._mean_sample

        if type(Y) == np.ma.core.MaskedArray:
            return Y_transform.filled().data
        else:
            return Y_transform

