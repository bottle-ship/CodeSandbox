import numpy as np
import torch

from .tensor_like import TensorLikeDataset

__all__ = ["DummyDataset"]


class DummyDataset(TensorLikeDataset):
    r"""Dataset as a fill with NaN values.

    This dataset is created to generate a dummy dataset containing a specified number of samples,
    each initialized with a shape of `(n_samples,)` fill with NaN values.

    Parameters
    ----------
    n_samples : int
        Number of samples to be generated for the dataset.

    Examples
    --------
    >>> import torch
    >>> from learning_kit.data.dummy import DummyDataset
    >>> dataset = DummyDataset(n_samples=100)
    >>> len(dataset)
    100
    >>> print(dataset[0])
    tensor(nan, dtype=torch.float64)

    """

    def __init__(self, n_samples: int):
        super(DummyDataset, self).__init__(torch.tensor(np.ones((n_samples,)) * np.nan))
