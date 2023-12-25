import typing as t

import torch
from torch.utils.data import Dataset

from .dummy import DummyDataset
from .tensor_like import TensorLikeDataset

__all__ = ["XYDataset"]

T = t.TypeVar('T')


class XYDataset(Dataset[t.Tuple[torch.Tensor, torch.Tensor]]):
    r"""Dataset combining two datasets X and Y.

    Parameters
    ----------
    x : TensorLikeDataset
        Dataset containing input data `X`.

    y : TensorLikeDataset, optional, default=None
        Dataset containing target data `Y`. If `None`, a DummyDataset is created to
        provide dummy target values based on the length of `x`.

    Examples
    --------
    >>> import torch
    >>> from learning_kit.data.tensor_like import TensorLikeDataset
    >>> from learning_kit.data.xy_dataset import XYDataset
    >>> torch.manual_seed(0)
    >>> torch.set_default_dtype(torch.float32)
    >>> x_dataset = TensorLikeDataset(torch.rand((100, 2)))
    >>> y_dataset = TensorLikeDataset(torch.rand((100, 1)))
    >>> xy_dataset1 = XYDataset(x_dataset, y_dataset)
    >>> print(xy_dataset1[0])
    (tensor([0.4963, 0.7682]), tensor([0.5210]))
    >>> xy_dataset2 = XYDataset(x_dataset)
    >>> print(xy_dataset2[0])
    (tensor([0.4963, 0.7682]), tensor(nan, dtype=torch.float64))

    """
    x: TensorLikeDataset
    y: t.Optional[TensorLikeDataset]

    def __init__(self, x: TensorLikeDataset, y: t.Optional[TensorLikeDataset] = None):
        self.x = x
        self.y = y if y is not None else DummyDataset(n_samples=len(self.x))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
