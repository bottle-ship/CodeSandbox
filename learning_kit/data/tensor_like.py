import typing as t

import numpy as np
import torch
from torch.utils import data

__all__ = ["TensorLikeDataset"]


class TensorLikeDataset(data.TensorDataset):
    r"""Dataset wrapping tensor-likes.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Parameters
    ----------
    *tensors : np.ndarray or torch.Tensor
        Tensor-likes that have the same size of the first dimension.

    Attributes
    ----------
    tensors : tuple of torch.Tensor
        A tuple containing tensor-like objects. Each object must have the same size
        along the first dimension to be used as samples in the dataset.

    Notes
    -----
    This dataset class extends `torch.utils.data.TensorDataset` to handle both
    NumPy arrays and PyTorch tensors as input data. It ensures that all provided
    tensors are converted to PyTorch tensors if they are not already.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from learning_kit.data.tensor_like import TensorLikeDataset
    >>> np.random.seed(0)
    >>> x = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=(100,))
    >>> dataset_numpy = TensorLikeDataset(x, y)
    >>> print(dataset_numpy[0])
    (tensor([0.5488, 0.7152, 0.6028, 0.5449, 0.4237], dtype=torch.float64), tensor(1, dtype=torch.int32))
    >>> x_torch = torch.tensor(x)
    >>> y_torch = torch.tensor(y)
    >>> dataset_torch = TensorLikeDataset(x_torch, y_torch)
    >>> print(dataset_numpy[0])
    (tensor([0.5488, 0.7152, 0.6028, 0.5449, 0.4237], dtype=torch.float64), tensor(1, dtype=torch.int32))

    """

    def __init__(self, *tensors: t.Union[np.ndarray, torch.Tensor]):
        super(TensorLikeDataset, self).__init__(
            *tuple([torch.tensor(tensor) if not torch.is_tensor(tensor) else tensor for tensor in tensors])
        )
