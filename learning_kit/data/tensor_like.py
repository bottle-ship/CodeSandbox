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
    >>> torch.manual_seed(0)
    >>> torch.set_default_dtype(torch.float32)
    >>> x = torch.randn(100, 5)
    >>> y = torch.randint(0, 2, size=(100,))
    >>> x_dataset_torch = TensorLikeDataset(x)
    >>> print(x_dataset_torch[0])
    tensor([-1.1258, -1.1524, -0.2506, -0.4339,  0.8487])
    >>> xy_dataset_torch = TensorLikeDataset(x, y)
    >>> print(xy_dataset_torch[0])
    (tensor([-1.1258, -1.1524, -0.2506, -0.4339,  0.8487]), tensor(0))
    >>> x_numpy = x.detach().numpy()
    >>> y_numpy = y.detach().numpy()
    >>> xy_dataset_numpy = TensorLikeDataset(x_numpy, y_numpy)
    >>> print(xy_dataset_numpy[0])
    (tensor([-1.1258, -1.1524, -0.2506, -0.4339,  0.8487]), tensor(0))

    """

    def __init__(self, *tensors: t.Union[np.ndarray, torch.Tensor]):
        super(TensorLikeDataset, self).__init__(
            *tuple([torch.tensor(tensor) if not torch.is_tensor(tensor) else tensor for tensor in tensors])
        )

    def __getitem__(self, index):
        tensors = tuple(tensor[index] for tensor in self.tensors)
        return tensors[0] if len(self.tensors) == 1 else tensors
