import os

import numpy as np
import pytest
import torch

from learning_kit.data.tensor_like import TensorLikeDataset


def test_tensor_like_dataset_with_numpy_ndarray():
    # Generate random data for testing
    np.random.seed(0)
    x = np.random.rand(100, 5)
    y = np.random.randint(0, 2, size=(100,))

    # Create dataset using NumPy arrays
    dataset_numpy = TensorLikeDataset(x, y)

    # Test the length of the dataset
    assert len(dataset_numpy) == 100

    # Test the first sample in the dataset
    sample_x, sample_y = dataset_numpy[0]
    assert isinstance(sample_x, torch.Tensor)
    assert isinstance(sample_y, torch.Tensor)
    assert sample_x.shape == (5,)  # Check the shape of input tensor
    assert sample_y.shape == ()    # Check the shape of target tensor

    # Create dataset using PyTorch tensors
    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)
    dataset_torch = TensorLikeDataset(x_torch, y_torch)

    # Test the length of the dataset created using PyTorch tensors
    assert len(dataset_torch) == 100

    # Test the first sample in the PyTorch dataset
    sample_x_torch, sample_y_torch = dataset_torch[0]
    assert isinstance(sample_x_torch, torch.Tensor)
    assert isinstance(sample_y_torch, torch.Tensor)
    assert torch.allclose(sample_x_torch, sample_x)
    assert torch.allclose(sample_y_torch, sample_y)


def test_tensor_like_dataset_with_invalid_inputs():
    with pytest.raises(IndexError):
        len(TensorLikeDataset())


if __name__ == "__main__":
    pytest.main(["-s", "-v", f"{os.path.abspath(__file__)}"])
