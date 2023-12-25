import os

import pytest
import torch

from learning_kit.data.tensor_like import TensorLikeDataset
from learning_kit.data.xy_dataset import XYDataset

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)


def test_xy_dataset():
    n_samples = 100

    x_dataset = TensorLikeDataset(torch.rand((n_samples, 2)))
    y_dataset = TensorLikeDataset(torch.rand((n_samples, 1)))
    xy_dataset1 = XYDataset(x_dataset, y_dataset)
    xy_dataset2 = XYDataset(x_dataset)

    assert len(xy_dataset1) == len(xy_dataset2) == n_samples
    assert not torch.isnan(xy_dataset1[0][1]).item()
    assert torch.isnan(xy_dataset2[0][1]).item()


if __name__ == "__main__":
    pytest.main(["-s", "-v", f"{os.path.abspath(__file__)}"])
