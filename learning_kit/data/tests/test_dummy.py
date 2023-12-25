import os

import pytest
import torch

from learning_kit.data.dummy import DummyDataset

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)


def test_dummy_dataset():
    n_samples = 100

    dataset = DummyDataset(n_samples=n_samples)
    assert len(dataset) == n_samples
    assert torch.isnan(dataset[0]).item()


if __name__ == "__main__":
    pytest.main(["-s", "-v", f"{os.path.abspath(__file__)}"])
