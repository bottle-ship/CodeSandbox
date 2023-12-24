import os

import pytest
import torch

from learning_kit.cv_splitter.random import train_test_random_split
from learning_kit.data.tensor_like import TensorLikeDataset


def test_train_test_random_split():
    n_splits = 3
    test_size = 0.1

    n_samples = 100
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    n_features_1 = 5
    n_features_2 = 3

    torch.manual_seed(0)
    x1 = torch.randn(n_samples, n_features_1)
    x2 = torch.randn(n_samples, n_features_2)
    y = torch.randint(0, 2, size=(n_samples,))

    x_dataset = TensorLikeDataset(x1, x2)
    y_dataset = TensorLikeDataset(y)

    cv_splits = train_test_random_split(x_dataset, y_dataset, n_splits=n_splits, test_size=test_size)
    cv_splits = [cv_split for cv_split in cv_splits]

    assert len(cv_splits) == n_splits
    for cv_split in cv_splits:
        x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = cv_split
        assert len(x_dataset_train) == len(y_dataset_train) == n_train
        assert len(x_dataset_test) == len(y_dataset_test) == n_test
        assert len(x_dataset_train[0]) == 2
        assert len(y_dataset_train[0]) == 1
        assert len(x_dataset_train[0][0]) == n_features_1
        assert len(x_dataset_train[0][1]) == n_features_2


if __name__ == "__main__":
    pytest.main(["-s", "-v", f"{os.path.abspath(__file__)}"])
