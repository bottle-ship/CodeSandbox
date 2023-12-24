import os

import numpy as np
import pytest
import torch

from learning_kit.cv_splitter.resample import train_test_resample_split
from learning_kit.data.tensor_like import TensorLikeDataset


def test_train_test_resample_split():
    n_splits = 3
    train_size = 0.9

    n_samples = 100
    n_train = int(n_samples * train_size)
    n_features_1 = 5
    n_features_2 = 3

    torch.manual_seed(0)
    x1 = torch.randn(n_samples, n_features_1)
    x2 = torch.randn(n_samples, n_features_2)
    y = torch.randint(0, 2, size=(n_samples,))

    x_dataset = TensorLikeDataset(x1, x2)
    y_dataset = TensorLikeDataset(y)

    cv_splits = train_test_resample_split(x_dataset, y_dataset, n_splits=n_splits, train_size=train_size)
    cv_splits = [cv_split for cv_split in cv_splits]

    assert len(cv_splits) == n_splits
    for cv_split in cv_splits:
        x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = cv_split
        assert len(x_dataset_train) == len(y_dataset_train) == n_train
        assert len(x_dataset_test) == len(y_dataset_test)
        assert len(x_dataset_train[0]) == 2
        assert len(y_dataset_train[0]) == 1
        assert len(x_dataset_train[0][0]) == n_features_1
        assert len(x_dataset_train[0][1]) == n_features_2
        assert len(x_dataset_train.indices) + len(x_dataset_test.indices) > n_samples
        assert len(np.unique(np.concatenate([x_dataset_train.indices, x_dataset_test.indices]))) == n_samples


if __name__ == "__main__":
    pytest.main(["-s", "-v", f"{os.path.abspath(__file__)}"])
