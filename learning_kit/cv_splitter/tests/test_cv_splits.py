import os
import typing as t

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from learning_kit.cv_splitter.kfold import train_test_kfold_split
from learning_kit.cv_splitter.random import train_test_random_split
from learning_kit.cv_splitter.resample import train_test_resample_split
from learning_kit.data.tensor_like import TensorLikeDataset

T = t.TypeVar('T')


def _generate_datasets(
        n_samples: int,
        n_features_1: int,
        n_features_2: int
) -> t.Tuple[TensorLikeDataset, TensorLikeDataset]:
    x1 = torch.arange(n_samples * n_features_1).reshape(n_samples, n_features_1)
    x2 = torch.arange(n_samples * n_features_2).reshape(n_samples, n_features_2)
    y = torch.arange(n_samples)

    x_dataset = TensorLikeDataset(x1, x2)
    y_dataset = TensorLikeDataset(y)

    return x_dataset, y_dataset


def _check_cv_coverage(
        cv_splits: t.List[t.Iterable[Subset[T]]],
        n_samples: int,
        n_features_1: int,
        n_features_2: int,
        expected_n_splits: int,
        expected_n_train: int,
        expected_n_test: t.Optional[int]
):
    if expected_n_test is not None:
        assert expected_n_train + expected_n_test == n_samples
    assert sum(1 for _ in cv_splits) == expected_n_splits

    for cv_split in cv_splits:
        x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = cv_split
        x1_train, x2_train = x_dataset_train[...]
        x1_test, x2_test = x_dataset_test[...]
        y_train = y_dataset_train[...]
        y_test = y_dataset_test[...]

        assert len(x1_train) == len(x2_train) == len(y_train) == expected_n_train
        if expected_n_test is None:
            assert len(x1_test) == len(x2_test) == len(y_test)

            x_train_indices = np.unique(x_dataset_train.indices)
            x_test_indices = np.unique(x_dataset_test.indices)
            y_train_indices = np.unique(y_dataset_train.indices)
            y_test_indices = np.unique(y_dataset_test.indices)

            assert len(x_train_indices) + len(x_test_indices) == n_samples
            assert len(y_train_indices) + len(y_test_indices) == n_samples
        else:
            assert len(x1_test) == len(x2_test) == len(y_test) == expected_n_test

            assert len(x1_train) + len(x1_test) == n_samples
            assert len(y_train) + len(y_test) == n_samples

        x1_cat = torch.cat([x1_train, x1_test], dim=0)
        x2_cat = torch.cat([x2_train, x2_test], dim=0)

        assert x1_cat.shape[1:] == (n_features_1,)
        assert x2_cat.shape[1:] == (n_features_2,)


@pytest.fixture
def setup() -> t.Tuple[int, int, int, int]:
    n_splits = 5
    n_samples = 10
    n_features_1 = 5
    n_features_2 = 3

    return n_splits, n_samples, n_features_1, n_features_2


def test_train_test_kfold_split(setup):
    n_splits, n_samples, n_features_1, n_features_2 = setup

    x_dataset, y_dataset = _generate_datasets(
        n_samples=n_samples, n_features_1=n_features_1, n_features_2=n_features_2
    )

    n_test = int(n_samples / n_splits)
    n_train = n_test * (n_splits - 1)

    cv_splits = train_test_kfold_split(x_dataset, y_dataset, n_splits=n_splits)
    _check_cv_coverage(
        cv_splits=cv_splits,
        n_samples=n_samples,
        n_features_1=n_features_1,
        n_features_2=n_features_2,
        expected_n_splits=n_splits,
        expected_n_train=n_train,
        expected_n_test=n_test
    )


def test_train_test_random_split(setup):
    n_splits, n_samples, n_features_1, n_features_2 = setup

    x_dataset, y_dataset = _generate_datasets(
        n_samples=n_samples, n_features_1=n_features_1, n_features_2=n_features_2
    )

    test_size = 0.2
    train_size = 1 - test_size
    n_train = int(n_samples * train_size)
    n_test = int(n_samples * test_size)

    cv_splits = train_test_random_split(x_dataset, y_dataset, n_splits=n_splits, test_size=test_size)
    _check_cv_coverage(
        cv_splits=cv_splits,
        n_samples=n_samples,
        n_features_1=n_features_1,
        n_features_2=n_features_2,
        expected_n_splits=n_splits,
        expected_n_train=n_train,
        expected_n_test=n_test
    )


def test_train_test_resample_split(setup):
    n_splits, n_samples, n_features_1, n_features_2 = setup

    x_dataset, y_dataset = _generate_datasets(
        n_samples=n_samples, n_features_1=n_features_1, n_features_2=n_features_2
    )

    train_size = 0.8
    n_train = int(n_samples * train_size)

    cv_splits = train_test_resample_split(x_dataset, y_dataset, n_splits=n_splits, train_size=train_size)
    _check_cv_coverage(
        cv_splits=cv_splits,
        n_samples=n_samples,
        n_features_1=n_features_1,
        n_features_2=n_features_2,
        expected_n_splits=n_splits,
        expected_n_train=n_train,
        expected_n_test=None
    )


@pytest.mark.parametrize("cv_func", [train_test_kfold_split, train_test_random_split, train_test_resample_split])
@pytest.mark.parametrize("random_state", [0, np.random.RandomState(seed=0)])
def test_split_random_state(setup, cv_func, random_state):
    n_splits, n_samples, n_features_1, n_features_2 = setup

    x_dataset, y_dataset = _generate_datasets(
        n_samples=n_samples, n_features_1=n_features_1, n_features_2=n_features_2
    )

    cv_func(x_dataset, y_dataset, random_state=random_state)


if __name__ == "__main__":
    pytest.main(["-s", "-v", f"{os.path.abspath(__file__)}"])
