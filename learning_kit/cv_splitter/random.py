import typing as t
from itertools import chain

import numpy as np
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit
)
from sklearn.utils.validation import indexable
from torch.utils.data import (
    Dataset,
    Subset
)

__all__ = ["train_test_random_split"]

T = t.TypeVar('T')


def train_test_random_split(
        *datasets: Dataset[T],
        n_splits: int = 5,
        test_size: t.Optional[t.Union[int, float]] = None,
        train_size: t.Optional[t.Union[int, float]] = None,
        stratify: t.Optional[np.ndarray] = None,
        random_state: t.Optional[t.Union[int, np.random.RandomState]] = None
) -> t.List[Subset[T]]:
    r"""Split datasets into random train and test subsets.

    This function divides one or more datasets into random train and test subsets for cross-validation.

    Parameters
    ----------
    datasets : sequence of torch.utils.data.Dataset
        One or more datasets to be split into train and test subsets.

    n_splits : int, default=5
        Number of splits to create for the train-test separation.

    test_size : float or int, optional, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.

    train_size : float or int, optional, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    stratify : np.ndarray, optional, default=None
        Array containing class labels or grouping information to be used for stratified sampling.
        This helps maintain the same distribution of classes/groups in train and test splits.

    random_state : int or np.random.RandomState instance, optional, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    Yields
    ------
    cv_split : list of torch.utils.data.Subset
        List containing tuples of train and test Subset pairs for each dataset split.

    Examples
    --------
    >>> import torch
    >>> from learning_kit.cv_splitter.random import train_test_random_split
    >>> from learning_kit.data.tensor_like import TensorLikeDataset
    >>> x1 = torch.randn(100, 5)
    >>> x2 = torch.randn(100, 3)
    >>> y = torch.randint(0, 2, size=(100,))
    >>> x_dataset = TensorLikeDataset(x1, x2)
    >>> y_dataset = TensorLikeDataset(y)
    >>> cv_splits = train_test_random_split(x_dataset, y_dataset, n_splits=3, test_size=0.1)
    >>> for i, cv_split in enumerate(cv_splits):
    ...    x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = cv_split
    ...    print(f"CV {i}")
    CV 0
    CV 1
    CV 2

    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset required as input.")

    datasets = indexable(*datasets)

    if stratify is not None:
        cv_cls = StratifiedShuffleSplit
    else:
        cv_cls = ShuffleSplit
    cv = cv_cls(n_splits, test_size=test_size, train_size=train_size, random_state=random_state)

    for train_indices, test_indices in cv.split(X=datasets[0], y=stratify):
        yield list(
            chain.from_iterable(
                [(Subset(d, indices=train_indices), Subset(d, indices=test_indices)) for d in datasets]
            )
        )
