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
) -> t.List[t.Iterable[Subset[T]]]:
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

    returns
    -------
    cv_split : list of iterable of torch.utils.data.Subset
        List containing tuples of train and test Subset pairs for each dataset split.

    Examples
    --------
    >>> import torch
    >>> from learning_kit.cv_splitter.random import train_test_random_split
    >>> from learning_kit.data.tensor_like import TensorLikeDataset
    >>> x = torch.arange(12)
    >>> x
    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    >>> y = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y
    tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> x_dataset = TensorLikeDataset(x)
    >>> y_dataset = TensorLikeDataset(y)
    >>> cv_splits = train_test_random_split(x_dataset, y_dataset, n_splits=3, test_size=0.5, random_state=0)
    >>> for i, cv_split in enumerate(cv_splits):
    ...    x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = cv_split
    ...    print(f"CV {i}:")
    ...    print(f"  x_dataset_train={x_dataset_train[...]}")
    ...    print(f"  x_dataset_test={x_dataset_test[...]}")
    ...    print(f"  y_dataset_train={y_dataset_train[...]}")
    ...    print(f"  y_dataset_test={y_dataset_test[...]}")
    CV 0:
      x_dataset_train=tensor([1, 7, 9, 3, 0, 5])
      x_dataset_test=tensor([ 6, 11,  4, 10,  2,  8])
      y_dataset_train=tensor([0, 1, 1, 0, 0, 0])
      y_dataset_test=tensor([1, 1, 0, 1, 0, 1])
    CV 1:
      x_dataset_train=tensor([11,  7,  6,  1, 10,  8])
      x_dataset_test=tensor([5, 2, 3, 4, 9, 0])
      y_dataset_train=tensor([1, 1, 1, 0, 1, 1])
      y_dataset_test=tensor([0, 0, 0, 0, 1, 0])
    CV 2:
      x_dataset_train=tensor([11,  0,  3,  4,  9,  8])
      x_dataset_test=tensor([ 6,  1, 10,  2,  7,  5])
      y_dataset_train=tensor([1, 0, 0, 0, 1, 1])
      y_dataset_test=tensor([1, 0, 1, 0, 1, 0])
    >>> # Specify train and test size
    >>> cv_splits = train_test_random_split(
    ...     x_dataset, y_dataset, n_splits=3, train_size=0.4, test_size=0.2, random_state=0
    ... )
    >>> for i, cv_split in enumerate(cv_splits):
    ...    x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = cv_split
    ...    print(f"CV {i}:")
    ...    print(f"  x_dataset_train={x_dataset_train[...]}")
    ...    print(f"  x_dataset_test={x_dataset_test[...]}")
    ...    print(f"  y_dataset_train={y_dataset_train[...]}")
    ...    print(f"  y_dataset_test={y_dataset_test[...]}")
    CV 0:
      x_dataset_train=tensor([10,  2,  8,  1])
      x_dataset_test=tensor([ 6, 11,  4])
      y_dataset_train=tensor([1, 0, 1, 0])
      y_dataset_test=tensor([1, 1, 0])
    CV 1:
      x_dataset_train=tensor([ 4,  9,  0, 11])
      x_dataset_test=tensor([5, 2, 3])
      y_dataset_train=tensor([0, 1, 0, 1])
      y_dataset_test=tensor([0, 0, 0])
    CV 2:
      x_dataset_train=tensor([ 2,  7,  5, 11])
      x_dataset_test=tensor([ 6,  1, 10])
      y_dataset_train=tensor([0, 1, 0, 1])
      y_dataset_test=tensor([1, 0, 1])
    >>> # Stratify shuffle split
    >>> stratify = y.detach().numpy()  # noqa
    >>> cv_splits = train_test_random_split(
    ...     x_dataset, y_dataset, n_splits=3, test_size=0.5, stratify=stratify, random_state=0
    ... )
    >>> for i, cv_split in enumerate(cv_splits):
    ...    x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = cv_split
    ...    print(f"CV {i}:")
    ...    print(f"  x_dataset_train={x_dataset_train[...]}")
    ...    print(f"  x_dataset_test={x_dataset_test[...]}")
    ...    print(f"  y_dataset_train={y_dataset_train[...]}")
    ...    print(f"  y_dataset_test={y_dataset_test[...]}")
    CV 0:
      x_dataset_train=tensor([ 7, 10,  2,  1,  9,  5])
      x_dataset_test=tensor([11,  4,  6,  8,  0,  3])
      y_dataset_train=tensor([1, 1, 0, 0, 1, 0])
      y_dataset_test=tensor([1, 0, 1, 1, 0, 0])
    CV 1:
      x_dataset_train=tensor([ 2,  4,  3,  7, 10,  9])
      x_dataset_test=tensor([11,  0,  8,  6,  1,  5])
      y_dataset_train=tensor([0, 0, 0, 1, 1, 1])
      y_dataset_test=tensor([1, 0, 1, 1, 0, 0])
    CV 2:
      x_dataset_train=tensor([7, 4, 1, 6, 3, 9])
      x_dataset_test=tensor([ 5,  8, 11,  0,  2, 10])
      y_dataset_train=tensor([1, 0, 0, 1, 0, 1])
      y_dataset_test=tensor([0, 1, 1, 0, 0, 1])

    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset required as input.")

    datasets = indexable(*datasets)

    if stratify is not None:
        cv_cls = StratifiedShuffleSplit
    else:
        cv_cls = ShuffleSplit
    cv = cv_cls(n_splits, test_size=test_size, train_size=train_size, random_state=random_state)

    return [
        list(
            chain.from_iterable(
                [(Subset(d, indices=train_indices), Subset(d, indices=test_indices)) for d in datasets]
            )
        )
        for train_indices, test_indices in cv.split(X=datasets[0], y=stratify)
    ]
