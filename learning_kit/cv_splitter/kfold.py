import typing as t
from itertools import chain

import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold
)
from sklearn.utils.validation import indexable
from torch.utils.data import (
    Dataset,
    Subset
)

__all__ = ["train_test_kfold_split"]

T = t.TypeVar('T')


def train_test_kfold_split(
        *datasets: Dataset[T],
        n_splits: int = 5,
        shuffle: bool = True,
        stratify: t.Optional[np.ndarray] = None,
        random_state: t.Optional[t.Union[int, np.random.RandomState]] = None
) -> t.List[t.Iterable[Subset[T]]]:
    r"""Split datasets into K-folds train and test subsets.

    This function divides one or more datasets into K-folds train and test subsets for cross-validation.

    Parameters
    ----------
    datasets : sequence of torch.utils.data.Dataset
        One or more datasets to be split into train and test subsets.

    n_splits : int, default=5
        Number of splits to create for the train-test separation.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.

    stratify : np.ndarray, optional, default=None
        Array containing class labels or grouping information to be used for stratified sampling.
        This helps maintain the same distribution of classes/groups in train and test splits.

    random_state : int or np.random.RandomState instance, optional, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    returns
    -------
    cv_splits : list of iterable of torch.utils.data.Subset
        List containing tuples of train and test Subset pairs for each dataset split.

    Examples
    --------
    >>> import torch
    >>> from learning_kit.cv_splitter.kfold import train_test_kfold_split
    >>> from learning_kit.data.tensor_like import TensorLikeDataset
    >>> x = torch.arange(12)
    >>> x
    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    >>> y = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y
    tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> x_dataset = TensorLikeDataset(x)
    >>> y_dataset = TensorLikeDataset(y)
    >>> cv_splits = train_test_kfold_split(x_dataset, y_dataset, n_splits=3, shuffle=False)
    >>> for i, cv_split in enumerate(cv_splits):
    ...    x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = cv_split
    ...    print(f"CV {i}:")
    ...    print(f"  x_dataset_train={x_dataset_train[...]}")
    ...    print(f"  x_dataset_test={x_dataset_test[...]}")
    ...    print(f"  y_dataset_train={y_dataset_train[...]}")
    ...    print(f"  y_dataset_test={y_dataset_test[...]}")
    CV 0:
      x_dataset_train=tensor([ 4,  5,  6,  7,  8,  9, 10, 11])
      x_dataset_test=tensor([0, 1, 2, 3])
      y_dataset_train=tensor([0, 0, 1, 1, 1, 1, 1, 1])
      y_dataset_test=tensor([0, 0, 0, 0])
    CV 1:
      x_dataset_train=tensor([ 0,  1,  2,  3,  8,  9, 10, 11])
      x_dataset_test=tensor([4, 5, 6, 7])
      y_dataset_train=tensor([0, 0, 0, 0, 1, 1, 1, 1])
      y_dataset_test=tensor([0, 0, 1, 1])
    CV 2:
      x_dataset_train=tensor([0, 1, 2, 3, 4, 5, 6, 7])
      x_dataset_test=tensor([ 8,  9, 10, 11])
      y_dataset_train=tensor([0, 0, 0, 0, 0, 0, 1, 1])
      y_dataset_test=tensor([1, 1, 1, 1])
    >>> # Stratify K-Folds
    >>> stratify = y.detach().numpy()  # noqa
    >>> cv_splits = train_test_kfold_split(x_dataset, y_dataset, n_splits=3, stratify=stratify, shuffle=False)
    >>> for i, cv_split in enumerate(cv_splits):
    ...    x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = cv_split
    ...    print(f"CV {i}:")
    ...    print(f"  x_dataset_train={x_dataset_train[...]}")
    ...    print(f"  x_dataset_test={x_dataset_test[...]}")
    ...    print(f"  y_dataset_train={y_dataset_train[...]}")
    ...    print(f"  y_dataset_test={y_dataset_test[...]}")
    CV 0:
      x_dataset_train=tensor([ 2,  3,  4,  5,  8,  9, 10, 11])
      x_dataset_test=tensor([0, 1, 6, 7])
      y_dataset_train=tensor([0, 0, 0, 0, 1, 1, 1, 1])
      y_dataset_test=tensor([0, 0, 1, 1])
    CV 1:
      x_dataset_train=tensor([ 0,  1,  4,  5,  6,  7, 10, 11])
      x_dataset_test=tensor([2, 3, 8, 9])
      y_dataset_train=tensor([0, 0, 0, 0, 1, 1, 1, 1])
      y_dataset_test=tensor([0, 0, 1, 1])
    CV 2:
      x_dataset_train=tensor([0, 1, 2, 3, 6, 7, 8, 9])
      x_dataset_test=tensor([ 4,  5, 10, 11])
      y_dataset_train=tensor([0, 0, 0, 0, 1, 1, 1, 1])
      y_dataset_test=tensor([0, 0, 1, 1])

    """
    n_datasets = len(datasets)
    if n_datasets == 0:
        raise ValueError("At least one dataset required as input.")

    datasets = indexable(*datasets)

    if stratify is not None:
        cv_cls = StratifiedKFold
    else:
        cv_cls = KFold
    cv = cv_cls(n_splits, shuffle=shuffle, random_state=random_state)

    return [
        list(
            chain.from_iterable(
                [(Subset(d, indices=train_indices), Subset(d, indices=test_indices)) for d in datasets]
            )
        )
        for train_indices, test_indices in cv.split(X=datasets[0], y=stratify)
    ]
