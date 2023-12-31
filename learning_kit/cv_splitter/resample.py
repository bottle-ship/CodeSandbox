import typing as t
from itertools import chain

import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import resample
from sklearn.utils.validation import indexable
from torch.utils.data import (
    Dataset,
    Subset
)

__all__ = ["train_test_resample_split"]

T = t.TypeVar('T')


def train_test_resample_split(
        *datasets: Dataset[T],
        n_splits: int = 5,
        train_size: t.Optional[t.Union[int, float]] = None,
        stratify: t.Optional[np.ndarray] = None,
        random_state: t.Optional[t.Union[int, np.random.RandomState]] = None
) -> t.List[t.Iterable[Subset[T]]]:
    r"""Split datasets into resample train and test subsets.

    This function divides one or more datasets into resample train and test subsets for cross-validation.

    Parameters
    ----------
    datasets : sequence of torch.utils.data.Dataset
        One or more datasets to be split into train and test subsets.

    n_splits : int, default=5
        Number of splits to create for the train-test separation.

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
    >>> from learning_kit.cv_splitter.resample import train_test_resample_split
    >>> from learning_kit.data.tensor_like import TensorLikeDataset
    >>> x = torch.arange(12)
    >>> x
    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    >>> y = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y
    tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> x_dataset = TensorLikeDataset(x)
    >>> y_dataset = TensorLikeDataset(y)
    >>> cv_splits = train_test_resample_split(x_dataset, y_dataset, n_splits=3, train_size=0.5, random_state=0)
    >>> for i, cv_split in enumerate(cv_splits):
    ...    x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = cv_split
    ...    print(f"CV {i}:")
    ...    print(f"  x_dataset_train={x_dataset_train[...]}")
    ...    print(f"  x_dataset_test={x_dataset_test[...]}")
    ...    print(f"  y_dataset_train={y_dataset_train[...]}")
    ...    print(f"  y_dataset_test={y_dataset_test[...]}")
    CV 0:
      x_dataset_train=tensor([ 5,  0,  3, 11,  3,  7])
      x_dataset_test=tensor([ 1,  2,  4,  6,  8,  9, 10])
      y_dataset_train=tensor([0, 0, 0, 1, 0, 1])
      y_dataset_test=tensor([0, 0, 0, 1, 1, 1, 1])
    CV 1:
      x_dataset_train=tensor([ 5, 11,  8,  9, 11,  5])
      x_dataset_test=tensor([ 0,  1,  2,  3,  4,  6,  7, 10])
      y_dataset_train=tensor([0, 1, 1, 1, 1, 0])
      y_dataset_test=tensor([0, 0, 0, 0, 0, 1, 1, 1])
    CV 2:
      x_dataset_train=tensor([ 8,  8,  6, 11,  2, 11])
      x_dataset_test=tensor([ 0,  1,  3,  4,  5,  7,  9, 10])
      y_dataset_train=tensor([1, 1, 1, 1, 0, 1])
      y_dataset_test=tensor([0, 0, 0, 0, 0, 1, 1, 1])
    >>> # Stratify resample split
    >>> stratify = y.detach().numpy()  # noqa
    >>> cv_splits = train_test_resample_split(
    ...     x_dataset, y_dataset, n_splits=3, train_size=0.5, stratify=stratify, random_state=0
    ... )
    >>> for i, cv_split in enumerate(cv_splits):
    ...    x_dataset_train, x_dataset_test, y_dataset_train, y_dataset_test = cv_split
    ...    print(f"CV {i}:")
    ...    print(f"  x_dataset_train={x_dataset_train[...]}")
    ...    print(f"  x_dataset_test={x_dataset_test[...]}")
    ...    print(f"  y_dataset_train={y_dataset_train[...]}")
    ...    print(f"  y_dataset_test={y_dataset_test[...]}")
    CV 0:
      x_dataset_train=tensor([9, 4, 0, 9, 9, 5])
      x_dataset_test=tensor([ 1,  2,  3,  6,  7,  8, 10, 11])
      y_dataset_train=tensor([1, 0, 0, 1, 1, 0])
      y_dataset_test=tensor([0, 0, 0, 1, 1, 1, 1, 1])
    CV 1:
      x_dataset_train=tensor([4, 6, 3, 7, 5, 9])
      x_dataset_test=tensor([ 0,  1,  2,  8, 10, 11])
      y_dataset_train=tensor([0, 1, 0, 1, 0, 1])
      y_dataset_test=tensor([0, 0, 0, 1, 1, 1])
    CV 2:
      x_dataset_train=tensor([9, 8, 9, 5, 0, 0])
      x_dataset_test=tensor([ 1,  2,  3,  4,  6,  7, 10, 11])
      y_dataset_train=tensor([1, 1, 1, 0, 0, 0])
      y_dataset_test=tensor([0, 0, 0, 0, 1, 1, 1, 1])

    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset required as input.")

    datasets = indexable(*datasets)
    indices = np.arange(0, len(datasets[0]))

    cv = ShuffleSplit(1, test_size=None, train_size=train_size, random_state=random_state)
    n_train = len(next(cv.split(X=datasets[0]))[0])

    train_indices_list = list()
    test_indices_list = list()
    for n in range(0, n_splits):
        if isinstance(random_state, int):
            local_random_state = random_state + n
        elif isinstance(random_state, np.random.RandomState):
            seed = tuple(random_state.get_state())[1]
            seed = seed + np.ones_like(seed) * n
            local_random_state = np.random.RandomState(seed=seed)
        else:
            local_random_state = None

        train_indices_list.append(
            resample(
                indices.copy(), replace=True, n_samples=n_train, stratify=stratify, random_state=local_random_state
            )
        )
        test_indices_list.append(np.delete(indices.copy(), np.unique(train_indices_list[-1])))

    return [
        list(
            chain.from_iterable(
                (Subset(dataset, indices=train_indices), Subset(dataset, indices=test_indices))
                for dataset in datasets
            )
        )
        for train_indices, test_indices in zip(train_indices_list, test_indices_list)
    ]
