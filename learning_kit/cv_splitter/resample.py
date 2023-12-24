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
) -> t.List[Subset[T]]:
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

    Yields
    ------
    cv_split : list of torch.utils.data.Subset
        List containing tuples of train and test Subset pairs for each dataset split.

    Examples
    --------
    >>> import torch
    >>> from learning_kit.cv_splitter.resample import train_test_resample_split
    >>> from learning_kit.data.tensor_like import TensorLikeDataset
    >>> x1 = torch.randn(100, 5)
    >>> x2 = torch.randn(100, 3)
    >>> y = torch.randint(0, 2, size=(100,))
    >>> x_dataset = TensorLikeDataset(x1, x2)
    >>> y_dataset = TensorLikeDataset(y)
    >>> cv_splits = train_test_resample_split(x_dataset, y_dataset, n_splits=3, train_size=0.9)
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
    indices = np.arange(0, len(datasets[0]))

    cv = ShuffleSplit(1, test_size=None, train_size=train_size, random_state=random_state)
    n_train = len(next(cv.split(X=datasets[0]))[0])

    train_indices_list = list()
    test_indices_list = list()
    for n in range(0, n_splits):
        if isinstance(random_state, int):
            local_random_state = random_state + n
        elif isinstance(random_state, np.random.RandomState):
            seed = random_state.get_state()[1][0] + n
            local_random_state = np.random.RandomState(seed=seed)
        else:
            local_random_state = None

        train_indices_list.append(
            resample(
                indices.copy(), replace=True, n_samples=n_train, stratify=stratify, random_state=local_random_state
            )
        )
        test_indices_list.append(np.delete(indices.copy(), np.unique(train_indices_list[-1])))

    for train_indices, test_indices in zip(train_indices_list, test_indices_list):
        yield list(
            chain.from_iterable(
                (Subset(dataset, indices=train_indices), Subset(dataset, indices=test_indices))
                for dataset in datasets
            )
        )
