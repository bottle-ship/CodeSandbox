import typing as t
from inspect import signature

import torch.nn as nn
import torch.optim as optim

__all__ = ["DeferredParamOptimizer"]


class DeferredParamOptimizer(object):
    r"""DeferredParamOptimizer allows deferred configuration of PyTorch optimizers with specified parameters.

    Parameters
    ----------
    cls : class of `torch.optim.Optimizer`
        A subclass of `torch.optim.Optimizer` representing the optimizer class to be configured.

    **kwargs
        Additional keyword arguments to be passed for configuring the optimizer.

    Raises
    ------
    KeyError
        If any provided keyword argument in `kwargs` is not a valid parameter for the specified optimizer class.

    Examples
    --------
    deferred_optimizer = DeferredParamOptimizer(cls=torch.optim.SGD, lr=0.01)
    optimizer = deferred_optimizer(model.parameters())

    """

    def __init__(self, cls: t.Type[optim.Optimizer], **kwargs):
        self.cls = cls
        self.kwargs = kwargs

        cls_parameters = signature(self.cls).parameters.keys()
        invalid_params = [key for key in kwargs if key not in cls_parameters]

        if len(invalid_params) > 0:
            raise KeyError(
                f"Invalid keyword argument(s) {', '.join(invalid_params)} for optimizer '{self.cls.__name__}'."
            )

    def __call__(self, params) -> optim.Optimizer:
        return self.configure_optimizer(params)

    def configure_optimizer(self, params: t.Iterator[nn.Parameter]) -> optim.Optimizer:
        r"""Configures and returns the optimizer using the specified parameters and keyword arguments.

        Parameters
        ----------
        params : iterator of `torch.nn.Parameter`
            Iterable of parameters to optimize or dicts defining parameter groups.

        Returns
        -------
        optimizer : `torch.optim.Optimizer`
            The configured optimizer.

        """
        return self.cls(params, **self.kwargs)
