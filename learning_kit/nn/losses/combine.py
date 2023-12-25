import typing as t

import numpy as np
import torch
import torch.nn as nn

__all__ = ["CombineLoss"]


class CombineLoss(nn.Module):
    r"""Combine loss module that combines multiple loss functions.

    This module combines multiple loss functions to compute a joint loss by applying each
    individual loss function to corresponding pairs of predicted and true tensors. It offers
    the flexibility to assign optional weights to each loss function.

    Parameters
    ----------
    loss_modules : list/tuple of torch.nn.Module or torch.nn.ModuleList
        A list, tuple, or torch.nn.ModuleList containing individual loss modules to be combined.

    weight : np.ndarray or torch.Tensor, optional, default=None
        A weights to apply to each loss module. If None, equal weights are assigned.

    Attributes
    ----------
    loss_modules : nn.ModuleList
        A list, tuple, or ModuleList containing individual loss modules.

    weight : torch.Tensor
        Tensor containing the weights assigned to each loss module.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from learning_kit.nn.losses.combine import CombineLoss
    >>> torch.set_default_dtype(torch.float32)
    >>> loss_module_1 = nn.L1Loss()
    >>> loss_module_2 = nn.MSELoss()
    >>> combined_loss = CombineLoss([loss_module_1, loss_module_2])
    >>> y_pred = (torch.tensor([1.0]), torch.tensor([2.0]))
    >>> y_true = (torch.tensor([0.5]), torch.tensor([2.5]))
    >>> total_loss = combined_loss(y_pred, y_true)
    >>> print(total_loss)
    tensor(0.3750)

    """
    loss_modules: nn.ModuleList
    weight: torch.Tensor

    def __init__(
            self,
            loss_modules: t.Union[t.List[nn.Module], t.Tuple[nn.Module], nn.ModuleList],
            weight: t.Optional[t.Union[np.ndarray, torch.Tensor]] = None
    ):
        super(CombineLoss, self).__init__()

        self.loss_modules = loss_modules if isinstance(loss_modules, nn.ModuleList) else nn.ModuleList(loss_modules)
        self.weight = torch.ones((len(loss_modules),)) if weight is None else weight
        self.weight = self.weight if torch.is_tensor(self.weight) else torch.tensor(self.weight)

    def forward(self, y_pred: t.Tuple[torch.Tensor, ...], y_true: t.Tuple[torch.Tensor, ...]) -> torch.Tensor:
        loss = torch.stack(
            [self.loss_modules[i](yi_pred, yi_true) for i, (yi_pred, yi_true) in enumerate(zip(y_pred, y_true))],
            dim=-1
        )
        loss = torch.mean(loss * self.weight.to(device=loss.device, dtype=loss.dtype))

        return loss
