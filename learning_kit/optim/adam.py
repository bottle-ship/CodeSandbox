import typing as t

import torch.optim as optim

from .deferred_param import DeferredParamOptimizer


class Adam(DeferredParamOptimizer):

    def __init__(
            self,
            lr: float = 1e-3,
            betas: t.Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0,
            amsgrad: bool = False,
            *,
            foreach: t.Optional[bool] = None,
            maximize: bool = False,
            capturable: bool = False,
            differentiable: bool = False,
            fused: t.Optional[bool] = None
    ):
        kwargs = locals().copy()
        kwargs.pop("self")
        kwargs.pop("__class__")
        super(Adam, self).__init__(optim.Adam, **kwargs)
