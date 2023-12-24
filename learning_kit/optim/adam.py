import typing as t

import torch.optim as optim

from .deferred_param import DeferredParamOptimizer


class Adam(DeferredParamOptimizer):

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: t.Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False,
                 differentiable: bool = False, fused: t.Optional[bool] = None):
        optimizer_kwargs = locals()
        optimizer_kwargs.pop("self")
        optimizer_kwargs.pop("__class__")
        super(Adam, self).__init__(optimizer_cls=optim.Adam, **optimizer_kwargs)
