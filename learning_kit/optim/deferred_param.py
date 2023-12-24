import typing as t

import torch.optim as optim


class DeferredParamOptimizer(object):

    def __init__(self, optimizer_cls: t.Type[optim.Optimizer], **kwargs):
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = kwargs

    def configure_optimizer(self, params) -> optim.Optimizer:
        return self.optimizer_cls(params, **self.optimizer_args)
