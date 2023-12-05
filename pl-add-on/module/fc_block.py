import inspect
import typing as t

import torch
import torch.nn as nn


class FullyConnectedBlock(nn.Sequential):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation: t.Optional[str] = "GELU",
            batch_norm: bool = False,
            dropout_probability: float = 0.5,
            **kwargs
    ):
        super(FullyConnectedBlock, self).__init__()
        self.append(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))
        if batch_norm:
            self.append(nn.BatchNorm1d(out_features))
        if activation is not None:
            if activation not in nn.modules.activation.__all__:
                raise ValueError
            activation = getattr(nn, activation)
            activation_args = dict()
            for param_name in inspect.signature(activation).parameters:
                if param_name in kwargs:
                    activation_args[param_name] = kwargs[param_name]
                    del kwargs[param_name]
            self.append(activation(**activation_args))
        if dropout_probability > 0:
            self.append(nn.Dropout(p=dropout_probability))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(FullyConnectedBlock, self).forward(x)
