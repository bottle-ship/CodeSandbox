import typing as t

import torch.nn as nn

from learning_kit.nn.activation import ActivationFunction

__all__ = [
    "FullyConnectedBlock",
    "MultiFullyConnectedBlock"
]


class FullyConnectedBlock(nn.Sequential):
    r"""Creates a block for a fully connected neural network with customizable components.

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    bias : bool, default=True
        Whether the linear layer learns an additive bias.

    activation : str or torch.nn.Module, optional, default="GELU"
        Activation function to apply. If it's a string, it should be module name in `torch.nn.modules.activation`.

    activation_kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to activation function.
        This argument is effective only if ``activation`` is a string.

    batch_norm : bool, default=False
        Whether to apply batch normalization.

    batch_norm_eps : float, default=1e-5
        A value added to the denominator for numerical stability.
        This argument is effective only if ``batch_norm`` is ``True``.

    batch_norm_momentum : float, default=0.1
        The value used for the running_mean and running_var computation.
        Can be set to ``None`` for cumulative moving average (i.e. simple average).
        This argument is effective only if ``batch_norm`` is ``True``.

    batch_norm_affine : bool, default=True
        Whether to have learnable affine parameters in the batch normalization layer.
        This argument is effective only if ``batch_norm`` is ``True``.

    batch_norm_track_running_stats : bool, default=True
        Whether to track the running mean and variance.
        This argument is effective only if ``batch_norm`` is ``True``.

    dropout_probability : float, default=0.0
        The probability for an element to be zeroed by dropout.
        If `dropout_probability` is greater than 0, dropout is applied to the block.

    dropout_inplace : bool, default=False
        Whether to perform operation in-place in the dropout layer.

    Examples
    --------
    >>> from learning_kit.nn.fc_block import FullyConnectedBlock
    >>> block = FullyConnectedBlock(
    >>>     in_features=128, out_features=64, activation="ReLU", batch_norm=True, dropout_probability=0.2
    >>> )

    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation: t.Optional[t.Union[str, nn.Module]] = "GELU",
            activation_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
            batch_norm: bool = False,
            batch_norm_eps: float = 1e-5,
            batch_norm_momentum: float = 0.1,
            batch_norm_affine: bool = True,
            batch_norm_track_running_stats: bool = True,
            dropout_probability: float = 0.0,
            dropout_inplace: bool = False
    ):
        super(FullyConnectedBlock, self).__init__()

        self.append(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))
        if batch_norm:
            self.append(
                nn.BatchNorm1d(
                    num_features=out_features,
                    eps=batch_norm_eps,
                    momentum=batch_norm_momentum,
                    affine=batch_norm_affine,
                    track_running_stats=batch_norm_track_running_stats
                )
            )
        if activation is not None:
            if isinstance(activation, str):
                activation_kwargs = dict() if activation_kwargs is None else activation_kwargs
                self.append(ActivationFunction(activation_name=activation, **activation_kwargs))
            else:
                self.append(activation)
        if dropout_probability > 0:
            self.append(nn.Dropout(p=dropout_probability, inplace=dropout_inplace))


class MultiFullyConnectedBlock(nn.Sequential):
    r"""A sequence of FullyConnectedBlock instances forming a multi-layer fully connected neural network.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_layer_sizes : list of int or tuple of int
        The ith element represents the number of neurons in the ith hidden layer.

    bias : bool, default=True
        Whether the linear layer learns an additive bias.

    activation : str or torch.nn.Module, optional, default="GELU"
        Activation function to apply. If it's a string, it should be module name in `torch.nn.modules.activation`.

    activation_kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to activation function.
        This argument is effective only if ``activation`` is a string.

    batch_norm : bool, default=False
        Whether to apply batch normalization.

    batch_norm_eps : float, default=1e-5
        A value added to the denominator for numerical stability.
        This argument is effective only if ``batch_norm`` is ``True``.

    batch_norm_momentum : float, default=0.1
        The value used for the running_mean and running_var computation.
        Can be set to ``None`` for cumulative moving average (i.e. simple average).
        This argument is effective only if ``batch_norm`` is ``True``.

    batch_norm_affine : bool, default=True
        Whether to have learnable affine parameters in the batch normalization layer.
        This argument is effective only if ``batch_norm`` is ``True``.

    batch_norm_track_running_stats : bool, default=True
        Whether to track the running mean and variance.
        This argument is effective only if ``batch_norm`` is ``True``.

    dropout_probability : float, default=0.0
        The probability for an element to be zeroed by dropout.
        If `dropout_probability` is greater than 0, dropout is applied to the block.

    dropout_inplace : bool, default=False
        Whether to perform operation in-place in the dropout layer.

    """

    def __init__(
            self,
            in_features: int,
            hidden_layer_sizes: t.Union[t.Tuple[int, ...], t.List[int]],
            bias: bool = True,
            activation: t.Optional[t.Union[str, nn.Module]] = "GELU",
            activation_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
            batch_norm: bool = False,
            batch_norm_eps: float = 1e-5,
            batch_norm_momentum: float = 0.1,
            batch_norm_affine: bool = True,
            batch_norm_track_running_stats: bool = True,
            dropout_probability: float = 0.0,
            dropout_inplace: bool = False
    ):
        super(MultiFullyConnectedBlock, self).__init__()

        self.in_features = in_features
        self.hidden_layer_sizes = hidden_layer_sizes

        hidden_layer_sizes = (self.in_features, ) + tuple(self.hidden_layer_sizes)
        for i in range(0, len(self.hidden_layer_sizes)):
            self.append(
                FullyConnectedBlock(
                    in_features=hidden_layer_sizes[i],
                    out_features=hidden_layer_sizes[i + 1],
                    bias=bias,
                    activation=activation,
                    activation_kwargs=activation_kwargs,
                    batch_norm=batch_norm,
                    batch_norm_eps=batch_norm_eps,
                    batch_norm_momentum=batch_norm_momentum,
                    batch_norm_affine=batch_norm_affine,
                    batch_norm_track_running_stats=batch_norm_track_running_stats,
                    dropout_probability=dropout_probability,
                    dropout_inplace=dropout_inplace
                )
            )
