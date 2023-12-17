import torch
import torch.nn as nn


class ActivationFunction(nn.Module):
    r"""Activation Function Module Wrapper.

    This class serves as a wrapper to apply various activation functions available in `torch.nn.modules.activation`.

    Parameters
    ----------
    activation_name : str
        Name of the activation function to be used. It should correspond to one of the supported
        activation functions available in `torch.nn.modules.activation`.

    **kwargs : dict
        Additional keyword arguments that can be passed to the chosen activation function.

    Attributes
    ----------
    activation_func : torch.nn.Module
        Instance of the selected activation function module.

    Raises
    ------
    ValueError
        If an unsupported or invalid activation function name is provided.

    Examples
    --------
    Instantiate an instance of ActivationFunction using ReLU activation:
    >>> activation_module = ActivationFunction('ReLU')  # Instantiate using ReLU activation
    >>> input_tensor = torch.randn(1, 10)  # Example input tensor
    >>> output = activation_module(input_tensor)  # Applying ReLU activation to input tensor

    """

    def __init__(self, activation_name: str, **kwargs):
        super(ActivationFunction, self).__init__()

        if activation_name not in nn.modules.activation.__all__:
            raise ValueError(
                f"Invalid activation function '{activation_name}'. Please provide a valid activation function. "
                f"Supported activation functions: {', '.join(nn.modules.activation.__all__)}.")

        self.activation_func = getattr(nn, activation_name)(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the ActivationFunction module.

        Applies the chosen activation function to the input tensor `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to which the activation function will be applied.

        Returns
        -------
        torch.Tensor
            Tensor after applying the activation function to the input `x`.

        """
        return self.activation_func(x)
