import typing as t

import pytorch_lightning as pl
import torch
import torch.nn as nn

from learning_kit.optim.deferred_param import DeferredParamOptimizer

__all__ = ["SingleModelTrainModule"]


class SingleModelTrainModule(pl.LightningModule):
    r"""TrainModule for training a single PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained.

    loss_func : callable
        The loss function to calculate the training loss.

    optimizer : torch.optim.Optimizer or DeferredParamOptimizer
        The optimizer to use for optimizing model parameters.

    Examples
    --------
    >>> import pytorch_lightning as pl  # noqa
    >>> import torch
    >>> import torch.nn as nn  # noqa
    >>> import torch.optim as optim  # noqa
    >>> from torch.utils.data import DataLoader
    >>> from learning_kit.data import TensorLikeDataset, XYDataset

    # Prepare the dataset
    >>> x = torch.randn(100, 5)
    >>> y = torch.randn(100, 1)
    >>> x_dataset = TensorLikeDataset(x)
    >>> y_dataset = TensorLikeDataset(y)
    >>> xy_dataset = XYDataset(x_dataset, y_dataset)
    >>> train_dataloader = DataLoader(xy_dataset, batch_size=64, shuffle=True)

    # Initialize the training module
    >>> model = nn.Linear(5, 1)
    >>> loss_function = nn.CrossEntropyLoss()
    >>> optimizer = optim.Adam(model.parameters(), lr=0.001)
    >>> trainer_module = SingleModelTrainModule(model, loss_function, optimizer)

    # Train the model
    >>> trainer = pl.Trainer()
    >>> trainer.fit(trainer_module, train_dataloader)

    """

    def __init__(
            self,
            model: nn.Module,
            loss_func: t.Callable[
                [
                    t.Union[torch.Tensor, t.Tuple[torch.Tensor, ...]],
                    t.Union[torch.Tensor, t.Tuple[torch.Tensor, ...]]
                ],
                torch.Tensor
            ],
            optimizer: t.Union[torch.optim.Optimizer, DeferredParamOptimizer]
    ):
        super(SingleModelTrainModule, self).__init__()
        self.save_hyperparameters()

        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer

    def forward(
            self,
            x: t.Union[torch.Tensor, t.Tuple[torch.Tensor, ...]]
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, ...]]:
        return self.model.forward(*x) if isinstance(x, (list, tuple)) else self.model.forward(x)

    def compute_loss(self, batch) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y, y_hat)

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, ...]]:
        x, y = batch
        y_hat = self.forward(x)

        return y_hat

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if isinstance(self.optimizer, DeferredParamOptimizer):
            return self.optimizer(params=self.model.parameters())
        else:
            return self.optimizer
