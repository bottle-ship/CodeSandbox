import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import (
    TensorBoardLogger,
    CSVLogger
)
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from callbacks.tqdm_progress import TQDMProgressBar
from optim.adam import Adam


class DefaultLightningModule(pl.LightningModule):

    def __init__(
            self,
            model: nn.Module,
            loss_func: nn.Module
    ):
        super(DefaultLightningModule, self).__init__()
        self.save_hyperparameters()

        self.model = model
        self.loss_func = loss_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch

        logits = self(x)
        loss = self.loss_func(logits, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_func(logits, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch) -> torch.Tensor:
        x, y = batch
        x = x.view(x.size(0), -1)
        x = self.model(x)

        return x

    def configure_optimizers(self):
        return Adam(lr=0.001).configure_optimizer(params=self.model.parameters())


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_train = MNIST(download=True, root="./data", train=True, transform=transform)
    mnist_val = MNIST(download=True, root="./data", train=False, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=64)

    # 모델 및 훈련 설정
    model = DefaultLightningModule(
        model=nn.Linear(28 * 28, 10),
        loss_func=nn.CrossEntropyLoss()
    )
    model_ckpt = ModelCheckpoint(
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_last=True,
        save_top_k=3,
        save_weights_only=True
    )
    model_ckpt.CHECKPOINT_NAME_LAST = "last-{epoch}-{val_loss:.4f}"

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=[
            TensorBoardLogger(save_dir="./", log_graph=True),
            # CSVLogger(save_dir="csv_logs/")
        ],
        callbacks=[
            TQDMProgressBar(enable_predict_progress_bar=False),
            EarlyStopping("val_loss", patience=10),
            model_ckpt
        ],
        max_epochs=1000,
        num_nodes=1 if torch.cuda.is_available() else None,
        enable_progress_bar=True,
        enable_model_summary=False
    )

    # 훈련 실행
    trainer.fit(model, train_loader, val_loader)
    trainer.predict(model, val_loader)


if __name__ == "__main__":
    main()
