import typing as t

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import (
    TensorBoardLogger
)
from sklearn.datasets import make_regression
from torch.optim import Adam
from torch.utils.data import DataLoader

from learning_kit.cv_splitter.random import train_test_random_split
from learning_kit.data.tensor_like import TensorLikeDataset
from learning_kit.data.xy_dataset import XYDataset
from learning_kit.nn.fc_block import MultiFullyConnectedBlock
from learning_kit.nn.losses.combine import CombineLoss
from learning_kit.optim.deferred_param import DeferredParamOptimizer
from learning_kit.trainer.modules.single import SingleModelTrainModule
from learning_kit.trainer.pl_trainer import PLTrainer


class FCNet(nn.Module):

    def __init__(self, n_inputs: int, n_outputs: int):
        super(FCNet, self).__init__()

        self.fc_net1 = MultiFullyConnectedBlock(
            in_features=n_inputs,
            hidden_layer_sizes=(256, 128, 64),
            batch_norm=False,
            dropout_probability=0
        )
        self.out_layer1 = nn.Linear(64, n_outputs)

        self.fc_net2 = MultiFullyConnectedBlock(
            in_features=n_inputs,
            hidden_layer_sizes=(256, 128, 64),
            batch_norm=False,
            dropout_probability=0
        )
        self.out_layer2 = nn.Linear(64, n_outputs)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self.out_layer1(self.fc_net1(x1)), self.out_layer2(self.fc_net2(x2))


def main():
    torch.set_default_dtype(torch.float32)

    x1, y1 = make_regression(n_samples=64 * 10, n_features=5, n_informative=5, n_targets=2)
    x2, y2 = make_regression(n_samples=64 * 10, n_features=5, n_informative=3, n_targets=2)

    x_dataset = TensorLikeDataset(x1, x2)
    y_dataset = TensorLikeDataset(y1, y2)

    cv_splits = train_test_random_split(x_dataset, y_dataset, n_splits=1, test_size=0.1)
    cv_splits = [cv_split for cv_split in cv_splits]
    x_train, x_test, y_train, y_test = cv_splits[0]

    xy_dataset_train = XYDataset(x_train, y_train)
    xy_dataset_test = XYDataset(x_test, y_test)

    train_loader = DataLoader(xy_dataset_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(xy_dataset_test, batch_size=64)

    model = SingleModelTrainModule(
        model=FCNet(n_inputs=5, n_outputs=2).to(dtype=torch.float64),
        loss_func=CombineLoss([nn.MSELoss(), nn.MSELoss()]),
        optimizer=DeferredParamOptimizer(Adam, lr=0.001)
    )
    model.example_input_array = xy_dataset_train.example_input_array
    model_ckpt = ModelCheckpoint(
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_last=True,
        save_top_k=3,
        save_weights_only=True
    )
    model_ckpt.CHECKPOINT_NAME_LAST = "last-{epoch}-{val_loss:.4f}"

    trainer = PLTrainer(
        accelerator="gpu",
        logger=[
            TensorBoardLogger(save_dir="./", log_graph=True),
            # CSVLogger(save_dir="csv_logs/")
        ],
        callbacks=[
            EarlyStopping("val_loss", patience=10),
            model_ckpt
        ],
        max_epochs=1000,
        num_nodes=1 if torch.cuda.is_available() else None,
        # enable_progress_bar=True,
        # enable_model_summary=False
    )

    # 훈련 실행
    trainer.fit(model, train_loader, val_loader)
    trainer.predict(model, val_loader)


if __name__ == "__main__":
    main()
