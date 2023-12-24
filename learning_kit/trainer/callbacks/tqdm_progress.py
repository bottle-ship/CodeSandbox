import typing as t

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import (
    TQDMProgressBar as _TQDMProgressBar,
    convert_inf
)
from pytorch_lightning.utilities.types import STEP_OUTPUT


class TQDMProgressBar(_TQDMProgressBar):

    def __init__(
            self,
            refresh_rate: int = 1,
            process_position: int = 0,
            sanity_check_description: str = "Sanity Checking",
            train_description: str = "Training",
            validation_description: str = "Validation",
            test_description: str = "Testing",
            predict_description: str = "Predicting",
            enable_sanity_progress_bar: bool = False,
            enable_train_progress_bar: bool = True,
            enable_val_progress_bar: bool = False,
            enable_test_progress_bar: bool = False,
            enable_predict_progress_bar: bool = True
    ):
        super(TQDMProgressBar, self).__init__(refresh_rate=refresh_rate, process_position=process_position)

        self._sanity_check_description = sanity_check_description
        self._train_description = train_description
        self._validation_description = validation_description
        self._test_description = test_description
        self._predict_description = predict_description
        self._enable_sanity_progress_bar = enable_sanity_progress_bar
        self._enable_train_progress_bar = enable_train_progress_bar
        self._enable_val_progress_bar = enable_val_progress_bar
        self._enable_test_progress_bar = enable_test_progress_bar
        self._enable_predict_progress_bar = enable_predict_progress_bar

    @property
    def sanity_check_description(self) -> str:
        return self._sanity_check_description

    @property
    def train_description(self) -> str:
        return self._train_description

    @property
    def validation_description(self) -> str:
        return self._validation_description

    @property
    def test_description(self) -> str:
        return self._test_description

    @property
    def predict_description(self) -> str:
        return self._predict_description

    def disable(self) -> None:
        raise NotImplementedError

    def enable(self) -> None:
        raise NotImplementedError

    @property
    def is_enabled_sanity_progress_bar(self) -> bool:
        return self._enable_sanity_progress_bar and self.refresh_rate > 0

    @property
    def is_disabled_sanity_progress_bar(self) -> bool:
        return not self.is_enabled_sanity_progress_bar

    @property
    def is_enabled_train_progress_bar(self) -> bool:
        return self._enable_train_progress_bar and self.refresh_rate > 0

    @property
    def is_disabled_train_progress_bar(self) -> bool:
        return not self.is_enabled_train_progress_bar

    @property
    def is_enabled_val_progress_bar(self) -> bool:
        return self._enable_val_progress_bar and self.refresh_rate > 0

    @property
    def is_disabled_val_progress_bar(self) -> bool:
        return not self.is_enabled_val_progress_bar

    @property
    def is_enabled_test_progress_bar(self) -> bool:
        return self._enable_test_progress_bar and self.refresh_rate > 0

    @property
    def is_disabled_test_progress_bar(self) -> bool:
        return not self.is_enabled_test_progress_bar

    @property
    def is_enabled_predict_progress_bar(self) -> bool:
        return self._enable_predict_progress_bar and self.refresh_rate > 0

    @property
    def is_disabled_predict_progress_bar(self) -> bool:
        return not self.is_enabled_predict_progress_bar

    def on_sanity_check_start(self, *_: t.Any) -> None:
        if self.is_enabled_sanity_progress_bar:
            super(TQDMProgressBar, self).on_sanity_check_start(*_)

    def on_sanity_check_end(self, *_: t.Any) -> None:
        if self.is_enabled_sanity_progress_bar:
            super(TQDMProgressBar, self).on_sanity_check_end(*_)

    def on_train_start(self, *_: t.Any) -> None:
        if self.is_enabled_train_progress_bar:
            self.train_progress_bar = self.init_train_tqdm()
            self.train_progress_bar.reset(self.trainer.max_epochs)

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: t.Any) -> None:
        pass

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: t.Any,
            batch_idx: int
    ) -> None:
        if self.is_enabled_train_progress_bar:
            self.train_progress_bar.set_postfix(self.get_train_postfix(trainer, pl_module, batch_idx + 1))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_enabled_train_progress_bar:
            n = trainer.current_epoch + 1
            if self._should_update(n, self.train_progress_bar.total):
                bar = self.train_progress_bar
                if not bar.disable:
                    bar.n = n
                    bar.refresh()
                self.train_progress_bar.set_postfix(
                    self.get_train_postfix(trainer, pl_module, convert_inf(self.total_train_batches))
                )

    def on_train_end(self, *_: t.Any) -> None:
        if self.is_enabled_train_progress_bar:
            super(TQDMProgressBar, self).on_train_end(*_)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_enabled_val_progress_bar:
            super(TQDMProgressBar, self).on_validation_start(trainer, pl_module)

    def on_validation_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: t.Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.is_enabled_val_progress_bar:
            super(TQDMProgressBar, self).on_validation_batch_start(
                trainer, pl_module, batch, batch_idx, dataloader_idx
            )

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: t.Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.is_enabled_val_progress_bar:
            super(TQDMProgressBar, self).on_validation_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_enabled_val_progress_bar:
            super(TQDMProgressBar, self).on_validation_end(trainer, pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_enabled_test_progress_bar:
            super(TQDMProgressBar, self).on_test_start(trainer, pl_module)

    def on_test_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: t.Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.is_enabled_test_progress_bar:
            super(TQDMProgressBar, self).on_test_batch_start(
                trainer, pl_module, batch, batch_idx, dataloader_idx
            )

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: t.Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.is_enabled_test_progress_bar:
            super(TQDMProgressBar, self).on_test_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_enabled_test_progress_bar:
            super(TQDMProgressBar, self).on_test_end(trainer, pl_module)

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_enabled_predict_progress_bar:
            super(TQDMProgressBar, self).on_predict_start(trainer, pl_module)

    def on_predict_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: t.Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.is_enabled_predict_progress_bar:
            super(TQDMProgressBar, self).on_predict_batch_start(
                trainer, pl_module, batch, batch_idx, dataloader_idx
            )

    def on_predict_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: t.Any,
            batch: t.Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.is_enabled_predict_progress_bar:
            super(TQDMProgressBar, self).on_predict_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_enabled_predict_progress_bar:
            super(TQDMProgressBar, self).on_predict_end(trainer, pl_module)

    def get_train_postfix(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", step: t.Optional[int]
    ) -> t.Dict[str, t.Union[int, str, float, t.Dict[str, float]]]:
        metrics = self.get_metrics(trainer, pl_module)
        metrics.pop("v_num")
        postfix = {"step": f"{step}/{convert_inf(self.total_train_batches)}"}
        postfix.update(metrics)

        return postfix
