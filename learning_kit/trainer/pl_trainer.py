import typing as t
from datetime import timedelta

import pytorch_lightning as pl
# noinspection PyProtectedMember
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ProgressBar
)
from pytorch_lightning.loggers import Logger
# noinspection PyProtectedMember
from pytorch_lightning.plugins import _PLUGIN_INPUT
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies import Strategy
# noinspection PyProtectedMember
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,
)

from learning_kit.trainer.callbacks.tqdm_progress import TQDMProgressBar

__all__ = ["PLTrainer"]


# pl.trainer.trainer.log = "A"


def _set_callbacks(callbacks: t.Optional[t.Union[t.List[Callback], Callback]]) -> t.List[Callback]:
    if callbacks is None:
        callbacks = list()
    if not isinstance(callbacks, list):
        callbacks = [callbacks]

    is_exist_progressbar = False
    is_exist_early_stopping = False
    for c in callbacks:
        if isinstance(c, ProgressBar):
            is_exist_progressbar = True
        elif isinstance(c, EarlyStopping):
            is_exist_early_stopping = True

    if not is_exist_progressbar:
        callbacks.append(
            TQDMProgressBar(
                enable_sanity_progress_bar=False,
                enable_train_progress_bar=True,
                enable_val_progress_bar=False,
                enable_test_progress_bar=False,
                enable_predict_progress_bar=False
            )
        )

    return callbacks


class PLTrainer(pl.Trainer):

    def __init__(
            self,
            *,
            accelerator: t.Union[str, Accelerator] = "auto",
            strategy: t.Union[str, Strategy] = "auto",
            devices: t.Union[t.List[int], str, int] = "auto",
            num_nodes: int = 1,
            precision: t.Optional[_PRECISION_INPUT] = None,
            logger: t.Optional[t.Union[Logger, t.Iterable[Logger], bool]] = None,
            callbacks: t.Optional[t.Union[t.List[Callback], Callback]] = None,
            fast_dev_run: t.Union[int, bool] = False,
            max_epochs: t.Optional[int] = None,
            min_epochs: t.Optional[int] = None,
            max_steps: int = -1,
            min_steps: t.Optional[int] = None,
            max_time: t.Optional[t.Union[str, timedelta, t.Dict[str, int]]] = None,
            limit_train_batches: t.Optional[t.Union[int, float]] = None,
            limit_val_batches: t.Optional[t.Union[int, float]] = None,
            limit_test_batches: t.Optional[t.Union[int, float]] = None,
            limit_predict_batches: t.Optional[t.Union[int, float]] = None,
            overfit_batches: t.Union[int, float] = 0.0,
            val_check_interval: t.Optional[t.Union[int, float]] = None,
            check_val_every_n_epoch: t.Optional[int] = 1,
            num_sanity_val_steps: t.Optional[int] = None,
            log_every_n_steps: t.Optional[int] = None,
            enable_checkpointing: t.Optional[bool] = None,
            enable_progress_bar: t.Optional[bool] = None,
            enable_model_summary: t.Optional[bool] = None,
            accumulate_grad_batches: int = 1,
            gradient_clip_val: t.Optional[t.Union[int, float]] = None,
            gradient_clip_algorithm: t.Optional[str] = None,
            deterministic: t.Optional[t.Union[bool, _LITERAL_WARN]] = None,
            benchmark: t.Optional[bool] = None,
            inference_mode: bool = True,
            use_distributed_sampler: bool = True,
            profiler: t.Optional[t.Union[Profiler, str]] = None,
            detect_anomaly: bool = False,
            barebones: bool = False,
            plugins: t.Optional[t.Union[_PLUGIN_INPUT, t.List[_PLUGIN_INPUT]]] = None,
            sync_batchnorm: bool = False,
            reload_dataloaders_every_n_epochs: int = 0,
            default_root_dir: t.Optional[_PATH] = None,
    ) -> None:
        kwargs = locals().copy()
        for key in ("self", "__class__"):
            kwargs.pop(key)

        kwargs["callbacks"] = _set_callbacks(kwargs["callbacks"])

        super(PLTrainer, self).__init__(**kwargs)
