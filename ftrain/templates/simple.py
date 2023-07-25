"""Pre-Implemented Simple Lightning Module."""
import typing as tp  # noqa: D100
from typing import Any, Sequence

import lightning as L  # noqa: N812
import torch
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import Metric, MetricCollection

from ..callbacks.callbacks import simple_callbacks

__all__ = ["SimpleLightningModule"]


class SimpleLightningModule(L.LightningModule):
    """A simple PyTorch Lightning module that provides common functionalities for training and validation.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        loss (torch.nn.modules.loss._Loss): The loss function used during training and validation.
        lrs_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
        lrs_scheduler_moniter (str, optional): The name of the metric to monitor for adjusting the learning rate. Default is None.
        lr_init (float, optional): The initial learning rate for the optimizer. Default is 0.01.
        example_input (torch.Tensor, optional): An example input tensor to be used for model summary. Default is None.
        metric (Metric or MetricCollection, optional): The evaluation metric(s) to be used during validation. Default is None.
        *args: Additional positional arguments to be passed to the parent class.
        **kwargs: Additional keyword arguments to be passed to the parent class.

    Note:
        - This class extends PyTorch Lightning's LightningModule and requires its functionalities.
        - The 'forward' method must be implemented in the subclass that inherits from this class.
    """

    def __init__(  # noqa: D107
        self,  # noqa: ANN101
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.modules.loss._Loss,
        lrs_scheduler: tp.Optional[
            torch.optim.lr_scheduler.LRScheduler
        ] = None,
        lrs_scheduler_moniter: tp.Optional[str] = None,
        lr_init: float = 0.01,
        example_input: tp.Optional[torch.Tensor] = None,
        metric: tp.Optional[tp.Union[Metric, MetricCollection]] = None,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*args, **kwargs)
        self._torch_model = model
        self._torch_optimizer = optimizer
        self.loss = loss
        self._torch_lrs_scheduler = lrs_scheduler
        self._torch_lrs_scheuler_monitor = lrs_scheduler_moniter
        self.learning_rate = lr_init
        self.example_input_array = example_input
        self._metrics = metric
        self.train_metrics = self._metrics.clone(prefix="train_")
        self.valid_metrics = self._metrics.clone(prefix="valid_")
        self.test_metric = self._metrics.clone(prefix="test_")
        self.save_hyperparameters("lr_init", "metric_type", "lrs_scheduler")

    def forward(
        self, input_ : torch.Tensor , *args, **kwargs  # noqa: ANN003
    ) -> torch.Tensor:  # noqa: ANN101, ANN401, D102
        """Implement Forward method."""
        return self._torch_model(input_)

    def training_step(  # noqa: D102
        self,  # noqa: ANN101
        batch: torch.Tensor,
        batch_idx: torch.Tensor,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> STEP_OUTPUT:
        X, y = batch  # noqa: N806
        fp = self.forward(X)
        loss = self.loss(fp, y)
        self.log("train_loss", loss)
        self.train_metrics.update(fp, y)
        return {"train_loss": loss}

    def on_train_epoch_end(self) -> None:  # noqa: ANN101
        """Called at the end of the train epoch.

        Computes and logs validation metrics, resets the validation metrics collection.
        """
        train_stat = self.train_metrics.compute()
        self.log_dict(train_stat)
        self.train_metrics.reset()
        return

    def validation_step(  # noqa: D102
        self,  # noqa: ANN101
        batch: torch.Tensor,
        batch_idx: torch.Tensor,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> STEP_OUTPUT | None:
        X, y = batch  # noqa: N806
        fp = self.forward(X)
        loss = self.loss(fp, y)
        self.log("val_loss", loss, prog_bar=True)
        self.valid_metrics.update(fp, y)
        return {"val_loss": loss}

    def on_validation_epoch_end(self) -> None:  # noqa: ANN101
        """Called at the end of the validation epoch.

        Computes and logs validation metrics, resets the validation metrics collection.
        """
        val_stat = self.valid_metrics.compute()
        self.log_dict(val_stat, prog_bar=True)
        self.valid_metrics.reset()
        return

    def test_step(  # noqa: D102
        self,  # noqa: ANN101
        batch: torch.Tensor,
        batch_idx: torch.Tensor,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> STEP_OUTPUT | None:
        X, y = batch  # noqa: N806
        fp = self.forward(X)
        loss = self.loss(fp, y)
        self.log("test_loss", loss, prog_bar=True, on_step=True)
        self.test_metric.update(fp, y)
        return {"test_loss": loss}

    def on_test_epoch_end(self) -> None:  # noqa: ANN101
        """Called at the end of the test epoch.

        Computes and logs validation metrics, resets the validation metrics collection.
        """
        test_stat = self.test_metric.compute()
        self.log_dict(test_stat)
        self.test_metric.reset()
        return

    def configure_callbacks(  # noqa: D102
        self,
    ) -> Sequence[Callback] | Callback:  # noqa: ANN101, D102
        # Callbacks from simple callback module
        _default_callbacks = simple_callbacks
        if self._torch_lrs_scheduler is None:
            _default_callbacks.pop("lr_monitor")
        return _default_callbacks

    def configure_optimizers(self) -> dict:  # noqa: ANN101, D102
        return_dict = {"optimizer": self._torch_optimizer}
        # Scheduler Available
        if self._torch_lrs_scheduler is not None:
            return_dict["lr_scheduler"] = {
                "scheduler": self._torch_lrs_scheduler
            }
            # If conditioned on metric
            if self._torch_lrs_scheuler_monitor is not None:
                return_dict["lr_scheduler"][
                    "monitor"
                ] = self._torch_lrs_scheuler_monitor
                return_dict["lr_scheduler"]["strict"] = True
        return return_dict
