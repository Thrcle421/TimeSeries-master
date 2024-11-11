from typing import Any, List, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class BaseModel(LightningModule):
    def __init__(
            self,
            num_pred_classes: int,
    ):
        super().__init__()
        self.num_pred_classes = num_pred_classes
        self.class_names = [str(i) for i in range(num_pred_classes)]

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(num_classes=num_pred_classes)
        self.val_acc = Accuracy(num_classes=num_pred_classes)
        self.test_acc = Accuracy(num_classes=num_pred_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # for recording model outputs
        self.train_outs: Optional[List[Any]] = None
        self.val_outs: Optional[List[Any]] = None
        self.test_outs: Optional[List[Any]] = None

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def step(self, batch: Any) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError()

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def training_step_record(self, loss: Tensor, preds: Tensor, targets: Tensor):
        # loss: Tensor
        # preds: N x Class
        # targets: N
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # Confusion Matrix 等指标的计算使用Callback，虽然会带来额外的内存消耗，但是他们与训练逻辑无关
        self.train_outs = outputs

    def validation_step_record(self, loss: Tensor, preds: Tensor, targets: Tensor):
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

        self.val_outs = outputs

    def test_step_record(self, loss: Tensor, preds: Tensor, targets: Tensor):
        self.test_loss(loss)
        if len(targets.shape) > 1:
            self.test_acc(preds, targets.argmax(dim=-1))
        else:
            self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_outs = outputs

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
