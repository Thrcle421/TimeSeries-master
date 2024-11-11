from typing import Any

import torch
from torch import nn

from src.models.components import BaseModel, ConvBlock


class FCN(BaseModel):
    def __init__(
            self,
            in_channels: int,
            num_pred_classes: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            **kwargs  # for interface compatibility
    ):
        super().__init__(num_pred_classes)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        # Network Structure
        self.layers = nn.Sequential(
            ConvBlock(in_channels, 128, 8, 1),
            ConvBlock(128, 256, 5, 1),
            ConvBlock(256, 128, 3, 1),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.final = nn.Linear(128, num_pred_classes)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.gap(x).squeeze(-1)
        return self.final(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, y = self.step(batch)
        return super().training_step_record(loss, preds, y)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, y = self.step(batch)
        return super().validation_step_record(loss, preds, y)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, y = self.step(batch)
        return super().test_step_record(loss, preds, y)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    import torchinfo

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "FCN.yaml")
    model = hydra.utils.instantiate(cfg)
    torchinfo.summary(model, input_size=(10, 1, 90))
