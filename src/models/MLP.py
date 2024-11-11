from typing import Any, List

import torch
from torch import nn

from src.models.components import BaseModel


class MLP(BaseModel):
    def __init__(
            self,
            num_pred_classes: int,
            x_shape: List[int],
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            **kwargs  # for interface compatibility
    ):
        super().__init__(num_pred_classes)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        flattened_shape = x_shape[1] * x_shape[2]

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),

            nn.Linear(flattened_shape, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, num_pred_classes),
            # nn.Softmax(dim=-1), # Embedded in celoss
        )

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "MLP.yaml")
    model = hydra.utils.instantiate(cfg)
    torchinfo.summary(model, input_size=(10, 1, 90))
