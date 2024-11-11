import torch
import wandb
from pytorch_lightning import Callback, Trainer

from src.models.components import BaseModel

# TODO: add meta data in DataModule -> label names

class Confusion_Matrix(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseModel):
        wandb.log(
            {
                "val_conf_mat": wandb.plot.confusion_matrix(
                    y_true=pl_module.val_outs[0]["targets"].detach().numpy(),
                    probs=pl_module.val_outs[0]["preds"],
                )
            },
            commit=False,
            step=trainer.global_step,
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: BaseModel):
        wandb.log(
            {
                "test_conf_mat": wandb.plot.confusion_matrix(
                    y_true=pl_module.test_outs[0]["targets"].detach().numpy(),
                    probs=pl_module.test_outs[0]["preds"],
                )
            }
        )
