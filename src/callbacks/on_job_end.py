from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import DictConfig


class JobEndCallback(Callback):
    def on_job_end(self, config: DictConfig, **kwargs: Any) -> None:
        print(f"Job ended,uploading...")
