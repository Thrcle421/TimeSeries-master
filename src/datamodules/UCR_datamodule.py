from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import OneHotEncoder

from src.datamodules.components import InputData

UCR_DATASETS = ['Haptics', 'Worms', 'Computers', 'UWaveGestureLibraryAll',
                'Strawberry', 'Car', 'BeetleFly', 'wafer', 'CBF', 'Adiac',
                'Lighting2', 'ItalyPowerDemand', 'yoga', 'Trace', 'ShapesAll',
                'Beef', 'MALLAT', 'MiddlePhalanxTW', 'Meat', 'Herring',
                'MiddlePhalanxOutlineCorrect', 'FordA', 'SwedishLeaf',
                'SonyAIBORobotSurface', 'InlineSkate', 'WormsTwoClass', 'OSULeaf',
                'Ham', 'uWaveGestureLibrary_Z', 'NonInvasiveFatalECG_Thorax1',
                'ToeSegmentation1', 'ScreenType', 'SmallKitchenAppliances',
                'WordsSynonyms', 'MoteStrain', 'synthetic_control', 'Cricket_X',
                'ECGFiveDays', 'Wine', 'Cricket_Y', 'TwoLeadECG', 'Two_Patterns',
                'Phoneme', 'MiddlePhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
                'DistalPhalanxTW', 'FacesUCR', 'ECG5000', '50words', 'HandOutlines',
                'Coffee', 'Gun_Point', 'FordB', 'InsectWingbeatSound', 'MedicalImages',
                'Symbols', 'ArrowHead', 'ProximalPhalanxOutlineAgeGroup',
                'SonyAIBORobotSurfaceII', 'ChlorineConcentration', 'Plane', 'Lighting7',
                'PhalangesOutlinesCorrect', 'ShapeletSim', 'DistalPhalanxOutlineAgeGroup',
                'uWaveGestureLibrary_X', 'FaceFour', 'RefrigerationDevices', 'ECG200',
                'ToeSegmentation2', 'CinC_ECG_torso', 'BirdChicken', 'OliveOil',
                'LargeKitchenAppliances', 'uWaveGestureLibrary_Y',
                'NonInvasiveFatalECG_Thorax2', 'FISH', 'ProximalPhalanxOutlineCorrect',
                'Cricket_Z', 'FaceAll', 'StarLightCurves', 'ElectricDevices', 'Earthquakes',
                'DiatomSizeReduction', 'ProximalPhalanxTW']


def load_ucr_data(data_path: Path,
                  encoder: Optional[OneHotEncoder] = None
                  ) -> Tuple[InputData, InputData, OneHotEncoder]:
    experiment = data_path.parts[-1]

    train = np.loadtxt(data_path / f'{experiment}_TRAIN', delimiter=',')
    test = np.loadtxt(data_path / f'{experiment}_TEST', delimiter=',')

    # 把标签转换为0开始
    label_lower_bound = min(np.min(train[:, 0]), np.min(test[:, 0]))
    # if encoder is None:
    #     encoder = OneHotEncoder(categories='auto', sparse=False)
    #     y_train = encoder.fit_transform(np.expand_dims(train[:, 0], axis=-1))
    # else:
    #     y_train = encoder.transform(np.expand_dims(train[:, 0], axis=-1))
    # y_test = encoder.transform(np.expand_dims(test[:, 0], axis=-1))

    y_train = train[:, 0] - label_lower_bound
    y_test = test[:, 0] - label_lower_bound
    #
    # if y_train.shape[1] == 2:
    #     # there are only 2 classes, so there only needs to be one
    #     # output
    #     y_train = y_train[:, 0]
    #     y_test = y_test[:, 0]

    # UCR data is univariate, so an additional dimension is added
    # at index 1 to make it of shape (N, Channels, Length)
    # as the model expects
    train_input = InputData(x=torch.from_numpy(train[:, 1:]).unsqueeze(1).float(),
                            y=torch.from_numpy(y_train).long())
    test_input = InputData(x=torch.from_numpy(test[:, 1:]).unsqueeze(1).float(),
                           y=torch.from_numpy(y_test).long())
    return train_input, test_input, encoder


class UCRDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/ucr-ts-archive-2015",  # UCR数据集顶层目录
            dataset: str = "ECG200",  # UCR数据集名称
            val_split: float = 0.2,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        assert dataset in UCR_DATASETS, f"dataset must be in {UCR_DATASETS}"
        assert Path(data_dir).exists(), f"data_dir {data_dir} does not exist"
        # 保存超参数(在类和ckpt中)
        self.save_hyperparameters(logger=False)

        # 这里使用自定义类型 "InputData"
        self.data_train: Optional[InputData] = None
        self.data_val: Optional[InputData] = None
        self.data_test: Optional[InputData] = None

    @property
    def num_classes(self):
        return len(self.data_train.y.unique(dim=0))

    @property
    def in_channels(self):
        return self.data_train.x.shape[1]

    @property
    def x_shape(self):
        # TODO: remove in_channels, use x_shape instead
        return self.data_train.x.shape

    def info(self):
        return f"""
        [UCR]
            Dataset: {self.hparams.dataset}
            Train shape: {self.data_train.x.shape}
            Val shape: {self.data_val.x.shape}
            Test shape: {self.data_test.x.shape}
            Num classes: {self.num_classes}
          """

    def setup(self, stage: Optional[str] = None):
        """加载数据，并设置data_train, data_val, data_test
        Lightning在fit和test阶段都会调用这个函数，所以要特别小心，不要两次分割数据集."""
        if not self.data_train and not self.data_val and not self.data_test:
            data_train, self.data_test, _ = load_ucr_data(Path(self.hparams.data_dir) / self.hparams.dataset)
            self.data_train, self.data_val = data_train.split(self.hparams.val_split)

    def train_dataloader(self):
        return self.data_train.dataloader(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return self.data_val.dataloader(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.data_test.dataloader(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "UCR.yaml")
    cfg.data_dir = str(root / "data" / "UCR_TS_Archive_2015")
    dataset: UCRDataModule = hydra.utils.instantiate(cfg)
    dataset.setup()
    print(dataset.info)
