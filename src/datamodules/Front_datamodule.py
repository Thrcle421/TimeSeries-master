from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from imblearn.over_sampling import BorderlineSMOTE
from pytorch_lightning import LightningDataModule

from src.datamodules.components import InputData
from src.datamodules.preprocess import scale_data

FrontExp_Datasets = ['PPMI', 'ADNI', 'FTD_fMRI', 'OCD_fMRI', 'ADNI_fMRI']


def balance_data(train_x: np.ndarray, train_y: np.ndarray):
    # 平衡样本
    smote = BorderlineSMOTE(random_state=42)

    shape = train_x.shape
    train_x = train_x.reshape(shape[0], -1)
    train_x, train_y = smote.fit_resample(train_x, train_y)
    train_x = train_x.reshape(-1, shape[1], shape[2])
    return train_x, train_y


def load_front_data(data_path: Path, scale: bool = False, balance: bool = True):
    # data_path / TRAIN.npz, test.npz
    train = np.load(data_path / 'TRAIN.npz')
    test = np.load(data_path / 'TEST.npz')

    # OneHot Encoding (Disabled)
    # encoder = OneHotEncoder(categories='auto', sparse=False)
    # train_y = encoder.fit_transform(np.expand_dims(train['train_y'], axis=-1))
    # test_y = encoder.transform(np.expand_dims(test['test_y'], axis=-1))
    train_y = train['train_y']
    test_y = test['test_y']

    # Scale data
    if scale:
        print("Scaling data...")
        train_x, test_x = scale_data(train['train_x'], test['test_x'], 'quantile', 'both')
    else:
        train_x, test_x = train['train_x'], test['test_x']

    # 归一化
    train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
    test_x = (test_x - test_x.min()) / (test_x.max() - test_x.min())

    # Normalize
    train_x = (train_x - train_x.mean()) / train_x.std()
    test_x = (test_x - test_x.mean()) / test_x.std()

    # 平衡样本
    if balance:
        print("Balancing data...")
        train_x, train_y = balance_data(train_x, train_y)

    train_input = InputData(x=torch.from_numpy(train_x).float(),
                            y=torch.from_numpy(train_y).long())
    test_input = InputData(x=torch.from_numpy(test_x).float(),
                           y=torch.from_numpy(test_y).long())
    return train_input, test_input


class FrontDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/Front",  # UCR数据集顶层目录
            dataset: str = "ADNI",  # UCR数据集名称
            fold: str = "",  # 折id
            dataset_name: str = "ADNI",  # 组合之后的目标数据集
            val_split: float = 0.2,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            scale: bool = False,  # 缩放数据控制
            balance=True,  # 平衡样本
    ):
        super().__init__()

        assert dataset in FrontExp_Datasets, f"dataset must be in {FrontExp_Datasets}"
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
        [Front]
            Dataset: {self.hparams.dataset_name}
            Train shape: {self.data_train.x.shape}
            Val shape: {self.data_val.x.shape}
            Test shape: {self.data_test.x.shape}
            Num classes: {self.num_classes}
          """

    def setup(self, stage: Optional[str] = None):
        """加载数据，并设置data_train, data_val, data_test
        Lightning在fit和test阶段都会调用这个函数，所以要特别小心，不要两次分割数据集."""
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val = load_front_data(Path(self.hparams.data_dir) / self.hparams.dataset_name,
                                                             self.hparams.scale, self.hparams.balance
                                                             )
            self.data_test = self.data_val

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
        # 保留test阶段的原因是：
        # 1.在进行数据分析的时候可以直接索引test的metric，不需要在val中取最好的一轮
        # 2.方便在表格中进行对比
        # 3.对UCR数据的兼容
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "Front.yaml")
    cfg.data_dir = str(root / "data" / "Front")
    dataset: FrontDataModule = hydra.utils.instantiate(cfg)
    dataset.setup()
    print(dataset.info())
