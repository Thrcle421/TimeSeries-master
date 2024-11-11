from collections import OrderedDict
from typing import Any

import torch
from torch import nn
from torchvision.ops import SqueezeExcitation

from src.models.components import BaseModel

# from tsai.models.layers import ConvBlock

class Permute(nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


def noop(x=None, *args, **kwargs):
    """Do nothing."""
    return x


class GAP1d(nn.Module):
    """Global Adaptive Pooling + Flatten."""

    def __init__(self, output_size=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        # gap and flatten
        return self.gap(x).view(x.size(0), -1)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.cat(*x, dim=self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class SqueezeExciteBlock(nn.Module):
    # [Squeeze and Excitation Networks Explained with PyTorch Implementation | Committed towards better future]
    # (https://amaarora.github.io/2020/07/24/SeNet.html)
    # 建模通道之间的相互依赖关系，通过网络的全局损失函数自适应重新矫正通道之间的特征响应强度。
    # [Squeeze-and-Excitation Block - 知乎]
    # (https://zhuanlan.zhihu.com/p/358791335)

    # @Overrides SqueezeExcitation
    ...


class ConvBlock(nn.Sequential):
    """Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type`
    layers."""

    def __init__(self, c_in, c_out, kernel_size=None):
        layers = [
            nn.Conv1d(c_in, c_out, kernel_size),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
        ]
        super().__init__(*layers)


class _RNN_FCN_Base_Backbone(nn.Module):
    def __init__(
            self,
            _cell,
            c_in,  # 输入通道
            c_out,  # 输出通道
            seq_len=None,  # 序列长度
            hidden_size=100,  # 隐藏层大小
            rnn_layers=1,  # RNN层数
            bias=True,  # 是否加上偏置
            cell_dropout=0,  # Dropout几率
            rnn_dropout=0.8,  # Dropout几率
            bidirectional=False,  # 双向
            shuffle=True,
            conv_layers=None,
            kernel_size: list = None,
            squeeze_excite=0,
            squeeze_channels: list = None,
    ):
        super().__init__()
        if squeeze_excite:
            print("API Change Warning: squeeze_excite will be replaced by squeeze_channels to be consistent with tv.")
            squeeze_channels = [i // squeeze_excite for i in conv_layers]
        # RNN - first arg is usually c_in.
        # Authors modified this to seq_len by not permuting x.
        # This is what they call shuffled data.
        if kernel_size is None:
            kernel_size = [7, 5, 3]
        if conv_layers is None:
            conv_layers = [128, 256, 128]
        self.rnn = _cell(
            seq_len if shuffle else c_in,
            hidden_size,
            num_layers=rnn_layers,
            bias=bias,
            batch_first=True,
            dropout=cell_dropout,
            bidirectional=bidirectional
        )
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else noop
        # You would normally permute x. Authors did the opposite.
        self.shuffle = Permute([0, 2, 1]) if not shuffle else noop

        # FCN
        assert len(conv_layers) == len(kernel_size)
        self.conv_block1 = ConvBlock(c_in, conv_layers[0], kernel_size[0])
        if squeeze_channels is not None:
            self.se1 = SqueezeExcitation(conv_layers[0], squeeze_channels[0])
            self.se2 = SqueezeExcitation(conv_layers[1], squeeze_channels[1])
        else:
            self.se1 = noop
            self.se2 = noop
        self.conv_block2 = ConvBlock(conv_layers[0], conv_layers[1], kernel_size[1])
        self.conv_block3 = ConvBlock(conv_layers[1], conv_layers[2], kernel_size[2])
        self.gap = GAP1d(1)

        # Common
        self.concat = Concat()

    def forward(self, x):
        # RNN
        rnn_input = self.shuffle(x)  # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1]  # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)

        # FCN
        x = self.conv_block1(x)
        x = self.se1(x)
        x = self.conv_block2(x)
        x = self.se2(x)
        x = self.conv_block3(x)
        x = self.gap(x)

        # Concat
        x = self.concat([last_out, x])
        return x


class _RNN_FCN_BasePlus(nn.Sequential):
    def __init__(
            self,
            _cell,
            c_in,
            c_out,
            seq_len=None,
            hidden_size=100,
            rnn_layers=1,
            bias=True,
            cell_dropout=0,
            rnn_dropout=0.8,
            bidirectional=False,
            shuffle=True,
            fc_dropout=0.,
            conv_layers=None,
            kernel_size=None,
            squeeze_excite=0,
            custom_head=None
    ):

        if kernel_size is None:
            kernel_size = [7, 5, 3]
        if conv_layers is None:
            conv_layers = [128, 256, 128]
        if shuffle:
            assert seq_len is not None, 'need seq_len if shuffle=True'

        backbone = _RNN_FCN_Base_Backbone(_cell, c_in, c_out, seq_len=seq_len, hidden_size=hidden_size,
                                          rnn_layers=rnn_layers, bias=bias,
                                          cell_dropout=cell_dropout, rnn_dropout=rnn_dropout,
                                          bidirectional=bidirectional, shuffle=shuffle,
                                          conv_layers=conv_layers, kernel_size=kernel_size,
                                          squeeze_excite=squeeze_excite)

        self.head_nf = hidden_size * (1 + bidirectional) + conv_layers[-1]
        if custom_head:
            if isinstance(custom_head, nn.Module):
                head = custom_head
            else:
                head = custom_head(self.head_nf, c_out, seq_len)
        else:
            head_layers = [nn.Dropout(fc_dropout)] if fc_dropout else []
            head_layers += [nn.Linear(self.head_nf, c_out)]
            head = nn.Sequential(*head_layers)

        layers = OrderedDict([
            ('backbone', backbone),
            ('head', head)
        ])
        super().__init__(layers)


# class RNN_FCNPlus(_RNN_FCN_BasePlus):
#     _cell = nn.RNN
#
#
# class GRU_FCNPlus(_RNN_FCN_BasePlus):
#     _cell = nn.GRU
#
#
# class MRNN_FCNPlus(_RNN_FCN_BasePlus):
#     _cell = nn.RNN
#
#     def __init__(self, *args, squeeze_excite=16, **kwargs):
#         super().__init__(*args, squeeze_excite=squeeze_excite, **kwargs)
#
#
# class MLSTM_FCNPlus(_RNN_FCN_BasePlus):
#     _cell = nn.LSTM
#
#     def __init__(self, *args, squeeze_excite=16, **kwargs):
#         super().__init__(*args, squeeze_excite=squeeze_excite, **kwargs)
#
#
# class MGRU_FCNPlus(_RNN_FCN_BasePlus):
#     _cell = nn.GRU
#
#     def __init__(self, *args, squeeze_excite=16, **kwargs):
#         super().__init__(*args, squeeze_excite=squeeze_excite, **kwargs)


# 搭建模型

class LSTM_FCNPlus(BaseModel):
    _cell = nn.LSTM

    def __init__(
            self,
            in_channels: int,
            num_pred_classes: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            x_shape: tuple = (10, 1, 90),
            **kwargs  # for interface compatibility
    ):
        super().__init__(num_pred_classes)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        # Network Structure
        self.model = _RNN_FCN_BasePlus(
            nn.LSTM,
            in_channels,
            c_out=num_pred_classes,
            seq_len=x_shape[-1],
        )

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "LSTM_FCNPlus.yaml")
    model = hydra.utils.instantiate(cfg)
    torchinfo.summary(model, input_size=(10, 1, 90))
