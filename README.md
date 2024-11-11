<div align="center">

# Time Series Classification

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

</div>

## Description

In the paper, we use data scaling to preprocess the data and analyze the Squeeze-and-Excitation block we use in the network LSTMFCN about which channel it measures can have better effects. During the training, we design the clus tering loss and the contrastive loss with the characteristics of plug and play to aid training and reduce overfitting. Dur ing the process, we can get the pseudo labels in the test set, we use the labels to the method called PWDBA(Pseudo Weighted DTW Barycentric Averaging), which can improve the stability of training.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/InEase/TimeSeries
cd TimeSeries

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```
