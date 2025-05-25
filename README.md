# Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design

An unofficial implementation of the ICML 2021 paper:
[Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design](https://proceedings.mlr.press/v139/foster21a) by Foster et al.[^1]

## Structure

- [`config/`](config/): hydra configuration files
- [`data/`](data/): simulate experiments
- [`loss/`](loss/): loss functions
- [`model/`](model/): design policy networks
- [`utils/`](utils/): helper functions


## Setup

### Requirements:

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Additional dependencies are listed in [requirements.txt](requirements.txt)

### Installation:

Create a new virtual environment:
```shell
# create a new conda environment
conda create -n dad python=3.12 -y
# activate the environment
conda activate dad
```

Install PyTorch (adjust based on your OS and CUDA version, see: https://pytorch.org/get-started/locally/):
```shell
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

Install remaining dependencies:
```shell
pip install -r requirements.txt
```

## Experiments

To reproduce the location-finding experiment with default settings:

```shell
python location_finding.py
```

To run with custom hyperparameters:

```shell
python location_finding.py data.K=1 data.theta_dist=uniform
```


### Logging with Weights & Biases

To enable logging and monitor training in real time, add the following flag:

```
wandb.use_wandb=True
```

Make sure you are logged into your [Weights & Biases](https://wandb.ai/) account.


## Credits

This repository aims to reproduce the main experimental results from the paper with simplified dependencies and a modular structure. If you find it useful, please feel free to build on it. Contributions are warmly welcome and appreciated :)

For the official implementation, please refer to [dad](https://github.com/ae-foster/dad).


[^1]: Foster, Adam, Desi R. Ivanova, Ilyas Malik, and Tom Rainforth. ‘Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design’. In Proceedings of the 38th International Conference on Machine Learning, 3384–95. PMLR, 2021. https://proceedings.mlr.press/v139/foster21a.
