# Introduction
This repository contains the official implementation of the paper: "Genome-wide mining and designing of short transcriptional enhancers in plants".

# Installation

First, clone the repository:
```bash
git clone https://github.com/shiyegao/BaseEvolve.git
cd BaseEvolve
```


Then, install the [uv](https://docs.astral.sh/uv/) package manager.

Once uv is installed, install the project dependencies:

```bash
uv sync
```


# Usage
## Regressor Training
To train enhancer activity regressors, run the following commands:

```bash
uv run scripts/cnn_reg.py
uv run scripts/adaboost_reg.py
uv run scripts/rf_reg.py
```

You can modify the model hyperparameters in the corresponding configuration files located at: `conf/regression`.

## Enhancer Design
To accelerate the design process, you can reduce the `num_iterations` parameter in the config file: `conf/design`.

### [Optional] Wandb

If you’d like to log experiments with [wandb](https://wandb.ai/site), set the `use_wandb` to `True` in the config file.
Also, specify your `wandb_proj_name` and `entity`.

### BaseEvolve Algorithm
Run the BaseEvolve algorithm with:
```bash
uv run scripts/design.py conf/design/baseevolve.py
```

### Baseline Methods
You can also run several baseline enhancer design strategies:
```bash
uv run scripts/design.py conf/design/random.py
uv run scripts/design.py conf/design/random_single.py
uv run scripts/design.py conf/design/beam.py
uv run scripts/design.py conf/design/anneal.py
uv run scripts/design.py conf/design/mcts.py
uv run scripts/design.py conf/design/ga.py
```
