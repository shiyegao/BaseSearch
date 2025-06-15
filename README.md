# Introduction
This repository contains the official implementation of the paper: "Genome-wide mining and designing of short transcriptional enhancers in plants".

# Installation

First, clone the repository:
```bash
git clone https://github.com/shiyegao/BaseSearch.git
cd BaseSearch
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
uv run cnn_reg.py
uv run adaboost_reg.py
uv run rf_reg.py
```

You can modify the model hyperparameters in the corresponding configuration files located at: `conf/regression`.

## Enhancer Design
To accelerate the design process, you can reduce the `num_iterations` parameter in the config file: `conf/design`.

### [Optional] Wandb

If you’d like to log experiments with [wandb](https://wandb.ai/site), set the `use_wandb` to `True` in the config file.
Also, specify your `wandb_proj_name` and `entity`.

### BaseSearch Algorithm
Run the BaseSearch algorithm with:
```bash
uv run design.py conf/design/basesearch.py
```

### Baseline Methods
You can also run several baseline enhancer design strategies:
```bash
uv run design.py conf/design/random.py
uv run design.py conf/design/random_single.py
uv run design.py conf/design/beam.py
uv run design.py conf/design/anneal.py
uv run design.py conf/design/mcts.py
uv run design.py conf/design/ga.py
```
