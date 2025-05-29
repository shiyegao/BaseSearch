# Introduction
This repo is the official implementation of the paper, "Genome-wide mining and designing of short transcriptional enhancers in plants".

# Installation

You should first install the [uv](https://docs.astral.sh/uv/) package manager.

Then you can install the dependencies by simply running the following command.

```bash
uv sync
```


# Usage
## Regressor Training
You can run the following commands to train the enhancer activity regressors. 

```bash
uv run scripts/cnn_reg.py
uv run scripts/adaboost_reg.py
uv run scripts/rf_reg.py
```

Moreover, you can change the hyperparameters in the config files under the path: `conf/regression`.

## Enhancer Design
To speed up the design process, you can set a smaller `num_iterations` in the config file: `conf/design`.

### [Optional] Wandb

If you want to use wandb for logging, you can set the `use_wandb` to `True` in the config file and set the `wandb_proj_name` and `entity` to your own wandb project and entity.

### BaseEvolve
You can run the following command for our BaseEvolve algorithm.
```bash
uv run scripts/design.py conf/design/baseevolve.py
```

### Baselines
You can run the following commands to run the baselines.

```bash
uv run scripts/design.py conf/design/random.py
uv run scripts/design.py conf/design/random_single.py
uv run scripts/design.py conf/design/beam.py
uv run scripts/design.py conf/design/anneal.py
uv run scripts/design.py conf/design/mcts.py
uv run scripts/design.py conf/design/ga.py
```
