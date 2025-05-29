import wandb
from itertools import product
import sys

sys.path.append(".")

from utils.tool import load_config, set_seed
from utils.scorer import load_scorer
import utils.designer as designer_utils


def sweep(cfg):
    # All hyperparameters with lists will be ablated
    param_dict = {}
    for k in dir(cfg):
        v = getattr(cfg, k)
        if isinstance(v, list):
            param_dict[k] = v

    keys = param_dict.keys()
    values = param_dict.values()
    param_combinations = [
        dict(zip(keys, combination)) for combination in product(*values)
    ]

    for i, ablation in enumerate(param_combinations):
        print(
            f"\nNow running {i + 1} / {len(param_combinations)} combinations: {ablation}"
        )

        for k, v in ablation.items():
            setattr(cfg, k, v)

        # Set seed
        set_seed(cfg.seed)

        # Run
        hyperparams = {
            attr: getattr(cfg, attr) for attr in dir(cfg) if not attr.startswith("__")
        }
        run_name = "_".join([k for k in keys if len(param_dict[k]) > 1])

        if cfg.use_wandb:
            wandb_run = wandb.init(
                project=cfg.wandb_proj_name,
                entity=cfg.entity,
                config=hyperparams,
                name=run_name,
            )

        # Design
        scorer = load_scorer(cfg.scorer, cfg.root_dir)
        designer = getattr(designer_utils, cfg.design["mode"])(cfg)
        designer.design(scorer)

        # Finish
        if cfg.use_wandb:
            wandb_run.finish()


if __name__ == "__main__":
    cfg = load_config()
    func = globals().get(cfg.task)
    if func and callable(func):
        func(cfg)
    else:
        print(f"Function '{cfg.task}' not found or not callable.")
