task = "sweep"
tag = "mcts_maize"
seed = [2024, 2025, 2026]


# Scorer
root_dir = "data/ckpt"
output_dir = "output"
scorer = [
    ["MaizeRFPredictor", "MaizeAdaboostPredictor", "MaizeConvNetPredictor"],
]


# Design
design = [
    {
        "mode": "MCTSDesigner",
        "num_iterations": 100000,
        "start": "AAACAACATACACATCTGTATTTCCATATGAAAGCACCCGTTTCCTTTCTTGATTATCTG",
        "simulations_per_iter": 10,
        "exploration_weight": 0.5,
        "rollout_depth": 1,
        "reuse_tree": False,
        "temperature": 0.5,
        "tree_depth_limit": 100,
    }
]


# If you want to use wandb, set the following parameters
use_wandb = False
wandb_proj_name = None
entity = None
