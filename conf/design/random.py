task = "sweep"
tag = "random_maize"
seed = [2024, 2025, 2026]


# Scorer
root_dir = "data/ckpt"
output_dir = "output"
scorer = [
    ["MaizeRFPredictor", "MaizeAdaboostPredictor", "MaizeCNNPredictor"],
]


# Design
design = {
    "mode": "RandomDesigner",
    "num_iterations": 100000,
    "start": "AAACAACATACACATCTGTATTTCCATATGAAAGCACCCGTTTCCTTTCTTGATTATCTG",
}


# If you want to use wandb, set the following parameters
use_wandb = False
wandb_proj_name = None
entity = None
