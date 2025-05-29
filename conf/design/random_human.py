task = "sweep"
tag = "random_test"
seed = [2024, 2025, 2026]


# Scorer
root_dir = "data/ckpt"
output_dir = "output"
scorer = [
    ["HumanRFPredictor", "HumanAdaboostPredictor", "HumanCNNPredictor"],
]


# Design
design = {
    "mode": "RandomDesigner",
    "num_iterations": 10000,
    "start": "AAACAACATACACATCTGTATTTCCATATGAAAGCACCCGTTTCCTTTCTTGATTATCTG",
}


# If you want to use wandb, set the following parameters
use_wandb = False
wandb_proj_name = None
entity = None
