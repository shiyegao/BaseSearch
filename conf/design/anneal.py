task = "sweep"
tag = "annealing_maize"
seed = [2024, 2025, 2026]


# Scorer
root_dir = "data/ckpt"
output_dir = "output"
scorer = [
    ["MaizeRFPredictor", "MaizeAdaboostPredictor", "MaizeCNNPredictor"],
]


# Design
design = [
    {
        "mode": "AnnealDesigner",
        "num_iterations": 100000,
        "initial_temp": 5,
        "min_temp": 0.001,
        "cooling_rate": 0.9,
        "mutation_rate": 0.1,
        "start": "AAACAACATACACATCTGTATTTCCATATGAAAGCACCCGTTTCCTTTCTTGATTATCTG",
    }
]


# If you want to use wandb, set the following parameters
use_wandb = False
wandb_proj_name = None
entity = None
