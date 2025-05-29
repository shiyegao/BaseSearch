task = "sweep"
tag = "ga_maize"
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
        "mode": "GADesigner",
        "num_iterations": 100000,
        "mutation_rate": 0.1,
        "population_size": 50,
        "crossover_rate": 0.5,
        "elitism_rate": 0.01,
        "start": "AAACAACATACACATCTGTATTTCCATATGAAAGCACCCGTTTCCTTTCTTGATTATCTG",
    }
]


# If you want to use wandb, set the following parameters
use_wandb = False
wandb_proj_name = None
entity = None
