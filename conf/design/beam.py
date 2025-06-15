task = "sweep"
tag = "beam_maize"
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
        "mode": "BeamDesigner",
        "num_iterations": 100000,
        "beam_width": 20,  # Default beam width is 10
        "num_children": 5,  # Number of children per beam sequence
        "mutation_rate": 0.1,  # Mutation rate for generating children
        "start": "AAACAACATACACATCTGTATTTCCATATGAAAGCACCCGTTTCCTTTCTTGATTATCTG",
    }
]

# If you want to use wandb, set the following parameters
use_wandb = False
wandb_proj_name = None
entity = None
