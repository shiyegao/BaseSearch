import os
import json
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from utils.tool import setup_logger


DNA_BASES = ["A", "T", "C", "G"]


class Designer:
    def __init__(self, cfg):
        self.tag = cfg.tag
        self.cfg = cfg
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = f"{cfg.output_dir}/{self.tag}/{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = setup_logger(self.output_dir)

        # Create output directory and save config
        self._save_config(cfg, self.output_dir)

        # Design parameters
        self.start = cfg.design["start"]
        self.num_iterations = cfg.design["num_iterations"]

        # Design results
        self.results = dict()
        self.max_score = 0
        self.best_sequence = self.start

    def _save_config(self, cfg, folder):
        """Save the configuration to a file."""
        # Convert to dict if it's not already
        if not isinstance(cfg, dict):
            cfg_dict = {
                attr: getattr(cfg, attr)
                for attr in dir(cfg)
                if not attr.startswith("__")
            }
        else:
            cfg_dict = cfg

        # Save as JSON
        with open(f"{folder}/config.json", "w") as f:
            json.dump(cfg_dict, f, indent=4)

        # Also save as YAML for better readability
        try:
            with open(f"{folder}/config.yaml", "w") as f:
                yaml.dump(cfg_dict, f, default_flow_style=False)
            self.logger.info("Config saved as YAML.")
        except Exception as e:
            self.logger.warning(
                f"Failed to save config as YAML. Only JSON version available. Error: {e}"
            )

    def design(self, start, scorer):
        """
        Design a sequence using the given scorer.
        """
        raise NotImplementedError

    def save_result(self, results):
        itrs = sorted(results.keys())
        scores = [results[itr]["score"] for itr in itrs]
        max_scores = [results[itr]["max_score"] for itr in itrs]
        seqs = [results[itr]["sequence"] for itr in itrs]

        # Plot score vs iteration
        plt.figure(figsize=(10, 6))
        # plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 14})
        plt.plot(itrs, scores, linewidth=2, label="Current Score")
        plt.plot(itrs, max_scores, linewidth=2, color="red", label="Max Score")
        plt.xlabel("Iteration", fontsize=25)
        plt.ylabel("Score", fontsize=25)
        plt.legend(fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/score_vs_iteration.png", dpi=300)
        plt.close()
        self.logger.info("Score vs iteration saved to png.")

        # Save to csv
        df = pd.DataFrame({"iteration": itrs, "score": scores, "sequence": seqs})
        df.to_csv(f"{self.output_dir}/score_vs_iteration.csv", index=False)
        self.logger.info("Score vs iteration saved to csv.")

    def log(self, itr, seq, score):
        if score > self.max_score:
            self.max_score = score
            self.best_sequence = seq

        # Log
        state = {
            "itr": itr,
            "sequence": seq,
            "score": score,
            "max_score": self.max_score,
            "best_sequence": self.best_sequence,
        }
        self.results[itr] = {
            "sequence": seq,
            "score": score,
            "max_score": self.max_score,
        }
        if self.cfg.use_wandb:
            wandb.log(state)
