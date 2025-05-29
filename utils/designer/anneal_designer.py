import random
import numpy as np
from tqdm import tqdm

from .base_designer import Designer
from .ga_designer import mutate_sequence


class AnnealDesigner(Designer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mutation_rate = cfg.design["mutation_rate"]
        self.initial_temp = cfg.design["initial_temp"]
        self.min_temp = cfg.design["min_temp"]
        self.cooling_rate = cfg.design["cooling_rate"]

    def design(self, scorer):
        visited = set()

        # Start with random sequence
        current_seq = self.start
        current_score = scorer.predict(current_seq)
        self.log(scorer.cnt, current_seq, current_score)

        # Initialize temperature
        temperature = self.initial_temp

        bar = tqdm(range(self.num_iterations), desc="Annealing", dynamic_ncols=True)
        itr = scorer.cnt
        while itr < self.num_iterations:
            # Generate a neighbor by mutation
            candidate_seq = mutate_sequence(current_seq, self.mutation_rate)
            if candidate_seq in visited:
                continue

            # Score the candidate
            candidate_score = scorer.predict(candidate_seq)
            self.log(scorer.cnt, candidate_seq, candidate_score)
            visited.add(candidate_seq)

            # Calculate acceptance probability
            delta = candidate_score - current_score
            acceptance_prob = min(1.0, np.exp(delta / temperature))

            # Accept or reject
            if delta > 0 or random.random() < acceptance_prob:
                current_seq = candidate_seq
                current_score = candidate_score

            # Cool down
            if itr > 0 and itr % 100 == 0:
                temperature = max(self.min_temp, temperature * self.cooling_rate)

            # Update and log
            bar.update(scorer.cnt - itr)
            itr = scorer.cnt

        self.logger.info("Annealing design finished.")
        self.logger.info(
            f"Best sequence: {self.best_sequence} with score: {self.max_score}"
        )
        self.save_result(self.results)
