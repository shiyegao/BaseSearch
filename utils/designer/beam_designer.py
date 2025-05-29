import random
from tqdm import tqdm
import heapq

from .base_designer import Designer


DNA_BASES = ["A", "T", "C", "G"]


def mutate_sequence(seq: str, mutation_rate: float = 0.1) -> str:
    """
    Mutate a DNA sequence with a given mutation rate.
    Each position has a mutation_rate probability of being changed to another base.
    """
    mutated_seq = ""
    for base in seq:
        if random.random() < mutation_rate:
            # Choose a different base
            other_bases = [b for b in DNA_BASES if b != base]
            mutated_seq += random.choice(other_bases)
        else:
            mutated_seq += base
    return mutated_seq


class BeamDesigner(Designer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.beam_width = cfg.design.get("beam_width", 10)  # Default beam width is 10
        self.num_children = cfg.design.get(
            "num_children", 5
        )  # Number of children per beam sequence
        self.mutation_rate = cfg.design.get(
            "mutation_rate", 0.1
        )  # Mutation rate for generating children

    def design(self, scorer):
        # Initialize beam with the start sequence
        beam = [(scorer.predict(self.start), self.start)]
        self.log(scorer.cnt, self.start, beam[0][0])

        visited = set()
        visited.add(self.start)

        # Main beam search loop
        pbar = tqdm(total=self.num_iterations, desc="Beam Search", dynamic_ncols=True)
        while True:
            iteration = scorer.cnt
            if iteration >= self.num_iterations:
                break

            # Generate children for each sequence in the beam
            candidates = []
            for _, parent_seq in beam:
                for _ in range(self.num_children):
                    child_seq = mutate_sequence(parent_seq, self.mutation_rate)
                    if child_seq not in visited:
                        visited.add(child_seq)
                        score = scorer.predict(child_seq)
                        self.log(scorer.cnt, child_seq, score)
                        candidates.append((score, child_seq))

            # Update beam with top candidates
            beam = heapq.nlargest(self.beam_width, candidates, key=lambda x: x[0])
            pbar.update(scorer.cnt - iteration)

        self.logger.info("Beam search finished.")
        self.save_result(self.results)
        return self.best_sequence
