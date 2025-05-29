import random
from tqdm import tqdm

from .base_designer import Designer, DNA_BASES


def generate_random_sequence(start: str):
    """
    Generate a random sequence from the start sequence.
    The sequence is different from the start sequence on some positions.
    """
    length = len(start)
    seq = ""
    for i in range(length):
        seq += random.choice(DNA_BASES)
    return seq


class RandomDesigner(Designer):
    def design(self, scorer):
        visited = set()
        bar = tqdm(range(self.num_iterations), desc="Designing", dynamic_ncols=True)
        itr = scorer.cnt
        while itr < self.num_iterations:
            # Generate random sequence
            while True:
                seq = generate_random_sequence(self.start)
                if seq not in visited:
                    break

            # Predict score
            score = scorer.predict(seq)
            self.log(scorer.cnt, seq, score)

            # Record
            visited.add(seq)
            bar.update(scorer.cnt - itr)
            itr = scorer.cnt

        self.logger.info("Designing finished.")
        self.save_result(self.results)
