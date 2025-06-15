import random
from tqdm import tqdm

from .base_designer import DNA_BASES
from .random_designer import RandomDesigner


def generate_random_single_sequence(start: str):
    """
    Generate a random single sequence from the start sequence.
    The sequence is different from the start sequence only on one position.
    """
    length = len(start)
    pos = random.randint(0, length - 1)
    _dna_bases = DNA_BASES.copy()
    _dna_bases.remove(start[pos])
    seq = start[:pos] + random.choice(_dna_bases) + start[pos + 1 :]
    return seq


class RandomSingleDesigner(RandomDesigner):
    def design(self, scorer):
        visited = set()
        bar = tqdm(range(self.num_iterations), desc="Designing", dynamic_ncols=True)
        seq_prev = self.start
        while scorer.cnt < self.num_iterations:
            itr = scorer.cnt

            # Generate random sequence
            while True:
                seq = generate_random_single_sequence(seq_prev)
                if seq not in visited:
                    break

            # Predict score
            score = scorer.predict(seq)

            # Record
            visited.add(seq)
            self.log(scorer.cnt, seq, score)

            bar.update(scorer.cnt - itr)
            seq_prev = seq

        self.logger.info("Designing finished.")
        self.save_result(self.results)
