from queue import PriorityQueue
from tqdm import tqdm

from .base_designer import Designer
from .ga_designer import mutate_sequence
from .random_single_designer import generate_random_single_sequence


class BaseSearchDesigner(Designer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.decay = cfg.design["decay"]
        self.mutation_rate = cfg.design.get("mutation_rate", 0)

    def get_top(self, priority_queue):
        value, seq = priority_queue.get()
        priority_queue.put((self.decay * value, seq))
        return seq

    def design(self, scorer):
        score = scorer.predict(self.start)
        self.log(scorer.cnt, self.start, score)

        priority_queue = PriorityQueue()
        priority_queue.put((-score, self.start))

        visited = set()
        bar = tqdm(range(self.num_iterations), desc="Designing", dynamic_ncols=True)
        itr = scorer.cnt
        while itr < self.num_iterations:
            start = self.get_top(priority_queue)

            # Generate random sequence
            if self.mutation_rate > 0:
                seq = mutate_sequence(start, self.mutation_rate)
            else:
                while True:
                    seq = generate_random_single_sequence(start)
                    if seq not in visited:
                        break

            # Predict score
            score = scorer.predict(seq)
            self.log(scorer.cnt, seq, score)

            # Record
            visited.add(seq)
            priority_queue.put((-score, seq))  # negative score for max heap

            bar.update(scorer.cnt - itr)
            itr = scorer.cnt

        self.logger.info("Designing finished.")
        self.save_result(self.results)
