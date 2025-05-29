import random
import numpy as np
from tqdm import tqdm

from .base_designer import Designer, DNA_BASES


def crossover(parent1, parent2, crossover_rate=0.7):
    """
    Perform crossover between two parent sequences.
    Returns two child sequences.
    """
    if random.random() > crossover_rate:
        return parent1, parent2

    # Single point crossover
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


def mutate_sequence(sequence, mutation_rate):
    """
    Mutate a sequence with a given mutation rate.
    Each position has a mutation_rate probability of being changed.
    """
    mutated = list(sequence)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Choose a base different from the current one
            options = [b for b in DNA_BASES if b != mutated[i]]
            mutated[i] = random.choice(options)
    return "".join(mutated)


class GADesigner(Designer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.population_size = cfg.design["population_size"]
        self.mutation_rate = cfg.design["mutation_rate"]
        self.crossover_rate = cfg.design["crossover_rate"]
        self.elitism_rate = cfg.design["elitism_rate"]

    def _initialize_population(self, sequence_length):
        """
        Initialize a random population of sequences.
        """
        population = []
        # Start with the initial sequence
        population.append(self.start)

        # Generate the rest randomly
        for _ in range(self.population_size - 1):
            seq = "".join(random.choice(DNA_BASES) for _ in range(sequence_length))
            population.append(seq)

        return population

    def _select_parent(self, population, scores):
        """
        Select a parent using tournament selection.
        """
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx]

    def design(self, scorer):
        visited = set()

        # Initialize population
        sequence_length = len(self.start)
        population = self._initialize_population(sequence_length)

        # Evaluate initial population
        scores = []
        for seq in population:
            score = scorer.predict(seq)
            scores.append(score)
            self.log(scorer.cnt, seq, score)

        # Main GA loop
        bar = tqdm(
            total=self.num_iterations, desc="Genetic Algorithm", dynamic_ncols=True
        )
        itr = scorer.cnt

        while itr < self.num_iterations:
            # Elitism: keep best individuals
            elites_count = max(1, int(self.population_size * self.elitism_rate))
            elite_indices = np.argsort(scores)[-elites_count:]
            new_population = [population[i] for i in elite_indices]

            # Generate new individuals
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._select_parent(population, scores)
                parent2 = self._select_parent(population, scores)

                # Crossover
                child1, child2 = crossover(parent1, parent2, self.crossover_rate)

                # Mutation
                child1 = mutate_sequence(child1, self.mutation_rate)
                child2 = mutate_sequence(child2, self.mutation_rate)

                # Add to new population if not visited
                for child in [child1, child2]:
                    if len(new_population) < self.population_size:
                        new_population.append(child)

            # Replace population
            population = new_population

            # Evaluate new population
            scores = []
            for seq in population:
                if seq in visited:
                    # Get previous score
                    for i, s in self.results.items():
                        if s["sequence"] == seq:
                            score = s["score"]
                            break
                else:
                    score = scorer.predict(seq)
                    self.log(scorer.cnt, seq, score)
                    visited.add(seq)
                scores.append(score)

            # Update progress bar
            bar.update(scorer.cnt - itr)
            itr = scorer.cnt

        self.logger.info("Genetic algorithm design finished.")
        self.logger.info(
            f"Best sequence: {self.best_sequence} with score: {self.max_score}"
        )
        self.save_result(self.results)
