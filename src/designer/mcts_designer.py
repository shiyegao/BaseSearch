import random
import math
from tqdm import tqdm
import numpy as np

from .base_designer import Designer, DNA_BASES


class Node:
    def __init__(self, sequence, parent=None):
        self.sequence = sequence
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = list(range(len(sequence)))
        self.score = None  # Store the actual score of this node

    def add_child(self, position, new_base, new_sequence):
        child = Node(new_sequence, self)
        self.untried_actions.remove(position)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.value += result

    def ucb_score(self, c_param=1.414):
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return float("inf")  # Handle case where parent has no visits
        exploitation = self.value / self.visits
        exploration = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c_param=1.414):
        """Select the child with the highest UCB score"""
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.ucb_score(c_param))

    def is_fully_expanded(self):
        """Check if all possible actions have been tried"""
        return len(self.untried_actions) == 0

    def is_terminal(self):
        """Check if the node is a terminal node (no more actions possible)"""
        return self.is_fully_expanded() and not self.children


class MCTSDesigner(Designer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.simulations_per_iter = cfg.design.get("simulations_per_iter", 10)
        self.c_param = cfg.design.get("exploration_weight", 1.414)
        self.rollout_depth = cfg.design.get("rollout_depth", 3)
        self.reuse_tree = cfg.design.get(
            "reuse_tree", True
        )  # Option to enable/disable tree reuse
        self.tree_depth_limit = cfg.design.get(
            "tree_depth_limit", 100
        )  # Limit tree size
        self.temperature = cfg.design.get(
            "temperature", 0.1
        )  # Temperature for score-based sampling
        self.results = dict()

    def design(self, scorer):
        self.results = dict()
        self.max_score = 0
        self.best_sequence = self.start
        bar = tqdm(range(self.num_iterations), desc="Designing", dynamic_ncols=True)
        itr = scorer.cnt

        # Initialize the root node
        self.root = Node(self.start)
        root = self.root

        # Evaluate root node once
        root.score = scorer.predict(root.sequence)
        self.log(scorer.cnt, root.sequence, root.score)

        while itr < self.num_iterations:
            # Run MCTS simulations
            for _ in range(self.simulations_per_iter):
                # Selection and expansion
                leaf = self._tree_policy(root)

                # Simulation (rollout)
                rollout_result = self._rollout(leaf, scorer)

                # Backpropagation
                self._backpropagate(leaf, rollout_result)

            # Select the next root node based on score-weighted probability from all nodes
            if self.reuse_tree:
                # Get all nodes in the tree and their scores
                all_nodes = self._get_all_nodes(self.root)

                # Filter nodes with scores and evaluate those without scores
                scored_nodes = []
                for node in all_nodes:
                    if node.score is not None:
                        scored_nodes.append(node)

                if scored_nodes:
                    # Select next root based on softmax probability of scores
                    root = self._sample_node_by_score(scored_nodes)

            # Update iteration counter
            bar.update(scorer.cnt - itr)
            itr = scorer.cnt

            # Prune tree if it gets too large
            if self._count_nodes(self.root) > self.tree_depth_limit:
                self._prune_tree(self.root)

        self.logger.info("Designing finished.")
        self.save_result(self.results)
        return self.best_sequence

    def _sample_node_by_score(self, nodes):
        """Sample a node based on softmax probability of scores"""
        if not nodes:
            return None

        # Apply temperature to control exploration/exploitation
        scores = np.array([node.score for node in nodes])

        # Handle potential numerical issues with very large or very different scores
        scores = scores - np.max(scores)  # Shift to prevent overflow

        # Apply softmax with temperature
        exp_scores = np.exp(scores / self.temperature)
        probs = exp_scores / np.sum(exp_scores)

        # Sample a node based on the calculated probabilities
        return np.random.choice(nodes, p=probs)

    def _tree_policy(self, node):
        """Select which node to expand next using UCT"""
        current = node
        # Keep going down the tree until we reach a leaf node
        while not current.is_terminal():
            if not current.is_fully_expanded():
                return self._expand(current)
            else:
                next_node = current.best_child(self.c_param)
                if next_node is None:
                    return current
                current = next_node
        return current

    def _expand(self, node):
        """Choose an unexplored action and add a child node for it"""
        if not node.untried_actions:
            return node

        position = random.choice(node.untried_actions)
        seq = node.sequence
        bases = DNA_BASES.copy()
        if position < len(seq):
            current_base = seq[position]
            bases.remove(current_base)
        new_base = random.choice(bases)
        new_sequence = seq[:position] + new_base + seq[position + 1 :]
        child = node.add_child(position, new_base, new_sequence)
        return child

    def _rollout(self, node, scorer):
        """Perform a random rollout from the given node and return the score"""
        current_sequence = node.sequence

        # Make a few random changes to the sequence
        for _ in range(self.rollout_depth):
            position = random.randint(0, len(current_sequence) - 1)
            bases = DNA_BASES.copy()
            current_base = current_sequence[position]
            bases.remove(current_base)
            new_base = random.choice(bases)
            current_sequence = (
                current_sequence[:position]
                + new_base
                + current_sequence[position + 1 :]
            )

        # Get score and update global best if needed
        score = scorer.predict(current_sequence)
        self.log(scorer.cnt, current_sequence, score)
        return score

    def _backpropagate(self, node, result):
        """Update the statistics of nodes from the given node to the root"""
        current = node
        while current:
            current.update(result)
            current = current.parent

    def _count_nodes(self, node):
        """Count the number of nodes in the tree"""
        count = 1  # Count the current node
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _prune_tree(self, root):
        """Prune the tree to keep only the most promising branches"""
        # Simple pruning: keep only the best N children for each node
        max_children = 5
        for node in self._get_all_nodes(root):
            if len(node.children) > max_children:
                # Sort children by actual score if available, otherwise by value/visits
                node.children.sort(
                    key=lambda child: child.score
                    if child.score is not None
                    else child.value / max(1, child.visits),
                    reverse=True,
                )
                node.children = node.children[:max_children]
        return root

    def _get_all_nodes(self, root):
        """Get all nodes in the tree using BFS"""
        all_nodes = [root]
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child in node.children:
                all_nodes.append(child)
                queue.append(child)
        return all_nodes
