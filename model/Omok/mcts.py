"""
MCTS for Omok (Gomoku) implementing AlphaZero-style batched NN predictions and optimized search.
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from .game_state import OmokGameState
from .neural_network import OmokNeuralNetwork
import torch
import random


class MCTSNode:
    def __init__(self, state: OmokGameState, parent: Optional['MCTSNode'] = None, prior_prob: float = 1.0):
        self.state = state
        self.parent = parent
        self.prior_prob = prior_prob
        self.children: Dict[int, 'MCTSNode'] = {}  # key: action_index
        self.visit_count = 0
        self.value_sum = 0.0
        self.mean_value = 0.0
        self.valid_actions = state.get_valid_moves()
        self.is_terminal = state.is_game_over()
        self._ucb_scores = None

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.valid_actions)

    def get_ucb_scores_vectorized(self, c_puct: float = 1.0) -> np.ndarray:
        if not self.children:
            return np.array([])
        if self.parent is None:
            children_list = list(self.children.values())
            visit_counts = np.array([child.visit_count for child in children_list])
            ucb_scores = np.where(visit_counts == 0, np.inf, 0.0)
            self._ucb_scores = ucb_scores
            return ucb_scores
        if self._ucb_scores is not None and self._ucb_scores_c_puct == c_puct and self._ucb_cache_parent_visit == self.parent.visit_count:
            return self._ucb_scores
        children_list = list(self.children.values())
        visit_counts = np.array([child.visit_count for child in children_list])
        mean_values = np.array([child.mean_value for child in children_list])
        prior_probs = np.array([child.prior_prob for child in children_list])
        parent_visit_sqrt = np.sqrt(self.parent.visit_count)
        ucb_scores = (mean_values + c_puct * prior_probs * parent_visit_sqrt / (1 + visit_counts))
        ucb_scores[visit_counts == 0] = np.inf
        self._ucb_scores = ucb_scores
        return ucb_scores

    def select_child(self, c_puct: float = 1.0) -> Optional['MCTSNode']:
        if not self.children:
            return None
        ucb_scores = self.get_ucb_scores_vectorized(c_puct)
        best_idx = np.argmax(ucb_scores)
        return list(self.children.values())[best_idx]

    def expand(self, action_index: int, prior_prob: float) -> 'MCTSNode':
        if action_index in self.children:
            return self.children[action_index]
        action = self.state.get_action_from_index(action_index)
        new_state = self.state.make_move(action)
        child = MCTSNode(new_state, parent=self, prior_prob=prior_prob)
        self.children[action_index] = child
        self._ucb_scores = None
        return child

    def update(self, value: float):
        self.visit_count += 1
        self.value_sum += value
        self.mean_value = self.value_sum / self.visit_count
        self._ucb_scores = None

    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        probs = np.zeros(self.state.board_size * self.state.board_size, dtype=np.float32)
        if self.visit_count == 0 or not self.children:
            return probs
        children_list = list(self.children.values())
        visit_counts = np.array([child.visit_count for child in children_list])
        action_indices = list(self.children.keys())
        if temperature == 0:
            max_visit_idx = np.argmax(visit_counts)
            probs[action_indices[max_visit_idx]] = 1.0
        else:
            temp_probs = visit_counts ** (1.0 / temperature)
            for i, ai in enumerate(action_indices):
                probs[ai] = temp_probs[i]
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs /= prob_sum
        return probs

    def get_best_action(self) -> Optional[Tuple[int, int]]:
        if not self.children:
            return None
        best_action_index = max(self.children.keys(), key=lambda a: self.children[a].visit_count)
        return self.state.get_action_from_index(best_action_index)


class MCTS:
    def __init__(self, neural_network: OmokNeuralNetwork, num_simulations: int = 800, c_puct: float = 1.0, temperature: float = 1.0, device: Optional[torch.device] = None, batch_size: int = 32):
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size
        self.state_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.batch_queue: List[Tuple[OmokGameState, str]] = []

    def _get_state_key(self, state: OmokGameState) -> str:
        return f"{hash(state.board.tobytes())}_{state.current_player}"

    def _get_neural_network_prediction_batch(self, state: OmokGameState) -> Tuple[np.ndarray, float]:
        state_key = self._get_state_key(state)
        if state_key in self.state_cache:
            return self.state_cache[state_key]
        self.batch_queue.append((state, state_key))
        if len(self.batch_queue) >= self.batch_size:
            self._process_batch_predictions()
        if state_key in self.state_cache:
            return self.state_cache[state_key]
        self._process_batch_predictions()
        return self.state_cache[state_key]

    def _process_batch_predictions(self) -> None:
        if not self.batch_queue:
            return
        batch_states = []
        batch_keys = []
        for state, key in self.batch_queue:
            if key not in self.state_cache:
                batch_states.append(state.get_encoded_state())
                batch_keys.append(key)
        if not batch_states:
            self.batch_queue = []
            return
        states_array = np.array(batch_states)
        policies, values = self.neural_network.predict_batch(states_array, self.device)
        for i, key in enumerate(batch_keys):
            policy = policies[i]
            value = values[i]
            state = self.batch_queue[i][0]
            action_mask = state.get_action_mask()
            policy = policy * action_mask
            prob_sum = np.sum(policy)
            if prob_sum > 0:
                policy /= prob_sum
            self.state_cache[key] = (policy, value)
        self.batch_queue = []

    def _get_neural_network_prediction(self, state: OmokGameState) -> Tuple[np.ndarray, float]:
        key = self._get_state_key(state)
        if key in self.state_cache:
            return self.state_cache[key]
        policy, value = self.neural_network.predict(state.get_encoded_state(), self.device)
        action_mask = state.get_action_mask()
        policy = policy * action_mask
        if np.sum(policy) > 0:
            policy /= np.sum(policy)
        self.state_cache[key] = (policy, value)
        return policy, value

    def search(self, state: OmokGameState) -> np.ndarray:
        root = MCTSNode(state)
        for _ in range(self.num_simulations):
            node = root
            # selection
            while node.is_fully_expanded() and not node.is_terminal:
                child = node.select_child(self.c_puct)
                if child is None:
                    break
                node = child
            # expansion
            if not node.is_terminal:
                policy, value = self._get_neural_network_prediction_batch(node.state)
                for action in node.valid_actions:
                    ai = node.state.get_action_index(action)
                    prior = policy[ai]
                    node.expand(ai, prior)
                if node != root:
                    value = self._simulate(node.state)
                else:
                    winner = node.state.get_winner()
                    if winner is None:
                        value = 0.0
                    else:
                        value = 1.0 if winner == 1 else -1.0
            # backprop
            while node is not None:
                node.update(value)
                node = node.parent
                value = -value
        self._process_batch_predictions()
        return root.get_action_probs(self.temperature)

    def _simulate(self, state: OmokGameState) -> float:
        cur = state
        max_moves = state.board_size * state.board_size
        moves = 0
        while not cur.is_game_over() and moves < max_moves:
            valid = cur.get_valid_moves()
            if not valid:
                break
            mv = random.choice(valid)
            cur = cur.make_move(mv)
            moves += 1
        if cur.is_game_over():
            winner = cur.get_winner()
            if winner == 1:
                return 1.0
            elif winner == -1:
                return -1.0
            else:
                return 0.0
        return 0.0

    def get_action(self, state: OmokGameState) -> Tuple[int, int]:
        probs = self.search(state)
        if self.temperature == 0:
            ai = int(np.argmax(probs))
        else:
            ai = int(np.random.choice(len(probs), p=probs))
        return state.get_action_from_index(ai)

    def get_best_action(self, state: OmokGameState) -> Tuple[int, int]:
        orig_temp = self.temperature
        self.temperature = 0
        action = self.get_action(state)
        self.temperature = orig_temp
        return action

    def clear_cache(self):
        self.state_cache.clear()
        self.batch_queue.clear()
