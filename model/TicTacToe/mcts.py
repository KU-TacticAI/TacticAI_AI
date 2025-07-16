"""
AlphaZero용 틱택토 MCTS

틱택토 게임에 맞춘 MCTS 구현입니다.
"""

import numpy as np
import torch
from collections import defaultdict
import copy
from typing import Optional

from .game_state import TicTacToeGameState
from .neural_network import TicTacToeNeuralNetwork

class MCTS:
    def __init__(self, neural_network: TicTacToeNeuralNetwork, num_simulations: int = 100, c_puct: float = 1.0, temperature: float = 1.0, device: Optional[torch.device] = None, batch_size: int = 16):
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size

    def search(self, root_state: TicTacToeGameState) -> np.ndarray:
        N = defaultdict(int)  # 방문 횟수
        W = defaultdict(float)  # 가치 합
        Q = defaultdict(float)  # 평균 가치
        P = dict()  # 정책 prior
        states = dict()  # 상태 캐시
        root_key = self._state_to_key(root_state)
        valid_moves = root_state.get_valid_moves()
        action_mask = np.zeros(9, dtype=np.float32)
        for move in valid_moves:
            action_mask[root_state.get_action_index(move)] = 1.0
        # 초기 정책/가치 예측
        policy, _ = self.neural_network.predict(root_state.get_encoded_state(), self.device)
        policy = policy * action_mask
        if np.sum(policy) > 0:
            policy = policy / np.sum(policy)
        else:
            policy = action_mask / np.sum(action_mask)
        P[root_key] = policy
        # 시뮬레이션 반복
        for _ in range(self.num_simulations):
            self._simulate(root_state, N, W, Q, P, states)
        # 방문 횟수 기반 정책 반환
        counts = np.zeros(9, dtype=np.float32)
        for move in valid_moves:
            idx = root_state.get_action_index(move)
            counts[idx] = N[(root_key, idx)]
        if self.temperature == 0:
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
        else:
            counts = counts ** (1.0 / self.temperature)
            if np.sum(counts) > 0:
                probs = counts / np.sum(counts)
            else:
                probs = action_mask / np.sum(action_mask)
        return probs

    def _simulate(self, state, N, W, Q, P, states):
        key = self._state_to_key(state)
        if state.is_game_over():
            winner = state.get_winner()
            if winner == 0:
                return 0.0  # 무승부
            return 1.0 if winner == state.current_player else -1.0
        if key not in P:
            # 신경망 예측
            policy, value = self.neural_network.predict(state.get_encoded_state(), self.device)
            valid_moves = state.get_valid_moves()
            action_mask = np.zeros(9, dtype=np.float32)
            for move in valid_moves:
                action_mask[state.get_action_index(move)] = 1.0
            policy = policy * action_mask
            if np.sum(policy) > 0:
                policy = policy / np.sum(policy)
            else:
                policy = action_mask / np.sum(action_mask)
            P[key] = policy
            return value
        # UCB로 액션 선택
        valid_moves = state.get_valid_moves()
        best_score = -float('inf')
        best_action = None
        for move in valid_moves:
            idx = state.get_action_index(move)
            u = self.c_puct * P[key][idx] * np.sqrt(sum(N[(key, state.get_action_index(m))] for m in valid_moves) + 1) / (1 + N[(key, idx)])
            score = Q[(key, idx)] + u
            if score > best_score:
                best_score = score
                best_action = move
        next_state = state.make_move(best_action)
        idx = state.get_action_index(best_action)
        v = self._simulate(next_state, N, W, Q, P, states)
        N[(key, idx)] += 1
        W[(key, idx)] += v if state.current_player == next_state.current_player * -1 else -v
        Q[(key, idx)] = W[(key, idx)] / N[(key, idx)]
        return -v

    def _state_to_key(self, state: TicTacToeGameState):
        return tuple(state.board.flatten()), state.current_player

    def clear_cache(self):
        """캐시 초기화 (호환성을 위한 메서드)"""
        # 현재 구현에서는 캐시가 없으므로 아무것도 하지 않음
        pass 