"""
MCTS (Monte Carlo Tree Search) 구현

알파제로에서 사용하는 MCTS 알고리즘을 구현합니다.
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from .game_state import OthelloGameState
from .neural_network import OthelloNeuralNetwork
import torch
import random

class MCTSNode:
    """MCTS 노드 클래스"""
    
    def __init__(self, 
                 state: OthelloGameState, 
                 parent: Optional['MCTSNode'] = None,
                 prior_prob: float = 1.0):
        """
        MCTS 노드 초기화
        
        Args:
            state: 게임 상태
            parent: 부모 노드
            prior_prob: 사전 확률 (신경망에서 예측한 확률)
        """
        self.state = state
        self.parent = parent
        self.prior_prob = prior_prob
        
        # 자식 노드들
        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}
        
        # MCTS 통계
        self.visit_count = 0
        self.value_sum = 0.0
        self.mean_value = 0.0
        
        # 유효한 액션들
        self.valid_actions = state.get_valid_moves()
        
        # 터미널 상태 여부
        self.is_terminal = state.is_game_over()
        
        # NumPy 최적화를 위한 캐시
        self._ucb_scores = None
        self._ucb_cache_c_puct = None
        self._ucb_cache_parent_visit = None
        self._sorted_children = None
        self._sorted_children_c_puct = None
    
    def is_fully_expanded(self) -> bool:
        """모든 유효한 액션이 확장되었는지 확인"""
        return len(self.children) == len(self.valid_actions)
    
    def get_ucb_score(self, c_puct: float = 1.0) -> float:
        """UCB (Upper Confidence Bound) 점수 계산"""
        if self.visit_count == 0:
            return float('inf')
        
        # 캐시 확인
        if (self._ucb_scores is not None and 
            self._ucb_cache_c_puct == c_puct and
            self._ucb_cache_parent_visit == self.parent.visit_count):
            return self._ucb_scores[0]  # 단일 노드이므로 첫 번째 값
        
        # UCB1 공식: mean_value + c_puct * prior_prob * sqrt(parent_visit) / (1 + visit_count)
        ucb = (self.mean_value + 
               c_puct * self.prior_prob * 
               math.sqrt(self.parent.visit_count) / (1 + self.visit_count))
        
        return ucb
    
    def get_ucb_scores_vectorized(self, c_puct: float = 1.0) -> np.ndarray:
        """모든 자식 노드의 UCB 점수를 한 번에 계산"""
        if not self.children:
            return np.array([])
        
        # 루트 노드인 경우 parent가 None이므로 처리
        if self.parent is None:
            # 루트 노드의 경우 모든 자식 노드가 동일한 우선순위를 가짐
            children_list = list(self.children.values())
            visit_counts = np.array([child.visit_count for child in children_list])
            
            # 방문하지 않은 노드는 무한대, 방문한 노드는 0
            ucb_scores = np.where(visit_counts == 0, np.inf, 0.0)
            
            # 캐시 저장
            self._ucb_scores = ucb_scores
            self._ucb_cache_c_puct = c_puct
            self._ucb_cache_parent_visit = 0
            
            return ucb_scores
        
        # 캐시 확인
        if (self._ucb_scores is not None and 
            self._ucb_cache_c_puct == c_puct and
            self._ucb_cache_parent_visit == self.parent.visit_count):
            return self._ucb_scores
        
        # 모든 자식 노드의 통계를 NumPy 배열로 변환
        children_list = list(self.children.values())
        visit_counts = np.array([child.visit_count for child in children_list])
        mean_values = np.array([child.mean_value for child in children_list])
        prior_probs = np.array([child.prior_prob for child in children_list])
        
        # UCB 계산 (벡터화)
        parent_visit_sqrt = np.sqrt(self.parent.visit_count)
        ucb_scores = (mean_values + 
                     c_puct * prior_probs * 
                     parent_visit_sqrt / (1 + visit_counts))
        
        # 방문하지 않은 노드는 무한대
        ucb_scores[visit_counts == 0] = np.inf
        
        # 캐시 저장
        self._ucb_scores = ucb_scores
        self._ucb_cache_c_puct = c_puct
        self._ucb_cache_parent_visit = self.parent.visit_count
        
        return ucb_scores
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """NumPy 벡터화된 자식 노드 선택"""
        if not self.children:
            return None
        
        ucb_scores = self.get_ucb_scores_vectorized(c_puct)
        best_idx = np.argmax(ucb_scores)
        
        return list(self.children.values())[best_idx]
    
    def expand(self, action: Tuple[int, int], prior_prob: float) -> 'MCTSNode':
        """새로운 자식 노드 확장"""
        if action in self.children:
            return self.children[action]
        
        # 액션 실행하여 새로운 상태 생성
        new_state = self.state.make_move(action)
        child = MCTSNode(new_state, parent=self, prior_prob=prior_prob)
        self.children[action] = child
        
        # 캐시 무효화
        self._ucb_scores = None
        self._sorted_children = None
        
        return child
    
    def update(self, value: float):
        """노드 통계 업데이트"""
        self.visit_count += 1
        self.value_sum += value
        self.mean_value = self.value_sum / self.visit_count
        
        # 캐시 무효화
        self._ucb_scores = None
        self._sorted_children = None
    
    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """
        NumPy 벡터화된 액션 확률 반환 (65,)
        """
        probs = np.zeros(65, dtype=np.float32)
        
        if self.visit_count == 0 or not self.children:
            return probs
        
        # 모든 자식 노드의 방문 횟수를 NumPy 배열로 변환
        children_list = list(self.children.values())
        visit_counts = np.array([child.visit_count for child in children_list])
        actions = list(self.children.keys())
        
        # 온도에 따른 확률 계산
        if temperature == 0:
            # 가장 많이 방문한 액션만 1.0
            max_visit_idx = np.argmax(visit_counts)
            action_idx = self.state.get_action_index(actions[max_visit_idx])
            probs[action_idx] = 1.0
        else:
            # 온도 적용
            temp_probs = visit_counts ** (1.0 / temperature)
            
            # 액션 인덱스에 확률 할당
            for i, action in enumerate(actions):
                action_idx = self.state.get_action_index(action)
                probs[action_idx] = temp_probs[i]
        
        # 정규화
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs /= prob_sum
        
        return probs
    
    def get_best_action(self) -> Optional[Tuple[int, int]]:
        """가장 많이 방문한 액션 반환"""
        if not self.children:
            return None
        
        best_action = max(self.children.keys(), 
                         key=lambda action: self.children[action].visit_count)
        return best_action


class MCTS:
    """Monte Carlo Tree Search 클래스"""
    
    def __init__(self, 
                 neural_network: OthelloNeuralNetwork,
                 num_simulations: int = 800,
                 c_puct: float = 1.0,
                 temperature: float = 1.0,
                 device: Optional[torch.device] = None,
                 batch_size: int = 32):
        """
        MCTS 초기화
        
        Args:
            neural_network: 신경망 모델
            num_simulations: 시뮬레이션 횟수
            c_puct: UCB 탐색 상수
            temperature: 온도 파라미터
            device: 사용할 디바이스
            batch_size: 배치 크기
        """
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size
        
        # 캐시 (상태 -> 정책, 가치)
        self.state_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        
        # 배치 처리를 위한 대기열
        self.batch_queue: List[Tuple[OthelloGameState, str]] = []
        
        # NumPy 최적화를 위한 캐시
        self._state_key_cache = {}
        self._action_mask_cache = {}
    
    def _get_state_key(self, state: OthelloGameState) -> str:
        """캐시된 상태 키 생성"""
        # NumPy 해시 사용
        board_hash = hash(state.board.tobytes())
        key = f"{board_hash}_{state.current_player}"
        return key
    
    def _get_cached_action_mask(self, state: OthelloGameState) -> np.ndarray:
        """캐시된 액션 마스크"""
        state_key = self._get_state_key(state)
        if state_key not in self._action_mask_cache:
            self._action_mask_cache[state_key] = state.get_action_mask()
        return self._action_mask_cache[state_key]
    
    def _get_neural_network_prediction(self, state: OthelloGameState) -> Tuple[np.ndarray, float]:
        """신경망에서 정책과 가치 예측 (단일 상태)"""
        state_key = self._get_state_key(state)
        
        # 캐시 확인
        if state_key in self.state_cache:
            return self.state_cache[state_key]
        
        # 신경망 예측
        encoded_state = state.get_encoded_state()
        policy, value = self.neural_network.predict(encoded_state, self.device)
        
        # 유효한 액션만 마스킹
        action_mask = state.get_action_mask()  # (65,)
        policy = policy * action_mask
        
        # 정규화
        if np.sum(policy) > 0:
            policy /= np.sum(policy)
        
        # 캐시에 저장
        self.state_cache[state_key] = (policy, value)
        
        return policy, value
    
    def _process_batch_predictions(self) -> None:
        """NumPy 벡터화된 배치 처리"""
        if not self.batch_queue:
            return
        
        # NumPy 배열로 변환
        batch_states = []
        batch_keys = []
        batch_indices = []
        
        for i, (state, state_key) in enumerate(self.batch_queue):
            if state_key not in self.state_cache:
                batch_states.append(state.get_encoded_state())
                batch_keys.append(state_key)
                batch_indices.append(i)
        
        if not batch_states:
            return
        
        # NumPy 배치로 변환
        states_array = np.array(batch_states)
        policies, values = self.neural_network.predict_batch(states_array, self.device)
        
        # 벡터화된 마스킹 및 정규화
        for i, (state_key, state_idx) in enumerate(zip(batch_keys, batch_indices)):
            policy = policies[i]
            value = values[i]
            
            # 액션 마스킹
            state = self.batch_queue[state_idx][0]
            action_mask = state.get_action_mask()
            policy = policy * action_mask
            
            # 정규화
            prob_sum = np.sum(policy)
            if prob_sum > 0:
                policy /= prob_sum
            
            self.state_cache[state_key] = (policy, value)
        
        # 처리된 항목 제거
        self.batch_queue = [item for j, item in enumerate(self.batch_queue) 
                           if j not in batch_indices]
    
    def _get_neural_network_prediction_batch(self, state: OthelloGameState) -> Tuple[np.ndarray, float]:
        """신경망에서 정책과 가치 예측 (배치 처리)"""
        state_key = self._get_state_key(state)
        
        # 캐시 확인
        if state_key in self.state_cache:
            return self.state_cache[state_key]
        
        # 배치 대기열에 추가
        self.batch_queue.append((state, state_key))
        
        # 배치 크기에 도달하면 처리
        if len(self.batch_queue) >= self.batch_size:
            self._process_batch_predictions()
        
        # 캐시에서 결과 반환 (처리 후)
        if state_key in self.state_cache:
            return self.state_cache[state_key]
        
        # 아직 처리되지 않은 경우 즉시 처리
        self._process_batch_predictions()
        return self.state_cache[state_key]
    
    def search(self, state: OthelloGameState) -> np.ndarray:
        """
        MCTS 검색 수행
        
        Args:
            state: 현재 게임 상태
            
        Returns:
            액션 확률 배열 (65,)
        """
        # 루트 노드 생성
        root = MCTSNode(state)
        
        # 시뮬레이션 수행
        for sim_idx in range(self.num_simulations):
            node = root
            
            # 선택 (Selection)
            while node.is_fully_expanded() and not node.is_terminal:
                child = node.select_child(self.c_puct)
                if child is None:  # 자식이 없는 경우
                    break
                node = child
            
            # 확장 (Expansion)
            if not node.is_terminal:
                # 신경망 예측 (배치 처리 사용)
                policy, value = self._get_neural_network_prediction_batch(node.state)
                
                # 유효한 액션들에 대해 자식 노드 확장
                for action in node.valid_actions:
                    action_idx = node.state.get_action_index(action)
                    prior_prob = policy[action_idx]
                    node.expand(action, prior_prob)
                
                # 루트 노드가 아닌 경우 시뮬레이션
                if node != root:
                    value = self._simulate(node.state)
                else:
                    # 터미널 상태인 경우 게임 결과 사용
                    winner = node.state.get_winner()
                    if winner is None:
                        value = 0.0  # 무승부
                    else:
                        value = 1.0 if winner == 1 else -1.0
            
            # 백프로파게이션 (Backpropagation)
            while node is not None:
                node.update(value)
                node = node.parent
                value = -value  # 상대방 관점에서의 가치
        
        # 남은 배치 처리
        self._process_batch_predictions()
        
        # 액션 확률 반환
        action_probs = root.get_action_probs(self.temperature)
        
        return action_probs
    
    def _simulate(self, state: OthelloGameState) -> float:
        """
        NumPy 최적화된 랜덤 시뮬레이션 수행
        
        Args:
            state: 시작 상태
            
        Returns:
            시뮬레이션 결과 (1: 승리, -1: 패배, 0: 무승부)
        """
        current_state = state
        max_moves = 60  # 최대 수 제한 (무한 루프 방지)
        move_count = 0
        
        while not current_state.is_game_over() and move_count < max_moves:
            valid_moves = current_state.get_valid_moves()
            if not valid_moves:
                break
            
            # NumPy 랜덤 선택
            move_idx = np.random.randint(len(valid_moves))
            action = valid_moves[move_idx]
            
            current_state = current_state.make_move(action)
            move_count += 1
        
        # 게임 결과 계산
        if current_state.is_game_over():
            winner = current_state.get_winner()
            if winner == 1:
                return 1.0
            elif winner == -1:
                return -1.0
            else:
                return 0.0
        
        return 0.0
    
    def get_action(self, state: OthelloGameState) -> Tuple[int, int] or str:
        """
        MCTS를 사용하여 액션 선택
        
        Args:
            state: 현재 게임 상태
            
        Returns:
            선택된 액션
        """
        action_probs = self.search(state)
        
        # 온도에 따른 액션 선택
        if self.temperature == 0:
            # 가장 확률이 높은 액션 선택
            action_idx = np.argmax(action_probs)
        else:
            # 확률에 따른 액션 선택
            action_idx = np.random.choice(65, p=action_probs)
        
        return state.get_action_from_index(action_idx)
    
    def get_best_action(self, state: OthelloGameState) -> Tuple[int, int] or str:
        """
        가장 좋은 액션 반환 (온도 = 0)
        
        Args:
            state: 현재 게임 상태
            
        Returns:
            가장 좋은 액션
        """
        original_temperature = self.temperature
        self.temperature = 0
        action = self.get_action(state)
        self.temperature = original_temperature
        return action
    
    def clear_cache(self):
        """캐시 초기화"""
        self.state_cache.clear() 
        self.batch_queue.clear()
        self._state_key_cache.clear()
        self._action_mask_cache.clear() 