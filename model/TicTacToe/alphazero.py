"""
알파제로 메인 학습 클래스

알파제로 알고리즘의 전체 학습 과정을 관리합니다.
"""

import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
from typing import List, Optional, Tuple, Dict, Any
import multiprocessing as mp
from functools import partial
from datetime import datetime
import signal
import sys
import pickle
import gzip
import json

from .game_state import TicTacToeGameState
from .neural_network import TicTacToeNeuralNetwork, PolicyValueLoss
from .mcts import MCTS
from .performance_profiler import PerformanceProfiler, profile_function, profile_section, get_profiler


class GameData:
    """게임 데이터 클래스"""
    
    def __init__(self, states: List[np.ndarray], 
                 policies: List[np.ndarray], 
                 values: List[float]):
        """
        게임 데이터 초기화
        
        Args:
            states: 게임 상태들
            policies: 정책들
            values: 가치들
        """
        self.states = states
        self.policies = policies
        self.values = values
    
    def __len__(self):
        return len(self.states)


class AlphaZero:
    """알파제로 메인 클래스"""
    
    def __init__(self,
                 model_dir: str = "models",
                 num_iterations: int = 100,
                 num_episodes: int = 100,
                 num_mcts_sims: int = 800,
                 num_epochs: int = 10,
                 batch_size: int = 512,
                 learning_rate: float = 0.002,
                 c_puct: float = 1.0,
                 temperature: float = 1.0,
                 device: Optional[torch.device] = None,
                 mcts_batch_size: int = 32,
                 enable_profiling: bool = True):
        """
        알파제로 초기화
        
        Args:
            model_dir: 모델 저장 디렉토리
            num_iterations: 전체 반복 횟수
            num_episodes: 각 반복당 자기대전 게임 수
            num_mcts_sims: MCTS 시뮬레이션 수
            num_epochs: 신경망 학습 에포크 수
            batch_size: 배치 크기
            learning_rate: 학습률
            c_puct: MCTS UCB 상수
            temperature: 온도 파라미터
            device: 사용할 디바이스
        """
        self.model_dir = model_dir
        self.num_iterations = num_iterations
        self.num_episodes = num_episodes
        self.num_mcts_sims = num_mcts_sims
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.c_puct = c_puct
        self.temperature = temperature
        self.mcts_batch_size = mcts_batch_size
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 모델 초기화
        self.neural_network = TicTacToeNeuralNetwork().to(self.device)
        self.mcts = MCTS(self.neural_network, num_mcts_sims, c_puct, temperature, self.device, mcts_batch_size)
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=learning_rate)
        
        # 손실 함수
        self.criterion = PolicyValueLoss()
        
        # 데이터 저장소
        self.data_buffer = deque(maxlen=10000)
        
        # 모델 디렉토리를 절대 경로로 변환하고 생성
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 학습 통계
        self.iteration = 0
        self.best_win_rate = 0.0
        
        # 프로파일링 설정
        self.enable_profiling = enable_profiling
        if enable_profiling:
            log_file = os.path.join(self.model_dir, f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            self.profiler = PerformanceProfiler(log_file=log_file, enable_console=True)
        else:
            self.profiler = None
        
        # 프로세스 풀 저장용 변수
        self.current_pool = None
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def self_play_episode(self) -> GameData:
        states = []
        policies = []
        values = []
        state = TicTacToeGameState()
        while not state.is_game_over():
            action_probs = self.mcts.search(state)  # (9,)
            if np.sum(action_probs) > 0:
                action_probs = action_probs / np.sum(action_probs)
            else:
                action_probs = np.ones(9) / 9
            states.append(state.get_encoded_state())
            policies.append(action_probs)
            action_idx = np.random.choice(9, p=action_probs)
            action = state.get_action_from_index(action_idx)
            valid_moves = state.get_valid_moves()
            if action in valid_moves:
                state = state.make_move(action)
            else:
                if valid_moves:
                    action = random.choice(valid_moves)
                    state = state.make_move(action)
        winner = state.get_winner()
        if winner is None:
            game_value = 0.0
        else:
            game_value = 1.0 if winner == 1 else -1.0 if winner == -1 else 0.0
        for i in range(len(states)):
            if i % 2 == 0:
                values.append(game_value)
            else:
                values.append(-game_value)
        return GameData(states, policies, values)

    @staticmethod
    def _parallel_self_play_episode(episode_id: int, num_mcts_sims: int, temperature: float, device: str, mcts_batch_size: int = 32) -> Tuple[int, GameData]:
        """
        병렬 실행을 위한 자기대전 에피소드 (정적 메서드)
        
        Args:
            episode_id: 에피소드 ID
            num_mcts_sims: MCTS 시뮬레이션 수
            temperature: 온도 파라미터
            device: 사용할 디바이스
            mcts_batch_size: MCTS 배치 크기
            
        Returns:
            (episode_id, game_data) 튜플
        """
        try:
            # 각 프로세스에서 독립적인 모델과 MCTS 생성 (프로파일링 없음)
            neural_network = TicTacToeNeuralNetwork().to(device)
            mcts = MCTS(neural_network, num_mcts_sims, temperature=temperature, device=device, batch_size=mcts_batch_size)
            
            states = []
            policies = []
            values = []
            
            # 초기 상태
            state = TicTacToeGameState()
            
            while not state.is_game_over():
                # MCTS 검색 (프로파일링 없음)
                action_probs = mcts.search(state)  # (9,)
                
                # 확률 정규화 (합이 1이 되도록)
                if np.sum(action_probs) > 0:
                    action_probs = action_probs / np.sum(action_probs)
                else:
                    # 모든 확률이 0인 경우 균등 분포로 설정
                    action_probs = np.ones(9) / 9
                
                # 상태와 정책 저장
                states.append(state.get_encoded_state())
                policies.append(action_probs)
                
                # 액션 선택 (온도에 따른 확률적 선택)
                action_idx = np.random.choice(9, p=action_probs)
                action = state.get_action_from_index(action_idx)
                
                # 액션 유효성 검사
                valid_moves = state.get_valid_moves()
                if action in valid_moves:
                    state = state.make_move(action)
                else:
                    # 유효하지 않은 액션인 경우, 유효한 액션 중에서 랜덤 선택
                    if valid_moves:
                        action = random.choice(valid_moves)
                        state = state.make_move(action)
            
            # 게임 결과
            winner = state.get_winner()
            if winner is None:
                game_value = 0.0  # 무승부
            else:
                game_value = 1.0 if winner == 1 else -1.0  # X 플레이어 관점
            
            # 각 상태에 대한 가치 설정
            for i in range(len(states)):
                # 현재 플레이어 관점에서의 가치
                if i % 2 == 0:  # X 플레이어 차례
                    values.append(game_value)
                else:  # O 플레이어 차례
                    values.append(-game_value)
            
            game_data = GameData(states, policies, values)
            return episode_id, game_data
            
        except KeyboardInterrupt:
            print(f"에피소드 {episode_id}가 중단되었습니다.")
            raise

    @profile_section("collect_self_play_data")
    def collect_self_play_data(self) -> List[GameData]:
        """
        자기대전 데이터 수집 (병렬화 버전)
        
        Returns:
            게임 데이터 리스트
        """
        if self.enable_profiling:
            self.profiler.start_timer("collect_self_play_data")
        
        print(f"자기대전 데이터 수집 중... ({self.num_episodes} 게임)")
        
        # CPU 코어 수 확인 (최대 8개로 제한)
        num_workers = min(mp.cpu_count(), 8)
        print(f"병렬 처리: {num_workers}개 프로세스 사용")
        
        # 병렬 실행을 위한 함수 준비
        parallel_func = partial(
            self._parallel_self_play_episode,
            num_mcts_sims=self.num_mcts_sims,
            temperature=self.temperature,
            device=str(self.device),
            mcts_batch_size=self.mcts_batch_size
        )
        
        # 멀티프로세싱으로 게임 실행
        try:
        with mp.Pool(processes=num_workers) as pool:
            self.current_pool = pool
                results = []
                for i, result in enumerate(pool.imap_unordered(parallel_func, range(self.num_episodes)), 1):
                    results.append(result)
                    print(f"진행 상황: {self.num_episodes}게임 중 {i}게임 완료", end='\r', flush=True)
                self.current_pool = None
            print()  # 줄바꿈
        except KeyboardInterrupt:
            print("\n자기대전 데이터 수집이 중단되었습니다.")
            if self.current_pool:
                self.current_pool.terminate()
                self.current_pool.join()
            self.current_pool = None
            raise
        
        # 결과 정렬 및 데이터 버퍼에 추가
        game_data_list = []
        for episode_id, game_data in sorted(results, key=lambda x: x[0]):
            game_data_list.append(game_data)
            
            # 데이터 버퍼에 추가
            for i in range(len(game_data.states)):
                self.data_buffer.append((
                    game_data.states[i],
                    game_data.policies[i],
                    game_data.values[i]
                ))
        
        total_states = sum(len(data.states) for data in game_data_list)
        print(f"데이터 수집 완료. 총 {len(self.data_buffer)} 개의 상태-정책-가치 쌍")
        
        if self.enable_profiling:
            self.profiler.end_timer("collect_self_play_data")
        
        return game_data_list

    @profile_section("train_neural_network")
    def train_neural_network(self, game_data_list: List[GameData]):
        """
        신경망 학습
        
        Args:
            game_data_list: 학습할 게임 데이터 리스트
        """
        if self.enable_profiling:
            self.profiler.start_timer("train_neural_network")
        
        print("신경망 학습 시작...")
        
        # 데이터 준비
        states = []
        policies = []
        values = []
        
        for game_data in game_data_list:
            states.extend(game_data.states)
            policies.extend(game_data.policies)
            values.extend(game_data.values)
        
        # 데이터셋 생성 (TicTacToe는 permute 필요)
        states_tensor = torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2).to(self.device)
        policies_tensor = torch.FloatTensor(np.array(policies)).to(self.device)
        values_tensor = torch.FloatTensor(np.array(values)).unsqueeze(1).to(self.device)
        
        dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 학습
        self.neural_network.train()
        
        for epoch in range(self.num_epochs):
            if self.enable_profiling:
                self.profiler.start_timer(f"epoch_{epoch}")
            
            total_loss = 0.0
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            num_batches = 0
            
            for batch_states, batch_policies, batch_values in dataloader:
                # 순전파
                policy_pred, value_pred = self.neural_network(batch_states)
                
                # 손실 계산
                total_loss_batch, policy_loss, value_loss = self.criterion(
                    policy_pred, value_pred, batch_policies, batch_values
                )
                
                # 역전파
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                
                # 통계 업데이트
                total_loss += total_loss_batch.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                num_batches += 1
            
            if self.enable_profiling:
                self.profiler.end_timer(f"epoch_{epoch}")
            
            # 평균 손실 출력
            avg_total_loss = total_loss / num_batches
            avg_policy_loss = policy_loss_sum / num_batches
            avg_value_loss = value_loss_sum / num_batches
            
            print(f"  Epoch {epoch+1}/{self.num_epochs}: "
                  f"Total Loss: {avg_total_loss:.4f}, "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}")
        
        print("신경망 학습 완료")
        
        if self.enable_profiling:
            self.profiler.end_timer("train_neural_network")

    def _signal_handler(self, signum, frame):
        """시그널 핸들러 - Ctrl+C 등으로 종료 시 모든 프로세스 정리"""
        print(f"\n시그널 {signum}을 받았습니다. 모든 프로세스를 정리하고 종료합니다...")
        
        # 현재 실행 중인 프로세스 풀 종료
        if self.current_pool:
            print("프로세스 풀을 종료합니다...")
            self.current_pool.terminate()
            self.current_pool.join()
            self.current_pool = None
        
        # 성능 프로파일링 요약 출력 (있는 경우)
        if self.enable_profiling and self.profiler:
            print("\n현재까지의 성능 프로파일링 요약:")
            self.profiler.print_summary()
        
        print("프로그램을 종료합니다.")
        sys.exit(0)

    @staticmethod
    def _parallel_evaluate_game(game_id: int, num_mcts_sims: int, temperature: float, device: str, mcts_batch_size: int = 32) -> Tuple[int, int]:
        """
        병렬 실행을 위한 모델 평가 게임 (정적 메서드)
        
        Args:
            game_id: 게임 ID
            num_mcts_sims: MCTS 시뮬레이션 수
            temperature: 온도 파라미터
            device: 사용할 디바이스
            mcts_batch_size: MCTS 배치 크기
            
        Returns:
            (game_id, winner) 튜플 (1: X 플레이어 승리, -1: O 플레이어 승리, 0: 무승부)
        """
        try:
            # 각 프로세스에서 독립적인 모델과 MCTS 생성 (프로파일링 없음)
            neural_network = TicTacToeNeuralNetwork().to(device)
            mcts = MCTS(neural_network, num_mcts_sims, temperature=temperature, device=device, batch_size=mcts_batch_size)
            
        state = TicTacToeGameState()
            
        while not state.is_game_over():
                if state.current_player == 1:  # 현재 모델 (X 플레이어)
                    action_probs = mcts.search(state)
            action_idx = np.argmax(action_probs)
            action = state.get_action_from_index(action_idx)
                    
                    # 액션 실행 (오류 발생 시 랜덤 선택)
                    try:
            valid_moves = state.get_valid_moves()
            if action in valid_moves:
                state = state.make_move(action)
            else:
                            # 유효하지 않은 액션인 경우 랜덤 선택
                            if valid_moves:
                                action = random.choice(valid_moves)
                                state = state.make_move(action)
                    except Exception:
                        # 유효하지 않은 경우 랜덤 선택
                        valid_moves = state.get_valid_moves()
                        if valid_moves:
                            action = random.choice(valid_moves)
                            state = state.make_move(action)
                else:  # 이전 모델 (O 플레이어) - 랜덤 플레이
                    valid_moves = state.get_valid_moves()
                    if valid_moves:
                        action = random.choice(valid_moves)
                        try:
                            state = state.make_move(action)
                        except Exception:
                            # 유효하지 않은 경우 다른 액션 선택
                if valid_moves:
                    action = random.choice(valid_moves)
                    state = state.make_move(action)
                    else:
                        break
            
            # 게임 결과 확인
        winner = state.get_winner()
            if winner is None:
                return game_id, 0  # 무승부
            else:
        return game_id, winner
                
        except KeyboardInterrupt:
            print(f"평가 게임 {game_id}가 중단되었습니다.")
            raise

    @profile_section("evaluate_model")
    def evaluate_model(self, num_games: int = 100, eval_mcts_sims: int = 200, eval_batch_size: int = 64) -> float:
        """
        모델 평가 (이전 모델과의 대전) - 병렬화 버전
        
        Args:
            num_games: 평가 게임 수
            eval_mcts_sims: 평가용 MCTS 시뮬레이션 수 (학습용보다 적게)
            eval_batch_size: 평가용 MCTS 배치 크기
        Returns:
            승률
        """
        if self.enable_profiling:
            self.profiler.start_timer("evaluate_model")
        
        print(f"모델 평가 중... ({num_games} 게임, {eval_mcts_sims} 시뮬레이션)")
        
        # CPU 코어 수 확인 (최대 8개로 제한)
        num_workers = min(mp.cpu_count(), 8)
        print(f"병렬 평가: {num_workers}개 프로세스 사용")
        
        # 병렬 실행을 위한 함수 준비
        parallel_func = partial(
            self._parallel_evaluate_game,
            num_mcts_sims=eval_mcts_sims,  # 평가용으로 줄임
            temperature=self.temperature,
            device=str(self.device),
            mcts_batch_size=eval_batch_size  # 평가용 배치 크기
        )
        
        # 멀티프로세싱으로 게임 실행
        try:
            with mp.Pool(processes=num_workers) as pool:
                self.current_pool = pool
                results = []
                for i, result in enumerate(pool.imap_unordered(parallel_func, range(num_games)), 1):
                    results.append(result)
                    print(f"평가 진행 상황: {num_games}게임 중 {i}게임 완료", end='\r', flush=True)
                self.current_pool = None
            print()  # 줄바꿈
        except KeyboardInterrupt:
            print("\n모델 평가가 중단되었습니다.")
            if self.current_pool:
                self.current_pool.terminate()
                self.current_pool.join()
                self.current_pool = None
            raise
        
        # 결과 집계
        wins = 0
        for game_id, winner in results:
            if winner == 1:  # X 플레이어 승리
                wins += 1
        
        win_rate = wins / num_games
        print(f"평가 완료. 승률: {win_rate:.2%} ({wins}/{num_games})")
        
        if self.enable_profiling:
            self.profiler.end_timer("evaluate_model")
        
        return win_rate

    def save_model(self, iteration: int, save_optimizer: bool = True, save_buffer: bool = True, 
                   save_metadata: bool = True, compress: bool = False):
        """
        모델 저장 (개선된 버전)
        
        Args:
            iteration: 반복 번호
            save_optimizer: 옵티마이저 상태 저장 여부
            save_buffer: 리플레이 버퍼 저장 여부
            save_metadata: 메타데이터 저장 여부
            compress: 압축 저장 여부
        """
        # 기본 모델 파일 경로
        base_path = os.path.join(self.model_dir, f"model_iteration_{iteration}")
        
        # 1. 신경망 모델 저장
        model_path = f"{base_path}.pth"
        self.neural_network.save_model(model_path)
        print(f"신경망 모델 저장: {model_path}")
        
        # 2. 옵티마이저 상태 저장
        if save_optimizer:
            optimizer_path = f"{base_path}_optimizer.pth"
            torch.save(self.optimizer.state_dict(), optimizer_path)
            print(f"옵티마이저 상태 저장: {optimizer_path}")
        
        # 3. 리플레이 버퍼 저장
        if save_buffer and self.data_buffer:
            buffer_path = f"{base_path}_buffer.pkl"
            if compress:
                buffer_path += ".gz"
                with gzip.open(buffer_path, 'wb') as f:
                    pickle.dump(list(self.data_buffer), f)
            else:
                with open(buffer_path, 'wb') as f:
                    pickle.dump(list(self.data_buffer), f)
            print(f"리플레이 버퍼 저장: {buffer_path}")
        
        # 4. 메타데이터 저장
        if save_metadata:
            metadata = {
                'iteration': iteration,
                'best_win_rate': self.best_win_rate,
                'num_iterations': self.num_iterations,
                'num_episodes': self.num_episodes,
                'num_mcts_sims': self.num_mcts_sims,
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'c_puct': self.c_puct,
                'temperature': self.temperature,
                'mcts_batch_size': self.mcts_batch_size,
                'save_timestamp': datetime.now().isoformat(),
                'model_config': {
                    'device': str(self.device),
                    'enable_profiling': self.enable_profiling
                }
            }
            
            metadata_path = f"{base_path}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"메타데이터 저장: {metadata_path}")
        
        # 5. 체크포인트 요약 파일 생성
        checkpoint_info = {
            'iteration': iteration,
            'best_win_rate': self.best_win_rate,
            'timestamp': datetime.now().isoformat(),
            'files': {
                'model': f"model_iteration_{iteration}.pth",
                'optimizer': f"model_iteration_{iteration}_optimizer.pth" if save_optimizer else None,
                'buffer': f"model_iteration_{iteration}_buffer.pkl" + (".gz" if compress else "") if save_buffer else None,
                'metadata': f"model_iteration_{iteration}_metadata.json" if save_metadata else None
            }
        }
        
        checkpoint_path = f"{base_path}_checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_info, f, indent=2, ensure_ascii=False)
        
        print(f"체크포인트 정보 저장: {checkpoint_path}")
        print(f"모델 저장 완료: iteration {iteration}")
        
        # 이전 파일들 정리 (최신 2개만 유지)
        self._cleanup_old_files(iteration, keep_latest=2)
    
    def load_model(self, iteration: int, load_optimizer: bool = True, load_buffer: bool = True, 
                   load_metadata: bool = True, resume_training: bool = True):
        """
        모델 로드 (개선된 버전)
        
        Args:
            iteration: 로드할 반복 번호
            load_optimizer: 옵티마이저 상태 로드 여부
            load_buffer: 리플레이 버퍼 로드 여부
            load_metadata: 메타데이터 로드 여부
            resume_training: 학습 재개 여부 (True면 다음 반복부터 시작)
        """
        base_path = os.path.join(self.model_dir, f"model_iteration_{iteration}")
        
        # 1. 신경망 모델 로드
        model_path = f"{base_path}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {model_path}")
        
        self.neural_network = TicTacToeNeuralNetwork.load_model(model_path, self.device)
        self.mcts.neural_network = self.neural_network
        print(f"신경망 모델 로드: {model_path}")
        
        # 2. 옵티마이저 상태 로드
        if load_optimizer:
            optimizer_path = f"{base_path}_optimizer.pth"
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
                print(f"옵티마이저 상태 로드: {optimizer_path}")
            else:
                print(f"옵티마이저 파일을 찾을 수 없음: {optimizer_path}")
        
        # 3. 리플레이 버퍼 로드
        if load_buffer:
            buffer_path = f"{base_path}_buffer.pkl"
            buffer_gz_path = f"{base_path}_buffer.pkl.gz"
            
            if os.path.exists(buffer_gz_path):
                with gzip.open(buffer_gz_path, 'rb') as f:
                    buffer_data = pickle.load(f)
                print(f"압축된 리플레이 버퍼 로드: {buffer_gz_path}")
            elif os.path.exists(buffer_path):
                with open(buffer_path, 'rb') as f:
                    buffer_data = pickle.load(f)
                print(f"리플레이 버퍼 로드: {buffer_path}")
            else:
                print(f"리플레이 버퍼 파일을 찾을 수 없음: {buffer_path}")
                buffer_data = []
            
            if buffer_data:
                self.data_buffer = deque(buffer_data, maxlen=10000)
                print(f"리플레이 버퍼 크기: {len(self.data_buffer)}")
        
        # 4. 메타데이터 로드
        if load_metadata:
            metadata_path = f"{base_path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 메타데이터에서 값들 복원
                self.best_win_rate = metadata.get('best_win_rate', 0.0)
                print(f"최고 승률 복원: {self.best_win_rate:.2%}")
                
                # 설정값들 출력
                print(f"로드된 설정:")
                print(f"  - 반복 횟수: {metadata.get('num_iterations')}")
                print(f"  - 각 반복당 게임 수: {metadata.get('num_episodes')}")
                print(f"  - MCTS 시뮬레이션 수: {metadata.get('num_mcts_sims')}")
                print(f"  - 학습 에포크 수: {metadata.get('num_epochs')}")
                print(f"  - 배치 크기: {metadata.get('batch_size')}")
                print(f"  - 학습률: {metadata.get('learning_rate')}")
                print(f"  - 저장 시간: {metadata.get('save_timestamp')}")
                
                print(f"메타데이터 로드: {metadata_path}")
            else:
                print(f"메타데이터 파일을 찾을 수 없음: {metadata_path}")
        
        # 5. 학습 재개 설정
        if resume_training:
            self.iteration = iteration + 1
            print(f"학습 재개: iteration {self.iteration}부터 시작")
        else:
            self.iteration = iteration
            print(f"모델 로드 완료: iteration {iteration}")
        
        print(f"모델 로드 완료: {model_path}")
    
    def save_checkpoint(self, iteration: int, description: str = ""):
        """
        체크포인트 저장 (간단한 버전)
        
        Args:
            iteration: 반복 번호
            description: 체크포인트 설명
        """
        checkpoint_data = {
            'iteration': iteration,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'best_win_rate': self.best_win_rate,
            'model_state_dict': self.neural_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'data_buffer': list(self.data_buffer) if self.data_buffer else [],
            'config': {
                'num_iterations': self.num_iterations,
                'num_episodes': self.num_episodes,
                'num_mcts_sims': self.num_mcts_sims,
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'c_puct': self.c_puct,
                'temperature': self.temperature,
                'mcts_batch_size': self.mcts_batch_size,
                'device': str(self.device),
                'enable_profiling': self.enable_profiling
            }
        }
        
        checkpoint_path = os.path.join(self.model_dir, f"checkpoint_iteration_{iteration}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"체크포인트 저장: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, iteration: int):
        """
        체크포인트 로드
        
        Args:
            iteration: 로드할 체크포인트 반복 번호
        """
        checkpoint_path = os.path.join(self.model_dir, f"checkpoint_iteration_{iteration}.pkl")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없음: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # 모델 상태 복원
        self.neural_network.load_state_dict(checkpoint_data['model_state_dict'])
        self.mcts.neural_network = self.neural_network
        
        # 옵티마이저 상태 복원
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # 데이터 버퍼 복원
        if checkpoint_data['data_buffer']:
            self.data_buffer = deque(checkpoint_data['data_buffer'], maxlen=10000)
        
        # 메타데이터 복원
        self.iteration = checkpoint_data['iteration'] + 1
        self.best_win_rate = checkpoint_data['best_win_rate']
        
        print(f"체크포인트 로드: {checkpoint_path}")
        print(f"반복 번호: {checkpoint_data['iteration']}")
        print(f"최고 승률: {self.best_win_rate:.2%}")
        print(f"설명: {checkpoint_data.get('description', '')}")
        print(f"저장 시간: {checkpoint_data['timestamp']}")
        print(f"학습 재개: iteration {self.iteration}부터 시작")
        
        return checkpoint_data
    
    def _cleanup_old_files(self, current_iteration: int, keep_latest: int = 2):
        """
        이전 파일들을 정리하여 최신 몇 개만 유지
        
        Args:
            current_iteration: 현재 반복 번호
            keep_latest: 유지할 최신 파일 개수
        """
        try:
            # 정리할 파일 패턴들 (best_model 파일들은 제외)
            file_patterns = [
                "model_iteration_*.pth",
                "model_iteration_*_optimizer.pth", 
                "model_iteration_*_buffer.pkl*",
                "model_iteration_*_metadata.json",
                "model_iteration_*_checkpoint.json",
                "checkpoint_iteration_*.pkl"
            ]
            
            # 각 패턴별로 파일 정리
            for pattern in file_patterns:
                import glob
                files = glob.glob(os.path.join(self.model_dir, pattern))
                
                # 파일명에서 반복 번호 추출
                file_iterations = []
                for file in files:
                    try:
                        # 파일명에서 숫자 추출
                        import re
                        match = re.search(r'iteration_(\d+)', file)
                        if match:
                            iter_num = int(match.group(1))
                            file_iterations.append((iter_num, file))
                    except:
                        continue
                
                # 반복 번호로 정렬
                file_iterations.sort(key=lambda x: x[0])
                
                # 최신 keep_latest개를 제외하고 삭제
                files_to_delete = file_iterations[:-keep_latest] if len(file_iterations) > keep_latest else []
                
                for iter_num, file in files_to_delete:
                    try:
                        os.remove(file)
                        print(f"  삭제된 파일: {os.path.basename(file)}")
                    except:
                        continue
            
            print(f"파일 정리 완료: 최신 {keep_latest}개 반복 파일만 유지")
            
        except Exception as e:
            print(f"파일 정리 중 오류 발생: {e}")

    @profile_section("train")
    def train(self):
        """전체 학습 과정"""
        if self.enable_profiling:
            self.profiler.start_timer("train")
        
        print("알파제로 학습 시작!")
        print(f"디바이스: {self.device}")
        print(f"반복 횟수: {self.num_iterations}")
        print(f"각 반복당 게임 수: {self.num_episodes}")
        print(f"MCTS 시뮬레이션 수: {self.num_mcts_sims}")
        
        start_time = time.time()
        
        for iteration in range(self.iteration, self.num_iterations):
            if self.enable_profiling:
                self.profiler.start_timer(f"iteration_{iteration}")
            
            print(f"\n=== 반복 {iteration + 1}/{self.num_iterations} ===")
            
            # 1. 자기대전 데이터 수집
            game_data_list = self.collect_self_play_data()
            
            # 2. 신경망 학습
            self.train_neural_network(game_data_list)
            
            # 3. 모델 평가 (5번째 반복 이후부터 5번마다)
            if iteration > 0 and (iteration + 1) % 5 == 0:
                win_rate = self.evaluate_model(eval_mcts_sims=200, eval_batch_size=64)
                
                # 최고 성능 모델 저장 (별도 파일로)
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    # 최고 성능 모델을 best_model.pth로 별도 저장
                    best_model_path = os.path.join(self.model_dir, "best_model.pth")
                    self.neural_network.save_model(best_model_path)
                    print(f"새로운 최고 성능 모델 저장: {best_model_path} (승률: {win_rate:.2%})")
                    
                    # 최고 성능 메타데이터 저장
                    best_metadata = {
                        'iteration': iteration,
                        'best_win_rate': self.best_win_rate,
                        'save_timestamp': datetime.now().isoformat(),
                        'model_config': {
                            'device': str(self.device),
                            'enable_profiling': self.enable_profiling
                        }
                    }
                    best_metadata_path = os.path.join(self.model_dir, "best_model_metadata.json")
                    with open(best_metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(best_metadata, f, indent=2, ensure_ascii=False)
                    print(f"최고 성능 메타데이터 저장: {best_metadata_path}")
            
            # 4. 최신 모델 저장 (매 반복마다) - 기본 저장
            self.save_model(iteration, save_optimizer=True, save_buffer=True, save_metadata=True)
            
            # 5. 최신 체크포인트 저장 (매 반복마다)
            self.save_checkpoint(iteration, f"최신 체크포인트 - 반복 {iteration + 1}")
            
            # 6. MCTS 캐시 초기화
            self.mcts.clear_cache()
            
            if self.enable_profiling:
                self.profiler.end_timer(f"iteration_{iteration}")
            
            # 진행 상황 출력
            elapsed_time = time.time() - start_time
            print(f"총 소요 시간: {elapsed_time:.1f}초")
            print(f"예상 남은 시간: {elapsed_time / (iteration + 1 - self.iteration) * (self.num_iterations - iteration - 1):.1f}초")
        
        # 성능 프로파일링 요약 출력
        if self.enable_profiling:
            self.profiler.print_summary()
            report_file = os.path.join(self.model_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            self.profiler.save_report(report_file)
        
        print("\n알파제로 학습 완료!")
        print(f"최고 승률: {self.best_win_rate:.2%}")
        print(f"총 소요 시간: {time.time() - start_time:.1f}초")
        
        if self.enable_profiling:
            self.profiler.end_timer("train")

    def play_against_human(self):
        state = TicTacToeGameState()
        print("[AlphaZero] Play against human! You are O (second player). Board:")
        print(state)
        while not state.is_game_over():
            if state.current_player == 1:
                action_probs = self.mcts.search(state)
                action_idx = np.argmax(action_probs)
                action = state.get_action_from_index(action_idx)
                print(f"AlphaZero plays: {action}")
                state = state.make_move(action)
            else:
                valid_moves = state.get_valid_moves()
                print("Your turn. Valid moves:", valid_moves)
                move = None
                while move not in valid_moves:
                    try:
                        move_str = input("Enter your move as row,col (e.g. 0,2): ")
                        row, col = map(int, move_str.strip().split(","))
                        move = (row, col)
                    except Exception:
                        print("Invalid input. Try again.")
                state = state.make_move(move)
            print(state)
        winner = state.get_winner()
        if winner == 1:
            print("AlphaZero wins!")
        elif winner == -1:
            print("You win!")
        else:
            print("Draw!") 