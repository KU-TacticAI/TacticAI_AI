"""
Omok(오목)용 AlphaZero 학습 클래스

Chess / Othello / TicTacToe 스타일을 따르며 병렬 자기대전, 프로파일러,
모델/옵티마이저/버퍼 저장·로드, 평가 및 학습 루프를 제공합니다.
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

from .game_state import OmokGameState
from .neural_network import OmokNeuralNetwork, PolicyValueLoss
from .mcts import MCTS
from .performance_profiler import PerformanceProfiler, profile_function, profile_section, get_profiler


class GameData:
    def __init__(self, states: List[np.ndarray], policies: List[np.ndarray], values: List[float]):
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)


class AlphaZeroOmok:
    def __init__(self,
                 model_dir: str = "models",
                 board_size: int = 15,
                 num_iterations: int = 100,
                 num_episodes: int = 100,
                 num_mcts_sims: int = 200,
                 num_epochs: int = 5,
                 batch_size: int = 256,
                 learning_rate: float = 0.001,
                 c_puct: float = 1.0,
                 temperature: float = 1.0,
                 device: Optional[torch.device] = None,
                 mcts_batch_size: int = 32,
                 enable_profiling: bool = True):

        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)

        self.board_size = board_size
        self.num_actions = board_size * board_size

        self.num_iterations = num_iterations
        self.num_episodes = num_episodes
        self.num_mcts_sims = num_mcts_sims
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.c_puct = c_puct
        self.temperature = temperature
        self.mcts_batch_size = mcts_batch_size

        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # 모델, MCTS, 옵티마이저
        self.neural_network = OmokNeuralNetwork(board_size=board_size).to(self.device)
        self.mcts = MCTS(self.neural_network, num_simulations=self.num_mcts_sims, c_puct=self.c_puct, temperature=self.temperature, device=self.device, batch_size=self.mcts_batch_size)
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=self.learning_rate)
        self.criterion = PolicyValueLoss()

        # 리플레이 버퍼
        self.data_buffer = deque(maxlen=20000)

        self.iteration = 0
        self.best_win_rate = 0.0

        # 프로파일러
        self.enable_profiling = enable_profiling
        if enable_profiling:
            log_file = os.path.join(self.model_dir, f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            self.profiler = PerformanceProfiler(log_file=log_file, enable_console=True)
            # set global profiler
            from .performance_profiler import _profiler as _prof
            try:
                # assign internal module global
                import importlib
                mod = importlib.import_module('model.Omok.performance_profiler')
                mod._profiler = self.profiler
            except Exception:
                pass
        else:
            self.profiler = None

        # 멀티프로세싱 풀
        self.current_pool = None

        # 시그널
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    @staticmethod
    def _parallel_self_play_episode(episode_id: int, num_mcts_sims: int, temperature: float, device: str, board_size: int, mcts_batch_size: int = 32) -> Tuple[int, GameData]:
        """
        병렬 자기대전 워커(프로세스별 독립 모델 사용)
        """
        try:
            # 각 프로세스에서 독립적인 모델과 MCTS 생성
            neural_network = OmokNeuralNetwork(board_size=board_size).to(device)
            mcts = MCTS(neural_network, num_simulations=num_mcts_sims, temperature=temperature, device=device, batch_size=mcts_batch_size)

            states = []
            policies = []
            values = []

            state = OmokGameState(board_size=board_size)
            while not state.is_game_over():
                action_probs = mcts.search(state)
                if np.sum(action_probs) > 0:
                    action_probs = action_probs / np.sum(action_probs)
                else:
                    action_probs = np.ones(board_size * board_size) / (board_size * board_size)

                states.append(state.get_encoded_state())
                policies.append(action_probs)

                action_idx = np.random.choice(board_size * board_size, p=action_probs)
                action = state.get_action_from_index(action_idx)
                valid = state.get_valid_moves()
                if action in valid:
                    state = state.make_move(action)
                else:
                    if valid:
                        state = state.make_move(random.choice(valid))

            winner = state.get_winner()
            if winner is None:
                game_value = 0.0
            else:
                game_value = 1.0 if winner == 1 else -1.0

            for i in range(len(states)):
                values.append(game_value if i % 2 == 0 else -game_value)

            return episode_id, GameData(states, policies, values)

        except KeyboardInterrupt:
            print(f"에피소드 {episode_id}가 중단되었습니다.")
            raise

    @profile_section("collect_self_play_data")
    def collect_self_play_data(self) -> List[GameData]:
        if self.enable_profiling:
            self.profiler.start_timer("collect_self_play_data")

        print(f"자기대전 데이터 수집 중... ({self.num_episodes} 게임)")

        num_workers = min(mp.cpu_count(), 8)
        print(f"병렬 처리: {num_workers}개 프로세스 사용")

        parallel_func = partial(
            self._parallel_self_play_episode,
            num_mcts_sims=self.num_mcts_sims,
            temperature=self.temperature,
            device=str(self.device),
            board_size=self.board_size,
            mcts_batch_size=self.mcts_batch_size
        )

        try:
            with mp.Pool(processes=num_workers) as pool:
                self.current_pool = pool
                results = []
                for i, result in enumerate(pool.imap_unordered(parallel_func, range(self.num_episodes)), 1):
                    results.append(result)
                    print(f"진행 상황: {self.num_episodes}게임 중 {i}게임 완료", end='\r', flush=True)
                self.current_pool = None
            print()
        except KeyboardInterrupt:
            print("\n자기대전 데이터 수집이 중단되었습니다.")
            if self.current_pool:
                self.current_pool.terminate()
                self.current_pool.join()
                self.current_pool = None
            raise

        game_data_list = []
        for episode_id, game_data in sorted(results, key=lambda x: x[0]):
            game_data_list.append(game_data)
            for i in range(len(game_data.states)):
                self.data_buffer.append((game_data.states[i], game_data.policies[i], game_data.values[i]))

        print(f"데이터 수집 완료. 리플레이 버퍼 크기: {len(self.data_buffer)}")

        if self.enable_profiling:
            self.profiler.end_timer("collect_self_play_data")

        return game_data_list

    @profile_section("train_neural_network")
    def train_neural_network(self, game_data_list: Optional[List[GameData]] = None):
        if self.enable_profiling:
            self.profiler.start_timer("train_neural_network")

        # 데이터 준비
        if game_data_list is None:
            samples = list(self.data_buffer)
            if not samples:
                return
            states = np.array([s for s, p, v in samples])
            policies = np.array([p for s, p, v in samples])
            values = np.array([v for s, p, v in samples]).reshape(-1, 1)
        else:
            states = np.array([s for gd in game_data_list for s in gd.states])
            policies = np.array([p for gd in game_data_list for p in gd.policies])
            values = np.array([v for gd in game_data_list for v in gd.values]).reshape(-1, 1)

        # states are (N, C, H, W) already
        dataset = TensorDataset(torch.FloatTensor(states), torch.FloatTensor(policies), torch.FloatTensor(values))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.neural_network.train()
        for epoch in range(self.num_epochs):
            if self.enable_profiling:
                self.profiler.start_timer(f"epoch_{epoch}")
            total_loss = 0.0
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            num_batches = 0
            for batch_states, batch_policies, batch_values in dataloader:
                batch_states = batch_states.to(self.device)
                batch_policies = batch_policies.to(self.device)
                batch_values = batch_values.to(self.device)

                policy_pred, value_pred = self.neural_network(batch_states)
                total_loss_batch, policy_loss, value_loss = self.criterion(policy_pred, value_pred, batch_policies, batch_values)

                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()

                total_loss += total_loss_batch.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                num_batches += 1

            if self.enable_profiling:
                self.profiler.end_timer(f"epoch_{epoch}")

            avg_total_loss = total_loss / num_batches
            avg_policy_loss = policy_loss_sum / num_batches
            avg_value_loss = value_loss_sum / num_batches
            print(f"  Epoch {epoch+1}/{self.num_epochs}: Total Loss: {avg_total_loss:.4f}, Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")

        if self.enable_profiling:
            self.profiler.end_timer("train_neural_network")

    def _signal_handler(self, signum, frame):
        print(f"\n시그널 {signum}을 받았습니다. 모든 프로세스를 정리하고 종료합니다...")
        if self.current_pool:
            print("프로세스 풀을 종료합니다...")
            try:
                self.current_pool.terminate()
                self.current_pool.join()
            except Exception:
                pass
            self.current_pool = None
        if self.enable_profiling and self.profiler:
            print("\n성능 프로파일링 요약:")
            self.profiler.print_summary()
        print("프로그램을 종료합니다.")
        sys.exit(0)

    @profile_section("evaluate_model")
    def evaluate_model(self, num_games: int = 50, eval_mcts_sims: int = 200, eval_batch_size: int = 32) -> float:
        if self.enable_profiling:
            self.profiler.start_timer("evaluate_model")

        print(f"모델 평가 중... ({num_games} 게임)")
        num_workers = min(mp.cpu_count(), 8)
        parallel_func = partial(self._parallel_evaluate_game, num_mcts_sims=eval_mcts_sims, temperature=self.temperature, device=str(self.device), board_size=self.board_size, mcts_batch_size=eval_batch_size)

        try:
            with mp.Pool(processes=num_workers) as pool:
                self.current_pool = pool
                results = []
                for i, result in enumerate(pool.imap_unordered(parallel_func, range(num_games)), 1):
                    results.append(result)
                    print(f"평가 진행 상황: {num_games}게임 중 {i}게임 완료", end='\r', flush=True)
                self.current_pool = None
            print()
        except KeyboardInterrupt:
            print("\n모델 평가가 중단되었습니다.")
            if self.current_pool:
                self.current_pool.terminate()
                self.current_pool.join()
                self.current_pool = None
            raise

        wins = sum(1 for gid, winner in results if winner == 1)
        win_rate = wins / num_games
        print(f"평가 완료. 승률: {win_rate:.2%} ({wins}/{num_games})")

        if self.enable_profiling:
            self.profiler.end_timer("evaluate_model")
        return win_rate

    @staticmethod
    def _parallel_evaluate_game(game_id: int, num_mcts_sims: int, temperature: float, device: str, board_size: int, mcts_batch_size: int = 32) -> Tuple[int, int]:
        try:
            neural_network = OmokNeuralNetwork(board_size=board_size).to(device)
            mcts = MCTS(neural_network, num_simulations=num_mcts_sims, temperature=temperature, device=device, batch_size=mcts_batch_size)
            state = OmokGameState(board_size=board_size)
            while not state.is_game_over():
                if state.current_player == 1:
                    action_probs = mcts.search(state)
                    action_idx = int(np.argmax(action_probs))
                    action = state.get_action_from_index(action_idx)
                    valid_moves = state.get_valid_moves()
                    if action in valid_moves:
                        state = state.make_move(action)
                    else:
                        if valid_moves:
                            state = state.make_move(random.choice(valid_moves))
                else:
                    valid_moves = state.get_valid_moves()
                    if valid_moves:
                        state = state.make_move(random.choice(valid_moves))
                    else:
                        break
            winner = state.get_winner()
            if winner is None:
                return game_id, 0
            return game_id, winner
        except KeyboardInterrupt:
            print(f"평가 게임 {game_id}가 중단되었습니다.")
            raise

    def save_model(self, iteration: int, save_optimizer: bool = True, save_buffer: bool = True, save_metadata: bool = True, compress: bool = False):
        base_path = os.path.join(self.model_dir, f"omok_model_iteration_{iteration}")
        model_path = f"{base_path}.pth"
        self.neural_network.save_model(model_path)
        print(f"신경망 저장: {model_path}")

        if save_optimizer:
            opt_path = f"{base_path}_optimizer.pth"
            torch.save(self.optimizer.state_dict(), opt_path)
            print(f"옵티마이저 저장: {opt_path}")

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
                'model_config': {'device': str(self.device), 'enable_profiling': self.enable_profiling}
            }
            meta_path = f"{base_path}_metadata.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"메타데이터 저장: {meta_path}")

        checkpoint_info = {
            'iteration': iteration,
            'best_win_rate': self.best_win_rate,
            'timestamp': datetime.now().isoformat(),
            'files': {'model': f"omok_model_iteration_{iteration}.pth", 'optimizer': f"omok_model_iteration_{iteration}_optimizer.pth" if save_optimizer else None, 'buffer': f"omok_model_iteration_{iteration}_buffer.pkl" + (".gz" if compress else "") if save_buffer else None, 'metadata': f"omok_model_iteration_{iteration}_metadata.json" if save_metadata else None}
        }
        cp_path = f"{base_path}_checkpoint.json"
        with open(cp_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_info, f, indent=2, ensure_ascii=False)
        print(f"체크포인트 저장: {cp_path}")

        self._cleanup_old_files(iteration, keep_latest=2)

    def load_model(self, iteration: int, load_optimizer: bool = True, load_buffer: bool = True, load_metadata: bool = True, resume_training: bool = True):
        base_path = os.path.join(self.model_dir, f"omok_model_iteration_{iteration}")
        model_path = f"{base_path}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.neural_network = OmokNeuralNetwork.load_model(model_path, device=self.device)
        self.mcts.neural_network = self.neural_network
        print(f"신경망 로드: {model_path}")

        if load_optimizer:
            opt_path = f"{base_path}_optimizer.pth"
            if os.path.exists(opt_path):
                self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
                print(f"옵티마이저 로드: {opt_path}")
            else:
                print(f"옵티마이저 파일 없음: {opt_path}")

        if load_buffer:
            buffer_path = f"{base_path}_buffer.pkl"
            buffer_gz = buffer_path + ".gz"
            buffer_data = []
            if os.path.exists(buffer_gz):
                with gzip.open(buffer_gz, 'rb') as f:
                    buffer_data = pickle.load(f)
                print(f"압축된 버퍼 로드: {buffer_gz}")
            elif os.path.exists(buffer_path):
                with open(buffer_path, 'rb') as f:
                    buffer_data = pickle.load(f)
                print(f"버퍼 로드: {buffer_path}")
            else:
                print(f"버퍼 파일 없음: {buffer_path}")
            if buffer_data:
                self.data_buffer = deque(buffer_data, maxlen=20000)

        if load_metadata:
            meta_path = f"{base_path}_metadata.json"
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.best_win_rate = metadata.get('best_win_rate', 0.0)
                print(f"메타데이터 로드: {meta_path}")
            else:
                print(f"메타데이터 없음: {meta_path}")

        if resume_training:
            self.iteration = iteration + 1
        else:
            self.iteration = iteration

        print(f"모델 로드 완료: iteration {self.iteration}")

    def save_checkpoint(self, iteration: int, description: str = ""):
        cp = {'iteration': iteration, 'description': description, 'timestamp': datetime.now().isoformat(), 'best_win_rate': self.best_win_rate}
        path = os.path.join(self.model_dir, f"checkpoint_iteration_{iteration}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(cp, f)
        print(f"체크포인트 저장: {path}")
        return path

    def load_checkpoint(self, iteration: int):
        path = os.path.join(self.model_dir, f"checkpoint_iteration_{iteration}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, 'rb') as f:
            cp = pickle.load(f)
        self.iteration = cp.get('iteration', 0) + 1
        self.best_win_rate = cp.get('best_win_rate', 0.0)
        print(f"체크포인트 로드: {path}")
        return cp

    def _cleanup_old_files(self, current_iteration: int, keep_latest: int = 2):
        try:
            import glob, re
            patterns = ["omok_model_iteration_*.pth", "omok_model_iteration_*_optimizer.pth", "omok_model_iteration_*_buffer.pkl*", "omok_model_iteration_*_metadata.json", "omok_model_iteration_*_checkpoint.json", "checkpoint_iteration_*.pkl"]
            for pattern in patterns:
                files = glob.glob(os.path.join(self.model_dir, pattern))
                iters = []
                for f in files:
                    m = re.search(r'iteration_(\d+)', f)
                    if m:
                        iters.append((int(m.group(1)), f))
                iters.sort(key=lambda x: x[0])
                to_delete = iters[:-keep_latest] if len(iters) > keep_latest else []
                for _, f in to_delete:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            print(f"파일 정리 완료: 최신 {keep_latest}개 유지")
        except Exception as e:
            print(f"파일 정리 오류: {e}")

    @profile_section("train")
    def train(self):
        if self.enable_profiling:
            self.profiler.start_timer("train")

        print("알파제로 Omok 학습 시작")
        print(f"디바이스: {self.device}")
        print(f"반복 횟수: {self.num_iterations}")
        print(f"각 반복당 게임 수: {self.num_episodes}")
        print(f"MCTS 시뮬레이션 수: {self.num_mcts_sims}")

        start_time = time.time()
        for iteration in range(self.iteration, self.num_iterations):
            if self.enable_profiling:
                self.profiler.start_timer(f"iteration_{iteration}")
            print(f"\n=== 반복 {iteration + 1}/{self.num_iterations} ===")

            game_data_list = self.collect_self_play_data()
            self.train_neural_network(game_data_list)

            # 주기적 평가(예시: 5회마다)
            if iteration > 0 and (iteration + 1) % 5 == 0:
                win_rate = self.evaluate_model(num_games=50, eval_mcts_sims=max(50, self.num_mcts_sims // 2), eval_batch_size=32)
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    best_path = os.path.join(self.model_dir, "best_model.pth")
                    self.neural_network.save_model(best_path)
                    print(f"새 최고 모델 저장: {best_path} (승률: {win_rate:.2%})")

            # 모델 저장 및 체크포인트
            self.save_model(iteration, save_optimizer=True, save_buffer=True, save_metadata=True)
            self.save_checkpoint(iteration, f"Iteration {iteration}")

            # 캐시 초기화
            self.mcts.clear_cache()

            if self.enable_profiling:
                self.profiler.end_timer(f"iteration_{iteration}")

            elapsed = time.time() - start_time
            print(f"총 소요 시간: {elapsed:.1f}초")
            remaining = elapsed / (iteration - self.iteration + 1) * (self.num_iterations - iteration - 1) if iteration - self.iteration + 1 > 0 else 0
            print(f"예상 남은 시간: {remaining:.1f}초")

        if self.enable_profiling:
            self.profiler.print_summary()
            report_file = os.path.join(self.model_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            self.profiler.save_report(report_file)
            self.profiler.end_timer("train")

        print("학습 완료")

    def play_against_human(self):
        state = OmokGameState(board_size=self.board_size)
        print(state)
        while not state.is_game_over():
            if state.current_player == 1:
                action = self.mcts.get_best_action(state)
                print(f"AI plays: {action}")
                state = state.make_move(action)
            else:
                valid = state.get_valid_moves()
                print("Your turn. Valid moves count:", len(valid))
                move = None
                while move not in valid:
                    try:
                        s = input("Enter move as r,c: ")
                        r, c = map(int, s.split(','))
                        move = (r, c)
                    except Exception:
                        print("Invalid input")
                state = state.make_move(move)
            print(state)

        winner = state.get_winner()
        if winner == 1:
            print("AI (Black) wins")
        elif winner == -1:
            print("Human (White) wins")
        else:
            print("Draw")
