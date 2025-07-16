"""
AlphaZero용 틱택토 게임 상태 클래스

틱택토 규칙, 상태, 유효한 수, 게임 종료, 상태 인코딩 등을 관리합니다.
"""

import numpy as np
from typing import List, Tuple, Optional, Any
import copy

class TicTacToeGameState:
    """AlphaZero용 틱택토 게임 상태 클래스"""
    def __init__(self, board: Optional[np.ndarray] = None, current_player: int = 1):
        """
        Args:
            board: 3x3 게임 보드 (None이면 초기 상태)
            current_player: 현재 플레이어 (1: X, -1: O)
        """
        if board is None:
            self.board = np.zeros((3, 3), dtype=np.int8)
        else:
            self.board = board.copy()
        self.current_player = current_player

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """현재 플레이어가 둘 수 있는 유효한 위치 목록 반환"""
        valid_moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 0:
                    valid_moves.append((row, col))
        return valid_moves

    def make_move(self, move: Tuple[int, int]) -> 'TicTacToeGameState':
        """수 실행 후 새로운 게임 상태 반환"""
        if move not in self.get_valid_moves():
            raise ValueError(f"Invalid move: {move}")
        new_state = TicTacToeGameState(self.board.copy(), self.current_player)
        row, col = move
        new_state.board[row, col] = self.current_player
        new_state.current_player = -self.current_player
        return new_state

    def is_game_over(self) -> bool:
        """게임이 종료되었는지 확인"""
        return self.get_winner() is not None or np.all(self.board != 0)

    def get_winner(self) -> Optional[int]:
        """
        승자 반환 (1: X승, -1: O승, 0: 무승부, None: 진행중)
        """
        # 가로, 세로, 대각선 체크
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3:
                return int(np.sign(np.sum(self.board[i, :])))
            if abs(np.sum(self.board[:, i])) == 3:
                return int(np.sign(np.sum(self.board[:, i])))
        diag1 = np.sum([self.board[i, i] for i in range(3)])
        diag2 = np.sum([self.board[i, 2 - i] for i in range(3)])
        if abs(diag1) == 3:
            return int(np.sign(diag1))
        if abs(diag2) == 3:
            return int(np.sign(diag2))
        # 무승부
        if np.all(self.board != 0):
            return 0
        # 진행중
        return None

    def get_encoded_state(self) -> np.ndarray:
        """
        신경망 입력용 인코딩된 상태 반환
        (3, 3, 3) 형태의 배열
        - 채널 0: 현재 플레이어의 돌 (1 또는 0)
        - 채널 1: 상대방 돌 (1 또는 0)
        - 채널 2: 현재 플레이어 표시 (1 또는 0)
        """
        encoded = np.zeros((3, 3, 3), dtype=np.float32)
        encoded[0] = (self.board == self.current_player).astype(np.float32)
        encoded[1] = (self.board == -self.current_player).astype(np.float32)
        encoded[2] = float(self.current_player == 1)
        return encoded

    def get_flattened_state(self) -> np.ndarray:
        """
        1차원 9길이 배열로 인코딩된 상태 반환
        - X: 1.0, O: -1.0, 빈칸: 0.0
        """
        return self.board.flatten().astype(np.float32)

    def get_action_index(self, move: Tuple[int, int]) -> int:
        """
        (row, col) -> 0~8 인덱스 변환
        """
        row, col = move
        return row * 3 + col

    def get_action_from_index(self, action_index: int) -> Tuple[int, int]:
        """
        0~8 인덱스 -> (row, col) 변환
        """
        return (action_index // 3, action_index % 3)

    def copy(self) -> 'TicTacToeGameState':
        return TicTacToeGameState(self.board.copy(), self.current_player)

    def __str__(self) -> str:
        symbols = {1: 'X', -1: 'O', 0: '.'}
        rows = [' '.join(symbols[self.board[r, c]] for c in range(3)) for r in range(3)]
        return '\n'.join(rows) 