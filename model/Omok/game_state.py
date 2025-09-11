"""
알파제로용 오목(Omok / Gomoku) 게임 상태 클래스

규칙: 기본적으로 5목(연속 5개) 승리, 보드 크기 기본 15x15.
게임 상태, 유효한 수, 게임 종료 검사, 상태 인코딩 등을 제공합니다.
"""

import numpy as np
from typing import List, Tuple, Optional


class OmokGameState:
    """Omok 게임 상태 클래스"""

    def __init__(self, board: Optional[np.ndarray] = None, current_player: int = 1, board_size: int = 15, win_length: int = 5):
        self.board_size = board_size
        self.win_length = win_length
        if board is None:
            self.board = np.zeros((board_size, board_size), dtype=np.int8)
        else:
            self.board = board.copy()
        self.current_player = current_player

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        valid = []
        bs = self.board_size
        b = self.board
        for r in range(bs):
            for c in range(bs):
                if b[r, c] == 0:
                    valid.append((r, c))
        return valid

    def make_move(self, move: Tuple[int, int]) -> 'OmokGameState':
        if move not in self.get_valid_moves():
            raise ValueError(f"Invalid move: {move}")
        new_state = OmokGameState(self.board.copy(), self.current_player, self.board_size, self.win_length)
        r, c = move
        new_state.board[r, c] = self.current_player
        new_state.current_player = -self.current_player
        return new_state

    def is_game_over(self) -> bool:
        return self.get_winner() is not None or np.all(self.board != 0)

    def get_winner(self) -> Optional[int]:
        # 검사: 가로, 세로, 양대각선에서 연속 win_length 검사
        bs = self.board_size
        L = self.win_length
        b = self.board

        def check_direction(r0, c0, dr, dc):
            player = b[r0, c0]
            if player == 0:
                return None
            cnt = 0
            r, c = r0, c0
            while 0 <= r < bs and 0 <= c < bs and b[r, c] == player:
                cnt += 1
                if cnt >= L:
                    return int(player)
                r += dr
                c += dc
            return None

        # 모든 셀에서 출발해서 4방향으로만 검사하면 충분
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(bs):
            for c in range(bs):
                if b[r, c] == 0:
                    continue
                for dr, dc in directions:
                    # 시작점에서 역방향이 같은 돌이면 중복 검사 방지
                    prev_r, prev_c = r - dr, c - dc
                    if 0 <= prev_r < bs and 0 <= prev_c < bs and b[prev_r, prev_c] == b[r, c]:
                        continue
                    res = check_direction(r, c, dr, dc)
                    if res is not None:
                        return res

        # 무승부(보드 가득참)
        if np.all(b != 0):
            return 0
        return None

    def get_encoded_state(self) -> np.ndarray:
        # (3, board_size, board_size)
        encoded = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        encoded[0] = (self.board == self.current_player).astype(np.float32)
        encoded[1] = (self.board == -self.current_player).astype(np.float32)
        encoded[2] = float(self.current_player == 1)
        return encoded

    def get_flattened_state(self) -> np.ndarray:
        return self.board.flatten().astype(np.float32)

    def get_action_index(self, move: Tuple[int, int]) -> int:
        r, c = move
        return r * self.board_size + c

    def get_action_from_index(self, action_index: int) -> Tuple[int, int]:
        return (action_index // self.board_size, action_index % self.board_size)

    def copy(self) -> 'OmokGameState':
        return OmokGameState(self.board.copy(), self.current_player, self.board_size, self.win_length)

    def __str__(self) -> str:
        symbols = {1: '●', -1: '○', 0: '.'}
        rows = [' '.join(symbols[self.board[r, c]] for c in range(self.board_size)) for r in range(self.board_size)]
        return '\n'.join(rows)

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        valid = self.get_valid_moves()
        for r, c in valid:
            mask[self.get_action_index((r, c))] = 1.0
        return mask
