from typing import List, Tuple, Optional, Any, Dict
import logging
import os
from .BoardGame import BoardGame

logger = logging.getLogger(__name__)


class Omok(BoardGame):
    """
    오목(Gomoku) 게임 구현
    - 기본 규칙: 바둑판 위에 교대로 돌을 놓아 가로/세로/대각선으로 연속 5개를 만든 플레이어가 승리
    - 보드 크기는 기본 15x15 (변경 가능)
    """

    def __init__(self, game_id: str, players: List[str], max_history: Optional[int] = None, size: int = 15):
        super().__init__(game_id, players, max_history)
        self.size = max(5, size)
        self.board: List[List[int]] = []
        self.winner: Optional[int] = None  # 0 or 1 for index into players, None if no winner yet

    def _get_mapping_file_path(self) -> Optional[str]:
        """맵핑 CSV는 기본 제공하지 않음(좌표 문자열을 직접 파싱함)."""
        return None

    def initialize(self, **kwargs) -> None:
        """보드를 초기화하고 현재 플레이어를 0으로 설정."""
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.current_player = 0
        self.winner = None
        logger.info(f"Omok 보드가 초기화되었습니다: size={self.size}x{self.size}")

    def _is_within(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """빈 칸의 (row, col) 튜플 목록 반환."""
        moves: List[Tuple[int, int]] = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    moves.append((r, c))
        return moves

    def _convert_move_format(self, move_str: str) -> Any:
        """
        문자열 좌표를 (row, col)으로 변환.
        예: 'a1' -> (0,0), 'h8' -> (7,7)
        컬럼은 알파벳(a,o 등), 행은 1-based 숫자.
        """
        if not move_str or not isinstance(move_str, str):
            return None
        s = move_str.strip().lower()
        # 단순한 형식: letter(s) + number(s), e.g. 'a1', 'o15'
        # 컬럼 문자 하나를 예상하지만 여러 문자도 처리 (예: 'aa1'는 비정상)
        col_char = s[0]
        row_part = s[1:]
        if not col_char.isalpha() or not row_part.isdigit():
            return None
        col = ord(col_char) - ord('a')
        try:
            row = int(row_part) - 1
        except ValueError:
            return None
        if not self._is_within(row, col):
            return None
        return (row, col)

    def make_move(self, move: Tuple[int, int]) -> bool:
        """돌을 놓고 승리 여부를 검사한다. 성공 시 True 반환."""
        if isinstance(move, str):
            parsed = self._convert_move_format(move)
            if parsed is None:
                logger.warning(f"Omok: 잘못된 문자열 수: {move}")
                return False
            move = parsed

        if not isinstance(move, tuple) or len(move) != 2:
            logger.warning("Omok: 잘못된 수 형식")
            return False
        row, col = move
        if not self._is_within(row, col):
            logger.warning(f"Omok: 범위 벗어난 수: {(row, col)}")
            return False
        if self.board[row][col] != 0:
            logger.warning(f"Omok: 이미 차 있는 칸: {(row, col)}")
            return False

        # 놓기 (첫 플레이어는 0 -> 값을 1, 두 번째는 -1)
        value = 1 if self.current_player == 0 else -1
        self.board[row][col] = value

        # 히스토리 기록을 위해 move를 사람이 이해하기 쉬운 문자열로 저장
        human_move = f"{chr(col + ord('a'))}{row + 1}"
        # record_history는 BoardGame.play에서 호출되므로 여기서는 저장하지 않음.

        # 승리 검사
        if self._check_five_in_a_row(row, col, value):
            self.winner = self.current_player
            logger.info(f"Omok: 승리 발생 - player={self.players[self.winner]} at {(row, col)}")

        return True

    def _count_in_direction(self, row: int, col: int, dr: int, dc: int, value: int) -> int:
        cnt = 0
        r, c = row + dr, col + dc
        while self._is_within(r, c) and self.board[r][c] == value:
            cnt += 1
            r += dr
            c += dc
        return cnt

    def _check_five_in_a_row(self, row: int, col: int, value: int) -> bool:
        """주어진 위치에서 5목인지 검사한다."""
        # 4방향(대칭 포함) 검사: (0,1), (1,0), (1,1), (1,-1)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            cnt = 1  # 현재 돌
            cnt += self._count_in_direction(row, col, dr, dc, value)
            cnt += self._count_in_direction(row, col, -dr, -dc, value)
            if cnt >= 5:
                return True
        return False

    def is_game_over(self) -> bool:
        """승자 존재 혹은 보드가 이미 찬 경우 게임 종료."""
        if self.winner is not None:
            return True
        # 보드가 가득 찼는지
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    return False
        return True

    def get_winner(self) -> Optional[str]:
        if self.winner is None:
            return None
        return self.players[self.winner]

    def get_board_state(self) -> List[List[int]]:
        return [row[:] for row in self.board]

    def print_board(self) -> str:
        """사람이 읽기 쉬운 문자열로 보드를 반환한다."""
        cols = ''.join(chr(i + ord('a')) for i in range(self.size))
        header = '   ' + ' '.join(cols) + '\n'
        lines = [header]
        for r in range(self.size):
            line = f"{r+1:2d} "
            for c in range(self.size):
                v = self.board[r][c]
                if v == 0:
                    line += ' .'
                elif v == 1:
                    line += ' X'
                else:
                    line += ' O'
            lines.append(line + '\n')
        return ''.join(lines)

    def get_game_status(self) -> Dict[str, Any]:
        valid_moves = self.get_valid_moves()
        return {
            "current_player": self.players[self.current_player],
            "valid_moves_count": len(valid_moves),
            "valid_moves": valid_moves,
            "board_full": all(self.board[r][c] != 0 for r in range(self.size) for c in range(self.size)),
            "game_over": self.is_game_over(),
            "winner": self.get_winner(),
            "move_count": sum(1 for r in range(self.size) for c in range(self.size) if self.board[r][c] != 0)
        }

    # 편의 함수
    def get_empty_positions(self) -> List[Tuple[int, int]]:
        return self.get_valid_moves()

    def can_pass(self) -> bool:
        return False

    def select_best_move_from_probabilities(self, probabilities) -> Optional[Any]:
        """
        Omok-specific override: probabilities is expected to be a flat array of length size*size.
        Return the highest-probability empty cell as (row, col), or None if none available.
        """
        try:
            import numpy as _np
        except Exception:
            _np = None

        if probabilities is None:
            logger.error("Omok: probabilities is None")
            return None

        if isinstance(probabilities, list):
            probs = _np.array(probabilities) if _np is not None else probabilities
        else:
            probs = probabilities

        # Ensure we have a numpy array for easy processing
        if _np is None:
            # Fallback: iterate over list
            flat = list(probs)
            if len(flat) != self.size * self.size:
                logger.warning(f"Omok: 확률 배열 길이가 보드와 다름: {len(flat)} vs {self.size*self.size}")
                # try to truncate or pad
                flat = (flat + [0] * (self.size*self.size))[:self.size*self.size]
            # iterate by sorted indices
            sorted_idx = sorted(range(len(flat)), key=lambda i: flat[i], reverse=True)
            for idx in sorted_idx:
                r, c = divmod(idx, self.size)
                if self.board[r][c] == 0:
                    return (r, c)
            # no empty
            return None
        else:
            probs = _np.array(probs).flatten()
            if probs.size != self.size * self.size:
                logger.warning(f"Omok: 확률 배열 길이가 보드와 다름: {probs.size} vs {self.size*self.size}")
                # truncate or pad
                if probs.size < self.size * self.size:
                    probs = _np.pad(probs, (0, self.size*self.size - probs.size))
                else:
                    probs = probs[:self.size*self.size]
            sorted_idx = _np.argsort(probs)[::-1]
            for idx in sorted_idx:
                r = int(idx // self.size)
                c = int(idx % self.size)
                if self.board[r][c] == 0:
                    return (r, c)
            return None
