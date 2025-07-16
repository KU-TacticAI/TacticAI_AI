"""
알파제로용 오셀로 게임 상태 클래스

알파제로 알고리즘에서 사용할 수 있도록 최적화된 오셀로 게임 상태를 관리합니다.
"""

import numpy as np
from typing import List, Tuple, Optional, Any
import copy


class OthelloGameState:
    """알파제로용 오셀로 게임 상태 클래스"""
    
    def __init__(self, board: Optional[np.ndarray] = None, current_player: int = 1):
        """
        게임 상태 초기화
        
        Args:
            board: 8x8 게임 보드 (None이면 초기 상태)
            current_player: 현재 플레이어 (1: 흑, -1: 백)
        """
        if board is None:
            self.board = self._get_initial_board()
        else:
            self.board = board.copy()
        
        self.current_player = current_player
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                          (0, 1), (1, -1), (1, 0), (1, 1)]
    
        # NumPy 최적화를 위한 캐시
        self._valid_moves_cache = None
        self._valid_moves_cache_player = None
        self._action_mask_cache = None
        self._action_mask_cache_player = None
        
        # NumPy 최적화를 위한 미리 계산된 마스크들
        self._direction_masks = self._precompute_direction_masks()
    
    def _precompute_direction_masks(self) -> np.ndarray:
        """8방향 마스크를 미리 계산"""
        masks = np.zeros((8, 8, 8), dtype=np.int8)  # (8방향, 8x8)
        
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                     (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dir_idx, (dr, dc) in enumerate(directions):
            for row in range(8):
                for col in range(8):
                    r, c = row + dr, col + dc
                    if 0 <= r < 8 and 0 <= c < 8:
                        masks[dir_idx, row, col] = 1
        
        return masks
    
    def _get_initial_board(self) -> np.ndarray:
        """초기 보드 상태 반환"""
        board = np.zeros((8, 8), dtype=np.int8)
        # 초기 4개 돌 배치
        board[3, 3] = -1  # d4: 백돌
        board[3, 4] = 1   # e4: 흑돌
        board[4, 3] = 1   # d5: 흑돌
        board[4, 4] = -1  # e5: 백돌
        return board
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """현재 플레이어가 놓을 수 있는 유효한 위치 목록 반환"""
        # 캐시 확인
        if (self._valid_moves_cache is not None and 
            self._valid_moves_cache_player == self.current_player):
            return self._valid_moves_cache
        
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self._is_valid_move(row, col):
                    valid_moves.append((row, col))
        
        # 캐시 저장
        self._valid_moves_cache = valid_moves
        self._valid_moves_cache_player = self.current_player
        return valid_moves
    
    def _is_valid_move_vectorized(self, row: int, col: int) -> bool:
        """NumPy 벡터화된 유효한 수 검사"""
        if self.board[row, col] != 0:
            return False
        
        # 8방향을 한 번에 검사
        opponent = -self.current_player
        
        for dr, dc in self.directions:
            # dr이나 dc가 0인 경우 처리
            if dr == 0 and dc == 0:
                continue
                
            # 방향 벡터 생성 (0으로 나누기 방지)
            if dr != 0:
                r_coords = np.arange(row + dr, row + 8*dr, dr)
            else:
                r_coords = np.full(7, row)
                
            if dc != 0:
                c_coords = np.arange(col + dc, col + 8*dc, dc)
            else:
                c_coords = np.full(7, col)
            
            # 유효한 좌표만 필터링
            valid_mask = (r_coords >= 0) & (r_coords < 8) & (c_coords >= 0) & (c_coords < 8)
            r_coords = r_coords[valid_mask]
            c_coords = c_coords[valid_mask]
            
            if len(r_coords) == 0:
                continue
            
            # 첫 번째 칸이 상대방 돌인지 확인
            if self.board[r_coords[0], c_coords[0]] != opponent:
                continue
            
            # 연속된 상대방 돌 찾기
            opponent_stones = (self.board[r_coords, c_coords] == opponent)
            if not np.any(opponent_stones):
                continue
            
            # 마지막 상대방 돌 이후에 자신의 돌이 있는지 확인
            last_opponent_idx = np.where(opponent_stones)[0][-1]
            if last_opponent_idx + 1 < len(r_coords):
                if self.board[r_coords[last_opponent_idx + 1], c_coords[last_opponent_idx + 1]] == self.current_player:
                    return True
        
        return False
    
    def _is_valid_move(self, row: int, col: int) -> bool:
        """특정 위치에 돌을 놓는 것이 유효한지 확인 (기존 방식 - 호환성용)"""
        if self.board[row, col] != 0:
            return False
        
        for dr, dc in self.directions:
            if self._can_flip_in_direction(row, col, dr, dc):
                return True
        
        return False
    
    def _can_flip_in_direction(self, row: int, col: int, dr: int, dc: int) -> bool:
        """특정 방향으로 돌을 뒤집을 수 있는지 확인"""
        opponent = -self.current_player
        r, c = row + dr, col + dc
        
        # 첫 번째 칸이 상대방 돌이어야 함
        if not (0 <= r < 8 and 0 <= c < 8) or self.board[r, c] != opponent:
            return False
        
        # 상대방 돌이 연속으로 있는지 확인
        r += dr
        c += dc
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r, c] == 0:
                return False
            elif self.board[r, c] == self.current_player:
                return True
            r += dr
            c += dc
        
        return False
    
    def make_move(self, move: Tuple[int, int]) -> 'OthelloGameState':
        """수를 실행하고 새로운 게임 상태 반환"""
        if move not in self.get_valid_moves():
            raise ValueError(f"Invalid move: {move}")
        
        new_state = OthelloGameState(self.board.copy(), self.current_player)
        row, col = move
        
        # 돌을 놓고 뒤집기
        new_state.board[row, col] = self.current_player
        new_state._flip_stones(row, col)
        
        # 플레이어 변경
        new_state.current_player = -self.current_player
        
        # 캐시 무효화
        new_state._valid_moves_cache = None
        new_state._valid_moves_cache_player = None
        new_state._action_mask_cache = None
        new_state._action_mask_cache_player = None
        
        return new_state
    
    def _flip_stones(self, row: int, col: int) -> None:
        """돌을 놓은 후 8방향으로 뒤집을 수 있는 돌들을 뒤집음"""
        for dr, dc in self.directions:
            flipped_positions = []
            r, c = row + dr, col + dc
            
            # 이 방향으로 뒤집을 수 있는 돌들을 찾음
            while (0 <= r < 8 and 0 <= c < 8 and 
                   self.board[r, c] == -self.current_player):
                flipped_positions.append((r, c))
                r += dr
                c += dc
            
            # 마지막에 자신의 돌이 있으면 뒤집기
            if (0 <= r < 8 and 0 <= c < 8 and 
                self.board[r, c] == self.current_player):
                for flip_row, flip_col in flipped_positions:
                    self.board[flip_row, flip_col] = self.current_player
    
    def is_game_over(self) -> bool:
        """게임이 종료되었는지 확인"""
        # 보드가 가득 찬 경우
        if np.all(self.board != 0):
            return True
        
        # 양쪽 모두 유효한 수가 없는 경우
        current_moves = self.get_valid_moves()
        if len(current_moves) > 0:
            return False
        
        # 상대방 차례에서도 유효한 수가 없는지 확인
        opponent_state = OthelloGameState(self.board.copy(), -self.current_player)
        opponent_moves = opponent_state.get_valid_moves()
        
        return len(opponent_moves) == 0
    
    def get_winner(self) -> Optional[int]:
        """승자 반환 (1: 흑승, -1: 백승, 0: 무승부, None: 게임 진행 중)"""
        if not self.is_game_over():
            return None
        
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        
        if black_count > white_count:
            return 1
        elif white_count > black_count:
            return -1
        else:
            return 0
    
    def get_score(self) -> Tuple[int, int]:
        """흑돌과 백돌의 개수 반환"""
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        return black_count, white_count
    
    def get_encoded_state(self) -> np.ndarray:
        """
        신경망 입력용 인코딩된 상태 반환 (NumPy 벡터화)
        
        Returns:
            (3, 8, 8) 형태의 배열
            - 채널 0: 현재 플레이어의 돌 (1 또는 0)
            - 채널 1: 상대방의 돌 (1 또는 0)
            - 채널 2: 현재 플레이어 표시 (1 또는 0)
        """
        encoded = np.zeros((3, 8, 8), dtype=np.float32)
        
        # 벡터화된 마스킹 연산
        encoded[0] = (self.board == self.current_player).astype(np.float32)
        encoded[1] = (self.board == -self.current_player).astype(np.float32)
        encoded[2] = float(self.current_player == 1)
        
        return encoded
    
    def get_flattened_state(self) -> np.ndarray:
        """
        1차원 64길이 배열로 인코딩된 상태 반환
        
        Returns:
            (64,) 형태의 배열
            - 검은돌(1): 1.0
            - 흰돌(-1): -1.0
            - 빈칸(0): 0.0
        """
        return self.board.flatten().astype(np.float32)
    
    def get_simple_encoded_state(self) -> np.ndarray:
        """
        간단한 2채널 인코딩된 상태 반환 (현재 플레이어 관점)
        
        Returns:
            (2, 8, 8) 형태의 배열
            - 채널 0: 현재 플레이어의 돌 (1 또는 0)
            - 채널 1: 상대방의 돌 (1 또는 0)
        """
        encoded = np.zeros((2, 8, 8), dtype=np.float32)
        
        # 채널 0: 현재 플레이어의 돌
        encoded[0] = (self.board == self.current_player).astype(np.float32)
        
        # 채널 1: 상대방의 돌
        encoded[1] = (self.board == -self.current_player).astype(np.float32)
        
        return encoded
    
    def get_encoded_state_from_flattened(self, flattened_board: np.ndarray) -> np.ndarray:
        """
        1차원 64길이 배열을 3채널 인코딩으로 변환
        
        Args:
            flattened_board: 1차원 보드 배열 (64,) - 검은돌(1), 흰돌(-1), 빈칸(0)
            
        Returns:
            (3, 8, 8) 형태의 배열 (기존 인코딩과 동일)
            - 채널 0: 현재 플레이어의 돌 (1 또는 0)
            - 채널 1: 상대방의 돌 (1 또는 0)
            - 채널 2: 현재 플레이어 표시 (1 또는 0)
        """
        # 1차원 배열을 2차원으로 변환
        board_2d = flattened_board.reshape(8, 8)
        
        # 3채널 인코딩 생성
        encoded = np.zeros((3, 8, 8), dtype=np.float32)
        
        # 채널 0: 현재 플레이어의 돌
        encoded[0] = (board_2d == self.current_player).astype(np.float32)
        
        # 채널 1: 상대방의 돌
        encoded[1] = (board_2d == -self.current_player).astype(np.float32)
        
        # 채널 2: 현재 플레이어 표시 (흑이면 1, 백이면 0)
        encoded[2] = float(self.current_player == 1)
        
        return encoded
    
    def get_encoded_state_from_65_array(self, array_65: np.ndarray) -> np.ndarray:
        """
        65길이 배열을 3채널 인코딩으로 변환
        
        Args:
            array_65: 65길이 배열
                - 인덱스 0: 턴 정보 (1: 흑돌, -1: 백돌)
                - 인덱스 1-64: 보드 상황 (검은돌: 1, 흰돌: -1, 빈칸: 0)
            
        Returns:
            (3, 8, 8) 형태의 배열
        """
        current_player = array_65[0]
        board_1d = array_65[1:65]
        board_2d = board_1d.reshape(8, 8)
        
        # 3채널 인코딩 생성
        encoded = np.zeros((3, 8, 8), dtype=np.float32)
        
        # 채널 0: 현재 플레이어의 돌
        encoded[0] = (board_2d == current_player).astype(np.float32)
        
        # 채널 1: 상대방의 돌
        encoded[1] = (board_2d == -current_player).astype(np.float32)
        
        # 채널 2: 현재 플레이어 표시 (흑이면 1, 백이면 0)
        encoded[2] = float(current_player == 1)
        
        return encoded
    
    def get_action_index(self, move: Tuple[int, int] or str) -> int:
        """액션을 인덱스로 변환"""
        if move == "pass":
            return 0
        row, col = move
        return row * 8 + col + 1  # 1-64 (0은 패스)
    
    def get_action_from_index(self, action_index: int) -> Tuple[int, int] or str:
        """인덱스를 액션으로 변환"""
        if action_index == 0:
            return "pass"
        action_index -= 1  # 1-64를 0-63으로 변환
        row = action_index // 8
        col = action_index % 8
        return (row, col)
    
    def copy(self) -> 'OthelloGameState':
        """게임 상태 복사"""
        return OthelloGameState(self.board.copy(), self.current_player)
    
    def __str__(self) -> str:
        """게임 상태 문자열 표현"""
        result = "  a b c d e f g h\n"
        for row in range(8):
            result += f"{row} "
            for col in range(8):
                if self.board[row, col] == 1:
                    result += "● "
                elif self.board[row, col] == -1:
                    result += "○ "
                else:
                    result += ". "
            result += f"{row}\n"
        result += "  a b c d e f g h\n"
        result += f"현재 플레이어: {'흑돌' if self.current_player == 1 else '백돌'}\n"
        return result 

    def get_action_mask(self) -> np.ndarray:
        """NumPy 벡터화된 액션 마스크 생성"""
        # 캐시 확인
        if (self._action_mask_cache is not None and 
            self._action_mask_cache_player == self.current_player):
            return self._action_mask_cache
        
        mask = np.zeros(65, dtype=np.float32)
        
        # 패스 액션 (인덱스 0)은 항상 유효
        mask[0] = 1.0
        
        # 보드 위치 액션들
        valid_moves = self.get_valid_moves()
        for row, col in valid_moves:
            action_idx = self.get_action_index((row, col))
            mask[action_idx] = 1.0
        
        # 캐시 저장
        self._action_mask_cache = mask
        self._action_mask_cache_player = self.current_player
        
        return mask 