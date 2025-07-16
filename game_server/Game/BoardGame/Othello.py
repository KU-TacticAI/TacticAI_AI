#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
오셀로(리버시) 게임 구현
BoardGame 클래스를 상속받아 구현된 완전한 오셀로 게임

게임 규칙:
- 8x8 보드에서 진행
- 흑돌(1)과 백돌(-1)로 게임
- 상대방 돌을 사이에 두고 자신의 돌을 놓으면 사이의 돌들이 뒤집힘
- 유효한 수가 없으면 패스
- 양쪽 모두 패스하거나 보드가 가득 차면 게임 종료
- 더 많은 돌을 가진 플레이어가 승리
"""

import os
import logging
from typing import List, Tuple, Optional, Any, Dict
from .BoardGame import BoardGame

logger = logging.getLogger(__name__)


class Othello(BoardGame):
    """오셀로 게임 클래스"""
    
    def __init__(self, game_id: str, players: List[str], max_history: Optional[int] = None):
        """
        오셀로 게임 초기화
        
        Args:
            game_id: 게임 ID
            players: 플레이어 목록 (2명)
            max_history: 최대 히스토리 기록 수
        """
        super().__init__(game_id, players, max_history)
        self.board: List[List[int]] = []
        self.consecutive_passes = 0  # 연속 패스 횟수
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                          (0, 1), (1, -1), (1, 0), (1, 1)]  # 8방향
    
    def _get_mapping_file_path(self) -> Optional[str]:
        """오셀로 맵핑 파일 경로 반환"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "Othello.csv")
    
    def initialize(self, **kwargs) -> None:
        """
        오셀로 보드 초기화
        
        보드 표현:
        - 0: 빈 칸
        - 1: 흑돌 (첫 번째 플레이어)
        - -1: 백돌 (두 번째 플레이어)
        """
        # 8x8 빈 보드 생성
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        
        # 초기 4개 돌 배치 (중앙 2x2 영역)
        self.board[3][3] = -1  # d4: 백돌
        self.board[3][4] = 1   # e4: 흑돌
        self.board[4][3] = 1   # d5: 흑돌
        self.board[4][4] = -1  # e5: 백돌
        
        # 게임 상태 초기화
        self.current_player = 0  # 흑돌(플레이어 0)이 먼저 시작
        self.consecutive_passes = 0
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        현재 플레이어가 놓을 수 있는 유효한 위치 목록 반환
        
        Returns:
            유효한 (row, col) 위치 목록
        """
        valid_moves = []
        player_stone = 1 if self.current_player == 0 else -1
        
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == 0:  # 빈 칸인 경우
                    if self._is_valid_move(row, col, player_stone):
                        valid_moves.append((row, col))
        
        return valid_moves
    
    def _is_valid_move(self, row: int, col: int, player_stone: int) -> bool:
        """
        특정 위치에 돌을 놓는 것이 유효한지 확인
        
        Args:
            row: 행 위치
            col: 열 위치
            player_stone: 플레이어의 돌 (1 또는 -1)
            
        Returns:
            유효한 수인지 여부
        """
        if self.board[row][col] != 0:  # 이미 돌이 있는 경우
            return False
        
        # 8방향으로 뒤집을 수 있는 돌이 있는지 확인
        for dr, dc in self.directions:
            if self._can_flip_in_direction(row, col, dr, dc, player_stone):
                return True
        
        return False
    
    def _can_flip_in_direction(self, row: int, col: int, dr: int, dc: int, 
                              player_stone: int) -> bool:
        """
        특정 방향으로 돌을 뒤집을 수 있는지 확인
        
        Args:
            row, col: 시작 위치
            dr, dc: 방향 벡터
            player_stone: 플레이어의 돌
            
        Returns:
            해당 방향으로 뒤집을 수 있는지 여부
        """
        opponent_stone = -player_stone
        r, c = row + dr, col + dc
        
        # 첫 번째 칸이 상대방 돌이어야 함
        if not (0 <= r < 8 and 0 <= c < 8) or self.board[r][c] != opponent_stone:
            return False
        
        # 상대방 돌이 연속으로 있는지 확인하고, 마지막에 자신의 돌이 있는지 확인
        r += dr
        c += dc
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r][c] == 0:  # 빈 칸
                return False
            elif self.board[r][c] == player_stone:  # 자신의 돌
                return True
            # 상대방 돌이면 계속 진행
            r += dr
            c += dc
        
        return False
    
    def make_move(self, move: Any) -> bool:
        """
        수를 실행하고 돌들을 뒤집음
        
        Args:
            move: (row, col) 튜플 또는 "pass" 문자열
            
        Returns:
            성공 여부
        """
        if move == "pass":
            self.consecutive_passes += 1
            return True
        
        if not isinstance(move, tuple) or len(move) != 2:
            return False
        
        row, col = move
        if not (0 <= row < 8 and 0 <= col < 8):
            return False
        
        player_stone = 1 if self.current_player == 0 else -1
        
        if not self._is_valid_move(row, col, player_stone):
            return False
        
        # 돌을 놓고 뒤집기
        self.board[row][col] = player_stone
        flipped = self._flip_stones(row, col, player_stone)
        
        if flipped > 0:
            self.consecutive_passes = 0  # 유효한 수가 있었으므로 패스 카운트 리셋
            return True
        else:
            # 뒤집은 돌이 없다면 원래대로 되돌림 (이론적으로는 발생하지 않아야 함)
            self.board[row][col] = 0
            return False
    
    def _flip_stones(self, row: int, col: int, player_stone: int) -> int:
        """
        돌을 놓은 후 8방향으로 뒤집을 수 있는 돌들을 뒤집음
        
        Args:
            row, col: 돌을 놓은 위치
            player_stone: 플레이어의 돌
            
        Returns:
            뒤집은 돌의 개수
        """
        total_flipped = 0
        
        for dr, dc in self.directions:
            flipped_positions = []
            r, c = row + dr, col + dc
            
            # 이 방향으로 뒤집을 수 있는 돌들을 찾음
            while (0 <= r < 8 and 0 <= c < 8 and 
                   self.board[r][c] == -player_stone):
                flipped_positions.append((r, c))
                r += dr
                c += dc
            
            # 마지막이 자신의 돌이면 뒤집기 실행
            if (0 <= r < 8 and 0 <= c < 8 and 
                self.board[r][c] == player_stone and flipped_positions):
                for flip_r, flip_c in flipped_positions:
                    self.board[flip_r][flip_c] = player_stone
                    total_flipped += 1
        
        return total_flipped
    
    def is_game_over(self) -> bool:
        """
        게임 종료 조건 확인
        
        Returns:
            게임이 종료되었는지 여부
        """
        # 두 플레이어가 연속으로 패스한 경우
        if self.consecutive_passes >= 2:
            return True
        
        # 보드가 가득 찬 경우
        if self._is_board_full():
            return True
        
        # 양쪽 플레이어 모두 유효한 수가 없는 경우
        if not self._has_valid_moves(0) and not self._has_valid_moves(1):
            return True
        
        return False
    
    def _is_board_full(self) -> bool:
        """보드가 가득 찼는지 확인"""
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == 0:
                    return False
        return True
    
    def _has_valid_moves(self, player: int) -> bool:
        """특정 플레이어가 유효한 수를 가지고 있는지 확인"""
        player_stone = 1 if player == 0 else -1
        
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == 0:
                    if self._is_valid_move(row, col, player_stone):
                        return True
        return False
    
    def get_winner(self) -> Optional[str]:
        """
        승자 결정
        
        Returns:
            승자의 플레이어 이름 또는 None (무승부)
        """
        if not self.is_game_over():
            return None
        
        # 각 플레이어의 돌 개수 계산
        black_count = 0
        white_count = 0
        
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == 1:
                    black_count += 1
                elif self.board[row][col] == -1:
                    white_count += 1
        
        if black_count > white_count:
            return self.players[0]  # 흑돌 승리
        elif white_count > black_count:
            return self.players[1]  # 백돌 승리
        else:
            return None  # 무승부
    
    def _convert_move_format(self, move_str: str) -> Any:
        """
        문자열 형태의 수를 게임에서 사용하는 형태로 변환
        
        Args:
            move_str: "a1", "h8", "pass" 등의 문자열
            
        Returns:
            (row, col) 튜플 또는 "pass" 문자열
        """
        if not move_str or not isinstance(move_str, str):
            return None
        
        move_str = move_str.strip().lower()
        
        if move_str == "pass":
            return "pass"
        
        if len(move_str) != 2:
            return None
        
        try:
            col_char = move_str[0]
            row_char = move_str[1]
            
            if not ('a' <= col_char <= 'h') or not ('1' <= row_char <= '8'):
                return None
            
            col = ord(col_char) - ord('a')  # a=0, b=1, ..., h=7
            row = int(row_char) - 1         # 1=0, 2=1, ..., 8=7
            
            return (row, col)
        except (ValueError, IndexError):
            return None
    
    def get_score(self) -> Dict[str, int]:
        """
        현재 점수 반환
        
        Returns:
            각 플레이어의 돌 개수
        """
        black_count = 0
        white_count = 0
        
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == 1:
                    black_count += 1
                elif self.board[row][col] == -1:
                    white_count += 1
        
        return {
            self.players[0]: black_count,
            self.players[1]: white_count
        }
    
    def get_board_state(self) -> List[List[int]]:
        """
        현재 보드 상태의 복사본 반환
        
        Returns:
            보드 상태 복사본
        """
        return [row[:] for row in self.board]
    
    def print_board(self) -> str:
        """
        보드를 사람이 읽기 쉬운 형태로 출력
        
        Returns:
            보드 상태 문자열
        """
        symbols = {0: '.', 1: '●', -1: '○'}
        board_str = "  a b c d e f g h\n"
        
        for i, row in enumerate(self.board):
            board_str += f"{i+1} "
            for cell in row:
                board_str += f"{symbols[cell]} "
            board_str += f"{i+1}\n"
        
        board_str += "  a b c d e f g h\n"
        
        # 점수 정보 추가
        scores = self.get_score()
        board_str += f"\n점수: {self.players[0]}(●): {scores[self.players[0]]}, "
        board_str += f"{self.players[1]}(○): {scores[self.players[1]]}\n"
        board_str += f"현재 턴: {self.players[self.current_player]}\n"
        
        return board_str
    
    def get_game_status(self) -> Dict[str, Any]:
        """
        현재 게임 상태 정보 반환
        
        Returns:
            게임 상태 정보 딕셔너리
        """
        valid_moves = self.get_valid_moves()
        scores = self.get_score()
        
        return {
            "current_player": self.players[self.current_player],
            "is_black_turn": self.current_player == 0,
            "valid_moves_count": len(valid_moves),
            "valid_moves": valid_moves,
            "scores": scores,
            "consecutive_passes": self.consecutive_passes,
            "board_full": self._is_board_full(),
            "game_over": self.is_game_over(),
            "winner": self.get_winner()
        }
    
    def can_pass(self) -> bool:
        """
        현재 플레이어가 패스할 수 있는지 확인
        (유효한 수가 없는 경우에만 패스 가능)
        
        Returns:
            패스 가능 여부
        """
        return len(self.get_valid_moves()) == 0

    def select_best_move_from_probabilities(self, probabilities: List[float]) -> Optional[Any]:
        """
        확률 배열에서 최적의 수를 선택 (오셀로 전용)
        유효한 수가 없으면 자동으로 패스를 반환
        
        Args:
            probabilities: 액션 확률 배열 (길이 65: 64개 위치 + 1개 패스)
            
        Returns:
            선택된 수 (row, col) 또는 "pass"
        """
        if not self.index_mapping:
            logger.error("인덱스 맵핑이 로드되지 않았습니다.")
            return None
            
        import numpy as np
        if isinstance(probabilities, list):
            probabilities = np.array(probabilities)
        
        # 현재 플레이어의 유효한 수 확인
        valid_moves = self.get_valid_moves()
        
        # 유효한 수가 없으면 패스
        if len(valid_moves) == 0:
            logger.info(f"플레이어 {self.current_player}에게 유효한 수가 없음 - 패스 처리")
            return "pass"
        
        # 확률이 높은 순서로 정렬하여 유효한 수 찾기
        sorted_indices = np.argsort(probabilities)[::-1]
        
        for idx in sorted_indices:
            if idx in self.index_mapping:
                move_str = self.index_mapping[idx]
                converted_move = self._convert_move_format(move_str)
                
                # 패스는 유효한 수가 없을 때만 허용
                if converted_move == "pass" and len(valid_moves) == 0:
                    return "pass"
                elif converted_move != "pass" and converted_move in valid_moves:
                    return converted_move
        
        # 유효한 수를 찾지 못했지만 유효한 수가 있다면 첫 번째 유효한 수 선택
        if valid_moves:
            logger.warning(f"확률 배열에서 유효한 수를 찾지 못함. 첫 번째 유효한 수 선택: {valid_moves[0]}")
            return valid_moves[0]
        
        # 이론적으로 도달하지 않아야 하는 경우
        logger.error("유효한 수도 없고 패스도 불가능한 상황")
        return None
