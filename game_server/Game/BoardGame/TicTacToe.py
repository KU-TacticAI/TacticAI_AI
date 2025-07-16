#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
틱택토(Tic-Tac-Toe) 게임 구현
BoardGame 클래스를 상속받아 구현된 완전한 틱택토 게임

게임 규칙:
- 3x3 보드에서 진행
- X(1)와 O(-1)로 게임
- 가로, 세로, 대각선 중 하나를 먼저 완성하면 승리
- 보드가 가득 차면 무승부
"""

import os
from typing import List, Tuple, Optional, Any, Dict
from .BoardGame import BoardGame


class TicTacToe(BoardGame):
    """틱택토 게임 클래스"""
    
    def __init__(self, game_id: str, players: List[str], max_history: Optional[int] = None):
        """
        틱택토 게임 초기화
        
        Args:
            game_id: 게임 ID
            players: 플레이어 목록 (2명)
            max_history: 최대 히스토리 기록 수
        """
        super().__init__(game_id, players, max_history)
        self.board: List[List[int]] = []
    
    def _get_mapping_file_path(self) -> Optional[str]:
        """틱택토 맵핑 파일 경로 반환"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "TicTacToe.csv")
    
    def initialize(self, **kwargs) -> None:
        """
        틱택토 보드 초기화
        
        보드 표현:
        - 0: 빈 칸
        - 1: X (첫 번째 플레이어)
        - -1: O (두 번째 플레이어)
        """
        # 3x3 빈 보드 생성
        self.board = [[0 for _ in range(3)] for _ in range(3)]
        
        # 게임 상태 초기화
        self.current_player = 0  # X(플레이어 0)가 먼저 시작
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        현재 플레이어가 놓을 수 있는 유효한 위치 목록 반환
        
        Returns:
            유효한 (row, col) 위치 목록
        """
        valid_moves = []
        
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == 0:  # 빈 칸인 경우
                    valid_moves.append((row, col))
        
        return valid_moves
    
    def make_move(self, move: Any) -> bool:
        """
        수를 실행하고 보드에 표시
        
        Args:
            move: (row, col) 튜플
            
        Returns:
            성공 여부
        """
        if not isinstance(move, tuple) or len(move) != 2:
            return False
        
        row, col = move
        if not (0 <= row < 3 and 0 <= col < 3):
            return False
        
        if self.board[row][col] != 0:  # 이미 표시된 칸
            return False
        
        # 현재 플레이어의 표시 (X=1, O=-1)
        player_mark = 1 if self.current_player == 0 else -1
        self.board[row][col] = player_mark
        
        return True
    
    def is_game_over(self) -> bool:
        """
        게임 종료 조건 확인
        
        Returns:
            게임이 종료되었는지 여부
        """
        # 승리 조건 확인
        if self._check_winner() is not None:
            return True
        
        # 보드가 가득 찬 경우 (무승부)
        if self._is_board_full():
            return True
        
        return False
    
    def _check_winner(self) -> Optional[int]:
        """
        승리자 확인
        
        Returns:
            승리한 플레이어 (0 또는 1) 또는 None
        """
        # 가로줄 확인
        for row in range(3):
            if (self.board[row][0] == self.board[row][1] == self.board[row][2] 
                and self.board[row][0] != 0):
                return 0 if self.board[row][0] == 1 else 1
        
        # 세로줄 확인
        for col in range(3):
            if (self.board[0][col] == self.board[1][col] == self.board[2][col] 
                and self.board[0][col] != 0):
                return 0 if self.board[0][col] == 1 else 1
        
        # 대각선 확인 (왼쪽 위 -> 오른쪽 아래)
        if (self.board[0][0] == self.board[1][1] == self.board[2][2] 
            and self.board[0][0] != 0):
            return 0 if self.board[0][0] == 1 else 1
        
        # 대각선 확인 (오른쪽 위 -> 왼쪽 아래)
        if (self.board[0][2] == self.board[1][1] == self.board[2][0] 
            and self.board[0][2] != 0):
            return 0 if self.board[0][2] == 1 else 1
        
        return None
    
    def _is_board_full(self) -> bool:
        """보드가 가득 찼는지 확인"""
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == 0:
                    return False
        return True
    
    def get_winner(self) -> Optional[str]:
        """
        승자 결정
        
        Returns:
            승자의 플레이어 이름 또는 None (무승부)
        """
        if not self.is_game_over():
            return None
        
        winner_player = self._check_winner()
        if winner_player is not None:
            return self.players[winner_player]
        else:
            return None  # 무승부
    
    def _convert_move_format(self, move_str: str) -> Any:
        """
        문자열 형태의 수를 게임에서 사용하는 형태로 변환
        
        Args:
            move_str: "a1", "b2", "c3" 등의 문자열
            
        Returns:
            (row, col) 튜플
        """
        if not move_str or not isinstance(move_str, str):
            return None
        
        move_str = move_str.strip().lower()
        
        if len(move_str) != 2:
            return None
        
        try:
            col_char = move_str[0]
            row_char = move_str[1]
            
            if not ('a' <= col_char <= 'c') or not ('1' <= row_char <= '3'):
                return None
            
            col = ord(col_char) - ord('a')  # a=0, b=1, c=2
            row = int(row_char) - 1         # 1=0, 2=1, 3=2
            
            return (row, col)
        except (ValueError, IndexError):
            return None
    
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
        symbols = {0: '.', 1: 'X', -1: 'O'}
        board_str = "  a b c\n"
        
        for i, row in enumerate(self.board):
            board_str += f"{i+1} "
            for cell in row:
                board_str += f"{symbols[cell]} "
            board_str += f"{i+1}\n"
        
        board_str += "  a b c\n"
        
        # 현재 턴 정보 추가
        if not self.is_game_over():
            current_symbol = "X" if self.current_player == 0 else "O"
            board_str += f"\n현재 턴: {self.players[self.current_player]} ({current_symbol})\n"
        else:
            winner = self.get_winner()
            if winner:
                board_str += f"\n🎉 승자: {winner}!\n"
            else:
                board_str += f"\n🤝 무승부!\n"
        
        return board_str
    
    def get_game_status(self) -> Dict[str, Any]:
        """
        현재 게임 상태 정보 반환
        
        Returns:
            게임 상태 정보 딕셔너리
        """
        valid_moves = self.get_valid_moves()
        
        return {
            "current_player": self.players[self.current_player],
            "is_x_turn": self.current_player == 0,
            "valid_moves_count": len(valid_moves),
            "valid_moves": valid_moves,
            "board_full": self._is_board_full(),
            "game_over": self.is_game_over(),
            "winner": self.get_winner(),
            "turn_number": 9 - len(valid_moves) + 1 if not self.is_game_over() else 9 - len(valid_moves)
        }
    
    def get_empty_positions(self) -> List[Tuple[int, int]]:
        """
        빈 위치들의 목록 반환 (get_valid_moves와 동일)
        
        Returns:
            빈 위치들의 (row, col) 목록
        """
        return self.get_valid_moves()
    
    def is_position_empty(self, row: int, col: int) -> bool:
        """
        특정 위치가 비어있는지 확인
        
        Args:
            row: 행 위치 (0-2)
            col: 열 위치 (0-2)
            
        Returns:
            해당 위치가 비어있는지 여부
        """
        if not (0 <= row < 3 and 0 <= col < 3):
            return False
        return self.board[row][col] == 0
    
    def get_position_value(self, row: int, col: int) -> int:
        """
        특정 위치의 값 반환
        
        Args:
            row: 행 위치 (0-2)
            col: 열 위치 (0-2)
            
        Returns:
            해당 위치의 값 (0: 빈칸, 1: X, -1: O)
        """
        if not (0 <= row < 3 and 0 <= col < 3):
            return 0
        return self.board[row][col]
    
    def count_marks(self) -> Dict[str, int]:
        """
        각 플레이어의 표시 개수 계산
        
        Returns:
            각 플레이어의 X, O 개수
        """
        x_count = 0
        o_count = 0
        
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == 1:
                    x_count += 1
                elif self.board[row][col] == -1:
                    o_count += 1
        
        return {
            self.players[0]: x_count,  # X 플레이어
            self.players[1]: o_count   # O 플레이어
        }
    
    def get_winning_positions(self) -> List[List[Tuple[int, int]]]:
        """
        모든 승리 가능한 위치 조합 반환
        
        Returns:
            승리 가능한 위치들의 목록
        """
        winning_positions = []
        
        # 가로줄
        for row in range(3):
            winning_positions.append([(row, 0), (row, 1), (row, 2)])
        
        # 세로줄
        for col in range(3):
            winning_positions.append([(0, col), (1, col), (2, col)])
        
        # 대각선
        winning_positions.append([(0, 0), (1, 1), (2, 2)])  # 왼쪽 위 -> 오른쪽 아래
        winning_positions.append([(0, 2), (1, 1), (2, 0)])  # 오른쪽 위 -> 왼쪽 아래
        
        return winning_positions
    
    def can_win_next_move(self, player: int) -> List[Tuple[int, int]]:
        """
        특정 플레이어가 다음 수로 승리할 수 있는 위치들 반환
        
        Args:
            player: 플레이어 인덱스 (0 또는 1)
            
        Returns:
            승리 가능한 위치들의 목록
        """
        player_mark = 1 if player == 0 else -1
        winning_moves = []
        
        for row, col in self.get_valid_moves():
            # 임시로 표시해보기
            self.board[row][col] = player_mark
            
            # 승리 확인
            if self._check_winner() == player:
                winning_moves.append((row, col))
            
            # 원래대로 되돌리기
            self.board[row][col] = 0
        
        return winning_moves
    
    def must_block_positions(self, player: int) -> List[Tuple[int, int]]:
        """
        특정 플레이어가 상대방을 막아야 하는 위치들 반환
        
        Args:
            player: 플레이어 인덱스 (0 또는 1)
            
        Returns:
            막아야 하는 위치들의 목록
        """
        opponent = 1 - player
        return self.can_win_next_move(opponent)
