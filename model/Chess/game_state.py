"""
AlphaZero용 체스 게임 상태 클래스

체스 규칙, 상태, 유효한 수, 게임 종료, 상태 인코딩 등을 관리합니다.
모든 특수 규칙(앙파상, 캐슬링, 프로모션, 체크메이트, 스테일메이트)을 포함합니다.
"""

import numpy as np
from typing import List, Tuple, Optional, Any, Dict
import copy

class ChessGameState:
    """AlphaZero용 체스 게임 상태 클래스"""
    
    # 체스말 상수
    EMPTY = 0
    WHITE_PAWN = 1
    WHITE_KNIGHT = 2
    WHITE_BISHOP = 3
    WHITE_ROOK = 4
    WHITE_QUEEN = 5
    WHITE_KING = 6
    BLACK_PAWN = 7
    BLACK_KNIGHT = 8
    BLACK_BISHOP = 9
    BLACK_ROOK = 10
    BLACK_QUEEN = 11
    BLACK_KING = 12
    
    def __init__(self, board: Optional[np.ndarray] = None, current_player: int = 1):
        """
        Args:
            board: 8x8 게임 보드 (None이면 초기 상태)
            current_player: 현재 플레이어 (1: 백, -1: 흑)
        """
        if board is None:
            self.board = self._get_initial_board()
        else:
            self.board = board.copy()
        
        self.current_player = current_player
        self.move_history = []
        
        # 캐슬링 권한
        self.castling_rights = {
            'white_kingside': True,
            'white_queenside': True,
            'black_kingside': True,
            'black_queenside': True
        }
        
        # 앙파상 타겟 (상대방 폰이 두 칸 이동했을 때 그 사이 위치)
        self.en_passant_target = None
        
        # 킹과 룩의 초기 위치 기록 (캐슬링 검증용)
        self.king_rook_positions = {
            'white_king': (7, 4),
            'white_kingside_rook': (7, 7),
            'white_queenside_rook': (7, 0),
            'black_king': (0, 4),
            'black_kingside_rook': (0, 7),
            'black_queenside_rook': (0, 0)
        }
        
    def _get_initial_board(self) -> np.ndarray:
        """초기 체스 보드 상태 반환"""
        board = np.zeros((8, 8), dtype=np.int8)
        
        # 백말 배치
        board[6, :] = self.WHITE_PAWN  # 폰
        board[7, 0] = self.WHITE_ROOK  # 룩
        board[7, 1] = self.WHITE_KNIGHT  # 나이트
        board[7, 2] = self.WHITE_BISHOP  # 비숍
        board[7, 3] = self.WHITE_QUEEN  # 퀸
        board[7, 4] = self.WHITE_KING  # 킹
        board[7, 5] = self.WHITE_BISHOP  # 비숍
        board[7, 6] = self.WHITE_KNIGHT  # 나이트
        board[7, 7] = self.WHITE_ROOK  # 룩
        
        # 흑말 배치
        board[1, :] = self.BLACK_PAWN  # 폰
        board[0, 0] = self.BLACK_ROOK  # 룩
        board[0, 1] = self.BLACK_KNIGHT  # 나이트
        board[0, 2] = self.BLACK_BISHOP  # 비숍
        board[0, 3] = self.BLACK_QUEEN  # 퀸
        board[0, 4] = self.BLACK_KING  # 킹
        board[0, 5] = self.BLACK_BISHOP  # 비숍
        board[0, 6] = self.BLACK_KNIGHT  # 나이트
        board[0, 7] = self.BLACK_ROOK  # 룩
        
        return board
    
    def is_white_piece(self, piece: int) -> bool:
        """백말인지 확인"""
        return 1 <= piece <= 6
    
    def is_black_piece(self, piece: int) -> bool:
        """흑말인지 확인"""
        return 7 <= piece <= 12
    
    def get_piece_color(self, piece: int) -> int:
        """말의 색상 반환 (1: 백, -1: 흑)"""
        if piece == 0:
            return 0
        return 1 if piece <= 6 else -1
    
    def get_valid_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """현재 플레이어가 할 수 있는 유효한 수 목록 반환 (체크 검증 포함)"""
        valid_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row, col]
                if piece != 0 and self.get_piece_color(piece) == self.current_player:
                    moves = self._get_piece_moves(row, col)
                    for move in moves:
                        if self._is_valid_move((row, col), move):
                            valid_moves.append(((row, col), move))
        
        return valid_moves
    
    def _get_piece_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """특정 말의 가능한 이동 목록 반환"""
        piece = self.board[row, col]
        
        if piece in [self.WHITE_PAWN, self.BLACK_PAWN]:
            return self._get_pawn_moves(row, col)
        elif piece in [self.WHITE_KNIGHT, self.BLACK_KNIGHT]:
            return self._get_knight_moves(row, col)
        elif piece in [self.WHITE_BISHOP, self.BLACK_BISHOP]:
            return self._get_bishop_moves(row, col)
        elif piece in [self.WHITE_ROOK, self.BLACK_ROOK]:
            return self._get_rook_moves(row, col)
        elif piece in [self.WHITE_QUEEN, self.BLACK_QUEEN]:
            return self._get_queen_moves(row, col)
        elif piece in [self.WHITE_KING, self.BLACK_KING]:
            return self._get_king_moves(row, col)
        
        return []
    
    def _get_pawn_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """폰의 가능한 이동 목록 (앙파상 포함)"""
        moves = []
        piece = self.board[row, col]
        is_white = self.is_white_piece(piece)
        direction = -1 if is_white else 1
        start_row = 6 if is_white else 1
        
        # 한 칸 앞으로
        new_row = row + direction
        if 0 <= new_row < 8 and self.board[new_row, col] == 0:
            moves.append((new_row, col))
            
            # 두 칸 앞으로 (시작 위치에서만)
            if row == start_row:
                new_row2 = row + 2 * direction
                if 0 <= new_row2 < 8 and self.board[new_row2, col] == 0:
                    moves.append((new_row2, col))
        
        # 대각선 공격
        for dcol in [-1, 1]:
            new_col = col + dcol
            new_row = row + direction
            if (0 <= new_row < 8 and 0 <= new_col < 8):
                target = self.board[new_row, new_col]
                if target != 0 and self.get_piece_color(target) != self.current_player:
                    moves.append((new_row, new_col))
        
        # 앙파상
        if self.en_passant_target:
            ep_row, ep_col = self.en_passant_target
            if (row + direction == ep_row and abs(col - ep_col) == 1):
                moves.append((ep_row, ep_col))
        
        return moves
    
    def _get_knight_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """나이트의 가능한 이동 목록"""
        moves = []
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]
        
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < 8 and 0 <= new_col < 8):
                target = self.board[new_row, new_col]
                if target == 0 or self.get_piece_color(target) != self.current_player:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_bishop_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """비숍의 가능한 이동 목록"""
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dr, dc in directions:
            for step in range(1, 8):
                new_row, new_col = row + step * dr, col + step * dc
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                
                target = self.board[new_row, new_col]
                if target == 0:
                    moves.append((new_row, new_col))
                elif self.get_piece_color(target) != self.current_player:
                    moves.append((new_row, new_col))
                    break
                else:
                    break
        
        return moves
    
    def _get_rook_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """룩의 가능한 이동 목록"""
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            for step in range(1, 8):
                new_row, new_col = row + step * dr, col + step * dc
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                
                target = self.board[new_row, new_col]
                if target == 0:
                    moves.append((new_row, new_col))
                elif self.get_piece_color(target) != self.current_player:
                    moves.append((new_row, new_col))
                    break
                else:
                    break
        
        return moves
    
    def _get_queen_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """퀸의 가능한 이동 목록 (룩 + 비숍)"""
        return self._get_rook_moves(row, col) + self._get_bishop_moves(row, col)
    
    def _get_king_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """킹의 가능한 이동 목록 (캐슬링 포함)"""
        moves = []
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in king_moves:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < 8 and 0 <= new_col < 8):
                target = self.board[new_row, new_col]
                if target == 0 or self.get_piece_color(target) != self.current_player:
                    moves.append((new_row, new_col))
        
        # 캐슬링
        moves.extend(self._get_castling_moves(row, col))
        
        return moves
    
    def _get_castling_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """캐슬링 가능한 이동 목록"""
        moves = []
        piece = self.board[row, col]
        
        if piece not in [self.WHITE_KING, self.BLACK_KING]:
            return moves
        
        is_white = self.is_white_piece(piece)
        player_key = 'white' if is_white else 'black'
        
        # 킹사이드 캐슬링
        if self.castling_rights[f'{player_key}_kingside']:
            if self._can_castle_kingside(is_white):
                moves.append((row, col + 2))
        
        # 퀸사이드 캐슬링
        if self.castling_rights[f'{player_key}_queenside']:
            if self._can_castle_queenside(is_white):
                moves.append((row, col - 2))
        
        return moves
    
    def _can_castle_kingside(self, is_white: bool) -> bool:
        """킹사이드 캐슬링 가능 여부"""
        row = 7 if is_white else 0
        
        # 킹과 룩이 초기 위치에 있는지 확인
        if self.board[row, 4] != (self.WHITE_KING if is_white else self.BLACK_KING):
            return False
        if self.board[row, 7] != (self.WHITE_ROOK if is_white else self.BLACK_ROOK):
            return False
        
        # 킹과 룩 사이가 비어있는지 확인
        for col in range(5, 7):
            if self.board[row, col] != 0:
                return False
        
        # 킹이 체크 상태가 아니고, 킹이 지나가는 칸이 공격받지 않는지 확인
        if self._is_in_check(self.board, 1 if is_white else -1):
            return False
        
        # 킹이 지나가는 칸이 공격받지 않는지 확인
        for col in range(4, 7):
            if self._is_square_under_attack(row, col, 1 if is_white else -1):
                return False
        
        return True
    
    def _can_castle_queenside(self, is_white: bool) -> bool:
        """퀸사이드 캐슬링 가능 여부"""
        row = 7 if is_white else 0
        
        # 킹과 룩이 초기 위치에 있는지 확인
        if self.board[row, 4] != (self.WHITE_KING if is_white else self.BLACK_KING):
            return False
        if self.board[row, 0] != (self.WHITE_ROOK if is_white else self.BLACK_ROOK):
            return False
        
        # 킹과 룩 사이가 비어있는지 확인
        for col in range(1, 4):
            if self.board[row, col] != 0:
                return False
        
        # 킹이 체크 상태가 아니고, 킹이 지나가는 칸이 공격받지 않는지 확인
        if self._is_in_check(self.board, 1 if is_white else -1):
            return False
        
        # 킹이 지나가는 칸이 공격받지 않는지 확인
        for col in range(2, 5):
            if self._is_square_under_attack(row, col, 1 if is_white else -1):
                return False
        
        return True
    
    def _is_square_under_attack(self, row: int, col: int, by_player: int) -> bool:
        """특정 칸이 상대방에게 공격받고 있는지 확인"""
        opponent = -by_player
        
        for r in range(8):
            for c in range(8):
                piece = self.board[r, c]
                if piece != 0 and self.get_piece_color(piece) == opponent:
                    moves = self._get_piece_moves_for_board(self.board, r, c)
                    if (row, col) in moves:
                        return True
        
        return False
    
    def _is_valid_move(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """이동이 유효한지 확인 (체크 검증 포함)"""
        # 임시로 이동을 실행해보고 체크 상태인지 확인
        temp_state = self.make_move((start, end))
        return not temp_state._is_in_check(temp_state.board, self.current_player)
    
    def _is_in_check(self, board: np.ndarray, player: int) -> bool:
        """플레이어가 체크 상태인지 확인"""
        # 킹 위치 찾기
        king_piece = self.WHITE_KING if player == 1 else self.BLACK_KING
        king_pos = None
        
        for row in range(8):
            for col in range(8):
                if board[row, col] == king_piece:
                    king_pos = (row, col)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return False
        
        # 상대방 말들이 킹을 공격할 수 있는지 확인
        opponent = -player
        for row in range(8):
            for col in range(8):
                piece = board[row, col]
                if piece != 0 and self.get_piece_color(piece) == opponent:
                    moves = self._get_piece_moves_for_board(board, row, col)
                    if king_pos in moves:
                        return True
        
        return False
    
    def _get_piece_moves_for_board(self, board: np.ndarray, row: int, col: int) -> List[Tuple[int, int]]:
        """특정 보드에서 말의 이동 목록 (체크 검증용)"""
        piece = board[row, col]
        
        if piece in [self.WHITE_PAWN, self.BLACK_PAWN]:
            return self._get_pawn_moves_for_board(board, row, col)
        elif piece in [self.WHITE_KNIGHT, self.BLACK_KNIGHT]:
            return self._get_knight_moves_for_board(board, row, col)
        elif piece in [self.WHITE_BISHOP, self.BLACK_BISHOP]:
            return self._get_bishop_moves_for_board(board, row, col)
        elif piece in [self.WHITE_ROOK, self.BLACK_ROOK]:
            return self._get_rook_moves_for_board(board, row, col)
        elif piece in [self.WHITE_QUEEN, self.BLACK_QUEEN]:
            return self._get_queen_moves_for_board(board, row, col)
        elif piece in [self.WHITE_KING, self.BLACK_KING]:
            return self._get_king_moves_for_board(board, row, col)
        
        return []
    
    def _get_pawn_moves_for_board(self, board: np.ndarray, row: int, col: int) -> List[Tuple[int, int]]:
        """보드에서 폰의 이동 목록 (체크 검증용)"""
        moves = []
        piece = board[row, col]
        is_white = self.is_white_piece(piece)
        direction = -1 if is_white else 1
        start_row = 6 if is_white else 1
        
        # 한 칸 앞으로
        new_row = row + direction
        if 0 <= new_row < 8 and board[new_row, col] == 0:
            moves.append((new_row, col))
            
            # 두 칸 앞으로 (시작 위치에서만)
            if row == start_row:
                new_row2 = row + 2 * direction
                if 0 <= new_row2 < 8 and board[new_row2, col] == 0:
                    moves.append((new_row2, col))
        
        # 대각선 공격
        for dcol in [-1, 1]:
            new_col = col + dcol
            new_row = row + direction
            if (0 <= new_row < 8 and 0 <= new_col < 8):
                target = board[new_row, new_col]
                if target != 0 and self.get_piece_color(target) != self.current_player:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_knight_moves_for_board(self, board: np.ndarray, row: int, col: int) -> List[Tuple[int, int]]:
        """보드에서 나이트의 이동 목록"""
        moves = []
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]
        
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < 8 and 0 <= new_col < 8):
                target = board[new_row, new_col]
                if target == 0 or self.get_piece_color(target) != self.current_player:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_bishop_moves_for_board(self, board: np.ndarray, row: int, col: int) -> List[Tuple[int, int]]:
        """보드에서 비숍의 이동 목록"""
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dr, dc in directions:
            for step in range(1, 8):
                new_row, new_col = row + step * dr, col + step * dc
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                
                target = board[new_row, new_col]
                if target == 0:
                    moves.append((new_row, new_col))
                elif self.get_piece_color(target) != self.current_player:
                    moves.append((new_row, new_col))
                    break
                else:
                    break
        
        return moves
    
    def _get_rook_moves_for_board(self, board: np.ndarray, row: int, col: int) -> List[Tuple[int, int]]:
        """보드에서 룩의 이동 목록"""
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            for step in range(1, 8):
                new_row, new_col = row + step * dr, col + step * dc
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                
                target = board[new_row, new_col]
                if target == 0:
                    moves.append((new_row, new_col))
                elif self.get_piece_color(target) != self.current_player:
                    moves.append((new_row, new_col))
                    break
                else:
                    break
        
        return moves
    
    def _get_queen_moves_for_board(self, board: np.ndarray, row: int, col: int) -> List[Tuple[int, int]]:
        """보드에서 퀸의 이동 목록"""
        return self._get_rook_moves_for_board(board, row, col) + self._get_bishop_moves_for_board(board, row, col)
    
    def _get_king_moves_for_board(self, board: np.ndarray, row: int, col: int) -> List[Tuple[int, int]]:
        """보드에서 킹의 이동 목록"""
        moves = []
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in king_moves:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < 8 and 0 <= new_col < 8):
                target = board[new_row, new_col]
                if target == 0 or self.get_piece_color(target) != self.current_player:
                    moves.append((new_row, new_col))
        
        return moves
    
    def make_move(self, move: Tuple[Tuple[int, int], Tuple[int, int]]) -> 'ChessGameState':
        """수를 실행하고 새로운 게임 상태 반환"""
        start, end = move
        start_row, start_col = start
        end_row, end_col = end
        
        new_state = ChessGameState(self.board.copy(), self.current_player)
        new_state.move_history = self.move_history.copy()
        new_state.move_history.append(move)
        new_state.castling_rights = self.castling_rights.copy()
        new_state.en_passant_target = self.en_passant_target
        new_state.king_rook_positions = self.king_rook_positions.copy()
        
        piece = new_state.board[start_row, start_col]
        
        # 캐슬링 처리
        if piece in [self.WHITE_KING, self.BLACK_KING] and abs(start_col - end_col) == 2:
            # 킹사이드 캐슬링
            if end_col > start_col:
                # 룩 이동
                rook_start_col = 7
                rook_end_col = 5
                rook_piece = new_state.board[start_row, rook_start_col]
                new_state.board[start_row, rook_end_col] = rook_piece
                new_state.board[start_row, rook_start_col] = 0
            # 퀸사이드 캐슬링
            else:
                # 룩 이동
                rook_start_col = 0
                rook_end_col = 3
                rook_piece = new_state.board[start_row, rook_start_col]
                new_state.board[start_row, rook_end_col] = rook_piece
                new_state.board[start_row, rook_start_col] = 0
        
        # 일반 이동
        new_state.board[end_row, end_col] = piece
        new_state.board[start_row, start_col] = 0
        
        # 앙파상 처리
        if piece in [self.WHITE_PAWN, self.BLACK_PAWN] and self.en_passant_target:
            ep_row, ep_col = self.en_passant_target
            if (end_row, end_col) == (ep_row, ep_col):
                # 앙파상으로 상대방 폰 제거
                captured_pawn_row = start_row
                new_state.board[captured_pawn_row, end_col] = 0
        
        # 앙파상 타겟 설정
        if piece in [self.WHITE_PAWN, self.BLACK_PAWN] and abs(start_row - end_row) == 2:
            new_state.en_passant_target = ((start_row + end_row) // 2, start_col)
        else:
            new_state.en_passant_target = None
        
        # 캐슬링 권한 업데이트
        if piece in [self.WHITE_KING, self.BLACK_KING]:
            player_key = 'white' if self.is_white_piece(piece) else 'black'
            new_state.castling_rights[f'{player_key}_kingside'] = False
            new_state.castling_rights[f'{player_key}_queenside'] = False
        
        if piece in [self.WHITE_ROOK, self.BLACK_ROOK]:
            player_key = 'white' if self.is_white_piece(piece) else 'black'
            if start_col == 7:  # 킹사이드 룩
                new_state.castling_rights[f'{player_key}_kingside'] = False
            elif start_col == 0:  # 퀸사이드 룩
                new_state.castling_rights[f'{player_key}_queenside'] = False
        
        # 폰 프로모션
        if piece in [self.WHITE_PAWN, self.BLACK_PAWN]:
            if (piece == self.WHITE_PAWN and end_row == 0) or (piece == self.BLACK_PAWN and end_row == 7):
                new_state.board[end_row, end_col] = self.WHITE_QUEEN if piece == self.WHITE_PAWN else self.BLACK_QUEEN
        
        # 플레이어 변경
        new_state.current_player = -self.current_player
        
        return new_state
    
    def is_game_over(self) -> bool:
        """게임이 종료되었는지 확인"""
        # 체크메이트 또는 스테일메이트 확인
        valid_moves = self.get_valid_moves()
        
        if len(valid_moves) == 0:
            return True
        
        # 50수 규칙 (간단한 구현)
        if len(self.move_history) >= 100:
            return True
        
        # 3회 반복 규칙 (간단한 구현)
        if len(self.move_history) >= 6:
            last_moves = self.move_history[-6:]
            if len(set(str(move) for move in last_moves)) <= 2:
                return True
        
        return False
    
    def get_winner(self) -> Optional[int]:
        """승자 반환 (1: 백승, -1: 흑승, 0: 무승부, None: 진행중)"""
        if not self.is_game_over():
            return None
        
        valid_moves = self.get_valid_moves()
        
        if len(valid_moves) == 0:
            # 체크메이트인지 스테일메이트인지 확인
            if self._is_in_check(self.board, self.current_player):
                return -self.current_player  # 체크메이트
            else:
                return 0  # 스테일메이트
        
        # 50수 규칙 또는 3회 반복 규칙으로 인한 무승부
        return 0
    
    def get_encoded_state(self) -> np.ndarray:
        """
        신경망 입력용 인코딩된 상태 반환
        (18, 8, 8) 형태의 배열 - 12기물+턴+캐슬링4+앙파상
        채널:
          0~5: 백 폰, 나이트, 비숍, 룩, 퀸, 킹
          6~11: 흑 폰, 나이트, 비숍, 룩, 퀸, 킹
          12: 턴 정보 (백=1, 흑=0, 모든 칸 동일)
          13~16: 캐슬링 권리 (백킹, 백퀸, 흑킹, 흑퀸, 모든 칸 동일)
          17: 앙파상 가능 칸만 1
        """
        encoded = np.zeros((18, 8, 8), dtype=np.float32)
        # 12채널: 백/흑 기물
        for r in range(8):
            for c in range(8):
                v = self.board[r, c]
                if v == self.WHITE_PAWN: encoded[0, r, c] = 1
                elif v == self.WHITE_KNIGHT: encoded[1, r, c] = 1
                elif v == self.WHITE_BISHOP: encoded[2, r, c] = 1
                elif v == self.WHITE_ROOK: encoded[3, r, c] = 1
                elif v == self.WHITE_QUEEN: encoded[4, r, c] = 1
                elif v == self.WHITE_KING: encoded[5, r, c] = 1
                elif v == self.BLACK_PAWN: encoded[6, r, c] = 1
                elif v == self.BLACK_KNIGHT: encoded[7, r, c] = 1
                elif v == self.BLACK_BISHOP: encoded[8, r, c] = 1
                elif v == self.BLACK_ROOK: encoded[9, r, c] = 1
                elif v == self.BLACK_QUEEN: encoded[10, r, c] = 1
                elif v == self.BLACK_KING: encoded[11, r, c] = 1
        # 턴 정보 (채널 12)
        encoded[12, :, :] = 1 if self.current_player == 1 else 0
        # 캐슬링 권리 (채널 13~16)
        encoded[13, :, :] = 1 if self.castling_rights.get('white_kingside', False) else 0
        encoded[14, :, :] = 1 if self.castling_rights.get('white_queenside', False) else 0
        encoded[15, :, :] = 1 if self.castling_rights.get('black_kingside', False) else 0
        encoded[16, :, :] = 1 if self.castling_rights.get('black_queenside', False) else 0
        # 앙파상 (채널 17)
        if self.en_passant_target is not None:
            ep_row, ep_col = self.en_passant_target
            encoded[17, ep_row, ep_col] = 1
        return encoded
    
    def get_action_index(self, move: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
        """이동을 인덱스로 변환 (4672개 가능한 이동)"""
        start, end = move
        start_row, start_col = start
        end_row, end_col = end
        
        # 간단한 인덱싱 (실제로는 더 정교한 인코딩 필요)
        start_idx = start_row * 8 + start_col
        end_idx = end_row * 8 + end_col
        return start_idx * 64 + end_idx
    
    def get_action_from_index(self, action_index: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """인덱스를 이동으로 변환"""
        start_idx = action_index // 64
        end_idx = action_index % 64
        
        start_row = start_idx // 8
        start_col = start_idx % 8
        end_row = end_idx // 8
        end_col = end_idx % 8
        
        return ((start_row, start_col), (end_row, end_col))
    
    def copy(self) -> 'ChessGameState':
        """게임 상태 복사"""
        new_state = ChessGameState(self.board.copy(), self.current_player)
        new_state.move_history = self.move_history.copy()
        new_state.castling_rights = self.castling_rights.copy()
        new_state.en_passant_target = self.en_passant_target
        new_state.king_rook_positions = self.king_rook_positions.copy()
        return new_state
    
    def __str__(self) -> str:
        """보드 출력"""
        symbols = {
            0: '.', self.WHITE_PAWN: 'P', self.WHITE_KNIGHT: 'N', self.WHITE_BISHOP: 'B',
            self.WHITE_ROOK: 'R', self.WHITE_QUEEN: 'Q', self.WHITE_KING: 'K',
            self.BLACK_PAWN: 'p', self.BLACK_KNIGHT: 'n', self.BLACK_BISHOP: 'b',
            self.BLACK_ROOK: 'r', self.BLACK_QUEEN: 'q', self.BLACK_KING: 'k'
        }
        
        result = "  a b c d e f g h\n"
        for row in range(8):
            result += f"{8-row} "
            for col in range(8):
                result += f"{symbols[self.board[row, col]]} "
            result += f"{8-row}\n"
        result += "  a b c d e f g h"
        return result 