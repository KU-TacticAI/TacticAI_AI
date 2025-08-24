import os
from typing import List, Optional, Tuple, Any, Dict
from .BoardGame import BoardGame
import logging


class Chess(BoardGame):
    """
    완전한 체스 게임 구현
    - 모든 기물의 이동 규칙
    - 특수 룰: 캐슬링, 앙파상, 프로모션
    - 체크, 체크메이트, 스테일메이트 처리
    - 숫자로 표현된 보드 (0: 빈칸, 양수: 백색, 음수: 흑색)
    """
    
    # 기물 상수 정의
    EMPTY = 0
    WHITE_PAWN = 1
    WHITE_KNIGHT = 2
    WHITE_BISHOP = 3
    WHITE_ROOK = 4
    WHITE_QUEEN = 5
    WHITE_KING = 6
    BLACK_PAWN = -1
    BLACK_KNIGHT = -2
    BLACK_BISHOP = -3
    BLACK_ROOK = -4
    BLACK_QUEEN = -5
    BLACK_KING = -6
    
    # 기물 이름 매핑
    PIECE_NAMES = {
        1: "백색 폰", -1: "흑색 폰",
        2: "백색 나이트", -2: "흑색 나이트",
        3: "백색 비숍", -3: "흑색 비숍",
        4: "백색 룩", -4: "흑색 룩",
        5: "백색 퀸", -5: "흑색 퀸",
        6: "백색 킹", -6: "흑색 킹"
    }
    
    def __init__(self, game_id: str, players: List[str], max_history: Optional[int] = None):
        super().__init__(game_id, players, max_history)
        self.board = None
        self.white_king_moved = False  # 백색 킹 이동 여부
        self.black_king_moved = False  # 흑색 킹 이동 여부
        self.white_rook_moved = [False, False]  # [a1 룩, h1 룩]
        self.black_rook_moved = [False, False]  # [a8 룩, h8 룩]
        self.en_passant_target = None  # 앙파상 가능한 위치
        self.halfmove_clock = 0  # 50수 규칙용
        self.fullmove_number = 1  # 총 수 카운트
        
    def _get_mapping_file_path(self) -> Optional[str]:
        """체스 맵핑 파일 경로 반환"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "Chess.csv")
    
    def initialize(self, **kwargs) -> None:
        """체스 보드 초기화"""
        # 8x8 보드 생성
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        
        # 백색 기물 배치 (1행, 2행)
        self.board[0] = [self.WHITE_ROOK, self.WHITE_KNIGHT, self.WHITE_BISHOP, self.WHITE_QUEEN,
                        self.WHITE_KING, self.WHITE_BISHOP, self.WHITE_KNIGHT, self.WHITE_ROOK]
        self.board[1] = [self.WHITE_PAWN] * 8
        
        # 흑색 기물 배치 (7행, 8행)
        self.board[6] = [self.BLACK_PAWN] * 8
        self.board[7] = [self.BLACK_ROOK, self.BLACK_KNIGHT, self.BLACK_BISHOP, self.BLACK_QUEEN,
                        self.BLACK_KING, self.BLACK_BISHOP, self.BLACK_KNIGHT, self.BLACK_ROOK]
        
        # 상태 초기화
        self.white_king_moved = False
        self.black_king_moved = False
        self.white_rook_moved = [False, False]
        self.black_rook_moved = [False, False]
        self.en_passant_target = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        logging.info("체스 보드가 초기화되었습니다")
    
    def _convert_move_format(self, move_str: str) -> Optional[Tuple[int, int, int, int, Optional[int]]]:
        """
        문자열 형태의 수를 좌표로 변환
        예: "e2e4" -> (1, 4, 3, 4, None)
        프로모션: "e7e8q" -> (6, 4, 7, 4, 5)
        """
        if not move_str or len(move_str) < 4:
            return None
            
        try:
            # 시작 위치
            from_file = ord(move_str[0]) - ord('a')  # a=0, b=1, ...
            from_rank = int(move_str[1]) - 1  # 1=0, 2=1, ...
            
            # 목표 위치
            to_file = ord(move_str[2]) - ord('a')
            to_rank = int(move_str[3]) - 1
            
            # 프로모션 확인
            promotion = None
            if len(move_str) > 4:
                promotion_char = move_str[4].lower()
                promotion_map = {'q': 5, 'r': 4, 'b': 3, 'n': 2}
                if promotion_char in promotion_map:
                    promotion = promotion_map[promotion_char]
            
            return (from_rank, from_file, to_rank, to_file, promotion)
        except (ValueError, IndexError):
            return None
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """유효한 보드 위치인지 확인"""
        return 0 <= row < 8 and 0 <= col < 8
    
    def _is_white_piece(self, piece: int) -> bool:
        """백색 기물인지 확인"""
        return piece > 0
    
    def _is_black_piece(self, piece: int) -> bool:
        """흑색 기물인지 확인"""
        return piece < 0
    
    def _is_enemy_piece(self, piece: int, is_white_turn: bool) -> bool:
        """적군 기물인지 확인"""
        return (is_white_turn and self._is_black_piece(piece)) or (not is_white_turn and self._is_white_piece(piece))
    
    def _is_friendly_piece(self, piece: int, is_white_turn: bool) -> bool:
        """아군 기물인지 확인"""
        return (is_white_turn and self._is_white_piece(piece)) or (not is_white_turn and self._is_black_piece(piece))
    
    def _get_piece_moves_without_castling(self, row: int, col: int) -> List[Tuple[int, int]]:
        """캐슬링을 제외한 특정 기물의 가능한 이동 위치 반환 (무한 재귀 방지용)"""
        piece = self.board[row][col]
        if piece == 0:
            return []
        
        piece_type = abs(piece)
        is_white = piece > 0
        moves = []
        
        if piece_type == 1:  # 폰
            moves = self._get_pawn_moves(row, col, is_white)
        elif piece_type == 2:  # 나이트
            moves = self._get_knight_moves(row, col, is_white)
        elif piece_type == 3:  # 비숍
            moves = self._get_bishop_moves(row, col, is_white)
        elif piece_type == 4:  # 룩
            moves = self._get_rook_moves(row, col, is_white)
        elif piece_type == 5:  # 퀸
            moves = self._get_queen_moves(row, col, is_white)
        elif piece_type == 6:  # 킹 (캐슬링 제외)
            moves = self._get_king_moves_basic(row, col, is_white)
        
        return moves
    
    def _get_king_moves_basic(self, row: int, col: int, is_white: bool) -> List[Tuple[int, int]]:
        """킹의 기본 이동 가능한 위치 (캐슬링 제외)"""
        moves = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position(new_row, new_col):
                target_piece = self.board[new_row][new_col]
                if target_piece == 0 or self._is_enemy_piece(target_piece, is_white):
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_piece_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """특정 기물의 가능한 이동 위치 반환"""
        piece = self.board[row][col]
        if piece == 0:
            return []
        
        piece_type = abs(piece)
        is_white = piece > 0
        moves = []
        
        if piece_type == 1:  # 폰
            moves = self._get_pawn_moves(row, col, is_white)
        elif piece_type == 2:  # 나이트
            moves = self._get_knight_moves(row, col, is_white)
        elif piece_type == 3:  # 비숍
            moves = self._get_bishop_moves(row, col, is_white)
        elif piece_type == 4:  # 룩
            moves = self._get_rook_moves(row, col, is_white)
        elif piece_type == 5:  # 퀸
            moves = self._get_queen_moves(row, col, is_white)
        elif piece_type == 6:  # 킹
            moves = self._get_king_moves(row, col, is_white)
        
        return moves
    
    def _get_pawn_moves(self, row: int, col: int, is_white: bool) -> List[Tuple[int, int]]:
        """폰의 이동 가능한 위치"""
        moves = []
        direction = 1 if is_white else -1
        start_row = 1 if is_white else 6
        
        # 앞으로 한 칸
        new_row = row + direction
        if self._is_valid_position(new_row, col) and self.board[new_row][col] == 0:
            moves.append((new_row, col))
            
            # 시작 위치에서 두 칸
            if row == start_row:
                new_row = row + 2 * direction
                if self._is_valid_position(new_row, col) and self.board[new_row][col] == 0:
                    moves.append((new_row, col))
        
        # 대각선 공격
        for dc in [-1, 1]:
            new_row = row + direction
            new_col = col + dc
            if self._is_valid_position(new_row, new_col):
                target_piece = self.board[new_row][new_col]
                if target_piece != 0 and self._is_enemy_piece(target_piece, is_white):
                    moves.append((new_row, new_col))
                # 앙파상
                elif self.en_passant_target and (new_row, new_col) == self.en_passant_target:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_knight_moves(self, row: int, col: int, is_white: bool) -> List[Tuple[int, int]]:
        """나이트의 이동 가능한 위치"""
        moves = []
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position(new_row, new_col):
                target_piece = self.board[new_row][new_col]
                if target_piece == 0 or self._is_enemy_piece(target_piece, is_white):
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_bishop_moves(self, row: int, col: int, is_white: bool) -> List[Tuple[int, int]]:
        """비숍의 이동 가능한 위치"""
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * dr, col + i * dc
                if not self._is_valid_position(new_row, new_col):
                    break
                
                target_piece = self.board[new_row][new_col]
                if target_piece == 0:
                    moves.append((new_row, new_col))
                elif self._is_enemy_piece(target_piece, is_white):
                    moves.append((new_row, new_col))
                    break
                else:  # 아군 기물
                    break
        
        return moves
    
    def _get_rook_moves(self, row: int, col: int, is_white: bool) -> List[Tuple[int, int]]:
        """룩의 이동 가능한 위치"""
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * dr, col + i * dc
                if not self._is_valid_position(new_row, new_col):
                    break
                
                target_piece = self.board[new_row][new_col]
                if target_piece == 0:
                    moves.append((new_row, new_col))
                elif self._is_enemy_piece(target_piece, is_white):
                    moves.append((new_row, new_col))
                    break
                else:  # 아군 기물
                    break
        
        return moves
    
    def _get_queen_moves(self, row: int, col: int, is_white: bool) -> List[Tuple[int, int]]:
        """퀸의 이동 가능한 위치 (룩 + 비숍)"""
        moves = []
        moves.extend(self._get_rook_moves(row, col, is_white))
        moves.extend(self._get_bishop_moves(row, col, is_white))
        return moves
    
    def _get_king_moves(self, row: int, col: int, is_white: bool) -> List[Tuple[int, int]]:
        """킹의 이동 가능한 위치"""
        moves = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position(new_row, new_col):
                target_piece = self.board[new_row][new_col]
                if target_piece == 0 or self._is_enemy_piece(target_piece, is_white):
                    moves.append((new_row, new_col))
        
        # 캐슬링 추가
        moves.extend(self._get_castling_moves(is_white))
        
        return moves
    
    def _get_castling_moves(self, is_white: bool) -> List[Tuple[int, int]]:
        """캐슬링 가능한 이동"""
        moves = []
        
        if is_white:
            if not self.white_king_moved and not self._is_in_check(True):
                # 킹사이드 캐슬링
                if (not self.white_rook_moved[1] and 
                    self.board[0][5] == 0 and self.board[0][6] == 0 and
                    not self._is_square_attacked(0, 5, False) and
                    not self._is_square_attacked(0, 6, False)):
                    moves.append((0, 6))
                
                # 퀸사이드 캐슬링
                if (not self.white_rook_moved[0] and 
                    self.board[0][1] == 0 and self.board[0][2] == 0 and self.board[0][3] == 0 and
                    not self._is_square_attacked(0, 2, False) and
                    not self._is_square_attacked(0, 3, False)):
                    moves.append((0, 2))
        else:
            if not self.black_king_moved and not self._is_in_check(False):
                # 킹사이드 캐슬링
                if (not self.black_rook_moved[1] and 
                    self.board[7][5] == 0 and self.board[7][6] == 0 and
                    not self._is_square_attacked(7, 5, True) and
                    not self._is_square_attacked(7, 6, True)):
                    moves.append((7, 6))
                
                # 퀸사이드 캐슬링
                if (not self.black_rook_moved[0] and 
                    self.board[7][1] == 0 and self.board[7][2] == 0 and self.board[7][3] == 0 and
                    not self._is_square_attacked(7, 2, True) and
                    not self._is_square_attacked(7, 3, True)):                    moves.append((7, 2))
        
        return moves
    
    def _is_square_attacked(self, row: int, col: int, by_white: bool) -> bool:
        """특정 칸이 공격받고 있는지 확인"""
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece == 0:
                    continue
                
                piece_is_white = self._is_white_piece(piece)
                if piece_is_white == by_white:
                    # 이 기물이 target 위치를 공격할 수 있는지 확인
                    if abs(piece) == 1:  # 폰 특별 처리
                        if self._pawn_attacks_square(r, c, piece_is_white, row, col):
                            return True
                    else:
                        # 캐슬링 체크를 제외한 기물 이동만 확인 (무한 재귀 방지)
                        possible_moves = self._get_piece_moves_without_castling(r, c)
                        if (row, col) in possible_moves:
                            return True
        return False
    
    def _pawn_attacks_square(self, pawn_row: int, pawn_col: int, is_white: bool, target_row: int, target_col: int) -> bool:
        """폰이 특정 칸을 공격하는지 확인"""
        direction = 1 if is_white else -1
        attack_row = pawn_row + direction
        
        return (attack_row == target_row and 
                abs(pawn_col - target_col) == 1)
    
    def _is_in_check(self, is_white_king: bool) -> bool:
        """킹이 체크 상태인지 확인"""
        king_piece = self.WHITE_KING if is_white_king else self.BLACK_KING
        king_pos = None
        
        # 킹 위치 찾기
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == king_piece:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return False
        
        return self._is_square_attacked(king_pos[0], king_pos[1], not is_white_king)
    
    def _would_be_in_check_after_move(self, from_row: int, from_col: int, to_row: int, to_col: int, is_white: bool) -> bool:
        """이동 후 체크 상태가 되는지 확인"""
        # 임시로 이동 실행
        original_piece = self.board[to_row][to_col]
        moving_piece = self.board[from_row][from_col]
        
        self.board[to_row][to_col] = moving_piece
        self.board[from_row][from_col] = 0
        
        # 체크 상태 확인
        in_check = self._is_in_check(is_white)
        
        # 원상복구
        self.board[from_row][from_col] = moving_piece
        self.board[to_row][to_col] = original_piece
        
        return in_check
    
    def get_valid_moves(self) -> List[Tuple[int, int, int, int, Optional[int]]]:
        """현재 플레이어의 유효한 모든 수"""
        is_white_turn = self.current_player == 0
        valid_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece == 0:
                    continue
                
                piece_is_white = self._is_white_piece(piece)
                if piece_is_white != is_white_turn:
                    continue
                
                possible_moves = self._get_piece_moves(row, col)
                
                for to_row, to_col in possible_moves:
                    # 체크 상태가 되는 수는 제외
                    if not self._would_be_in_check_after_move(row, col, to_row, to_col, is_white_turn):
                        # 폰 프로모션 처리
                        if abs(piece) == 1 and (to_row == 0 or to_row == 7):
                            # 프로모션 가능한 모든 기물
                            for promotion in [2, 3, 4, 5]:  # 나이트, 비숍, 룩, 퀸
                                valid_moves.append((row, col, to_row, to_col, promotion))
                        else:
                            valid_moves.append((row, col, to_row, to_col, None))
        
        return valid_moves
    
    def make_move(self, move: Tuple[int, int, int, int, Optional[int]]) -> bool:
        """수를 실행"""
        if not move or len(move) < 4:
            return False
        
        from_row, from_col, to_row, to_col = move[:4]
        promotion = move[4] if len(move) > 4 else None
        
        # 유효한 위치인지 확인
        if not (self._is_valid_position(from_row, from_col) and self._is_valid_position(to_row, to_col)):
            return False
        
        piece = self.board[from_row][from_col]
        if piece == 0:
            return False
        
        is_white_turn = self.current_player == 0
        piece_is_white = self._is_white_piece(piece)
        
        # 현재 플레이어의 기물인지 확인
        if piece_is_white != is_white_turn:
            return False
        
        # 유효한 수인지 확인
        valid_moves = self.get_valid_moves()
        if move not in valid_moves:
            return False
        
        # 특수 이동 처리
        captured_piece = self.board[to_row][to_col]
        
        # 앙파상 처리
        if abs(piece) == 1 and self.en_passant_target == (to_row, to_col):
            # 앙파상으로 잡힌 폰 제거
            captured_row = to_row - (1 if is_white_turn else -1)
            captured_piece = self.board[captured_row][to_col]
            self.board[captured_row][to_col] = 0
        
        # 캐슬링 처리
        if abs(piece) == 6 and abs(to_col - from_col) == 2:
            # 룩도 함께 이동
            if to_col == 6:  # 킹사이드
                rook_from_col, rook_to_col = 7, 5
            else:  # 퀸사이드
                rook_from_col, rook_to_col = 0, 3
            
            rook = self.board[from_row][rook_from_col]
            self.board[from_row][rook_to_col] = rook
            self.board[from_row][rook_from_col] = 0
        
        # 기본 이동
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = 0
        
        # 프로모션 처리
        if promotion and abs(piece) == 1 and (to_row == 0 or to_row == 7):
            promoted_piece = promotion if is_white_turn else -promotion
            self.board[to_row][to_col] = promoted_piece
        
        # 상태 업데이트
        self._update_game_state(piece, from_row, from_col, to_row, to_col, captured_piece)
        
        return True
    
    def _update_game_state(self, piece: int, from_row: int, from_col: int, to_row: int, to_col: int, captured_piece: int):
        """게임 상태 업데이트"""
        is_white = self._is_white_piece(piece)
        
        # 킹 이동 기록
        if abs(piece) == 6:
            if is_white:
                self.white_king_moved = True
            else:
                self.black_king_moved = True
        
        # 룩 이동 기록
        if abs(piece) == 4:
            if is_white:
                if from_col == 0:
                    self.white_rook_moved[0] = True
                elif from_col == 7:
                    self.white_rook_moved[1] = True
            else:
                if from_col == 0:
                    self.black_rook_moved[0] = True
                elif from_col == 7:
                    self.black_rook_moved[1] = True
        
        # 앙파상 타겟 설정
        self.en_passant_target = None
        if abs(piece) == 1 and abs(to_row - from_row) == 2:
            self.en_passant_target = ((from_row + to_row) // 2, from_col)
        
        # 50수 규칙 카운터
        if abs(piece) == 1 or captured_piece != 0:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        
        # 총 수 카운터
        if not is_white:
            self.fullmove_number += 1
    
    def is_game_over(self) -> bool:
        """게임 종료 여부 확인"""
        return (self._is_checkmate() or self._is_stalemate() or 
                self._is_insufficient_material() or self.halfmove_clock >= 50)
    
    def _is_checkmate(self) -> bool:
        """체크메이트 상태인지 확인"""
        is_white_turn = self.current_player == 0
        if not self._is_in_check(is_white_turn):
            return False
        
        return len(self.get_valid_moves()) == 0
    
    def _is_stalemate(self) -> bool:
        """스테일메이트 상태인지 확인"""
        is_white_turn = self.current_player == 0
        if self._is_in_check(is_white_turn):
            return False
        
        return len(self.get_valid_moves()) == 0
    
    def _is_insufficient_material(self) -> bool:
        """기물 부족으로 인한 무승부 확인"""
        pieces = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != 0:
                    pieces.append(abs(piece))
        
        pieces.sort()
        
        # 킹 vs 킹
        if pieces == [6, 6]:
            return True
        
        # 킹 + 비숍 vs 킹 또는 킹 + 나이트 vs 킹
        if pieces == [2, 6, 6] or pieces == [3, 6, 6]:
            return True
        
        # 킹 + 비숍 vs 킹 + 비숍 (같은 색 비숍)
        if pieces == [3, 3, 6, 6]:
            bishops = []
            for row in range(8):
                for col in range(8):
                    if abs(self.board[row][col]) == 3:
                        bishops.append((row + col) % 2)
            if len(set(bishops)) == 1:  # 같은 색 비숍
                return True
        
        return False
    
    def get_winner(self) -> Optional[str]:
        """승자 결정"""
        if not self.is_game_over():
            return None
        
        if self._is_checkmate():
            # 체크메이트 당한 플레이어의 상대가 승리
            return self.players[1 - self.current_player]
        
        # 무승부
        return None
    
    def get_board_state(self) -> List[List[int]]:
        """현재 보드 상태 반환"""
        return [row[:] for row in self.board]
    
    def get_game_status(self) -> Dict[str, Any]:
        """게임 상태 정보 반환"""
        is_white_turn = self.current_player == 0
        return {
            "current_player": self.players[self.current_player],
            "is_white_turn": is_white_turn,
            "in_check": self._is_in_check(is_white_turn),
            "valid_moves_count": len(self.get_valid_moves()),
            "halfmove_clock": self.halfmove_clock,
            "fullmove_number": self.fullmove_number,
            "en_passant_target": self.en_passant_target,
            "castling_rights": {
                "white_king_side": not self.white_king_moved and not self.white_rook_moved[1],
                "white_queen_side": not self.white_king_moved and not self.white_rook_moved[0],
                "black_king_side": not self.black_king_moved and not self.black_rook_moved[1],
                "black_queen_side": not self.black_king_moved and not self.black_rook_moved[0]
            }
        }
    
    def print_board(self) -> str:
        """보드를 문자열로 출력"""
        piece_symbols = {
            0: ".",
            1: "P", -1: "p",
            2: "N", -2: "n", 
            3: "B", -3: "b",
            4: "R", -4: "r",
            5: "Q", -5: "q",
            6: "K", -6: "k"
        }
        
        result = []
        result.append("  a b c d e f g h")
        for row in range(7, -1, -1):
            line = f"{row + 1} "
            for col in range(8):
                piece = self.board[row][col]
                line += piece_symbols[piece] + " "
            result.append(line)
        
        return "\n".join(result)
