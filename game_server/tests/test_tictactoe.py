#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
틱택토 게임 테스트 코드
"""

import pytest
import sys
import os

# 상위 디렉토리를 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Game.BoardGame.TicTacToe import TicTacToe


class TestTicTacToe:
    """틱택토 게임 테스트 클래스"""
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        self.tictactoe = TicTacToe("test_game", ["플레이어X", "플레이어O"])
        self.tictactoe.initialize()
    
    def test_initialization(self):
        """보드 초기화 테스트"""
        # 보드 크기 확인
        assert len(self.tictactoe.board) == 3
        assert len(self.tictactoe.board[0]) == 3
        
        # 모든 칸이 비어있는지 확인
        for row in range(3):
            for col in range(3):
                assert self.tictactoe.board[row][col] == 0
        
        # 초기 상태 확인
        assert self.tictactoe.current_player == 0  # X가 먼저
    
    def test_유효한_이동_감지(self):
        """유효한 이동 감지 테스트"""
        valid_moves = self.tictactoe.get_valid_moves()
        
        # 초기 상태에서 모든 9개 위치가 유효해야 함
        assert len(valid_moves) == 9
        
        expected_moves = [(i, j) for i in range(3) for j in range(3)]
        for move in expected_moves:
            assert move in valid_moves
    
    def test_이동_문자열_변환(self):
        """이동 문자열 변환 테스트"""
        # 정상적인 변환
        assert self.tictactoe._convert_move_format("a1") == (0, 0)
        assert self.tictactoe._convert_move_format("b2") == (1, 1)
        assert self.tictactoe._convert_move_format("c3") == (2, 2)
        
        # 대소문자 처리
        assert self.tictactoe._convert_move_format("A1") == (0, 0)
        assert self.tictactoe._convert_move_format("C3") == (2, 2)
        
        # 잘못된 형식
        assert self.tictactoe._convert_move_format("d1") is None  # 범위 초과
        assert self.tictactoe._convert_move_format("a4") is None  # 범위 초과
        assert self.tictactoe._convert_move_format("") is None
        assert self.tictactoe._convert_move_format("invalid") is None
    
    def test_기본_이동_실행(self):
        """기본 이동 실행 테스트"""
        # a1에 X 표시
        move = (0, 0)  # a1
        result = self.tictactoe.make_move(move)
        
        assert result is True
        assert self.tictactoe.board[0][0] == 1  # X가 표시됨
    
    def test_잘못된_이동_거부(self):
        """잘못된 이동이 거부되는지 테스트"""
        # 먼저 a1에 X 표시
        self.tictactoe.make_move((0, 0))
        
        # 이미 표시된 위치에 다시 표시 시도
        assert not self.tictactoe.make_move((0, 0))  # a1에 이미 X
        
        # 범위를 벗어난 위치
        assert not self.tictactoe.make_move((-1, 0))
        assert not self.tictactoe.make_move((3, 0))
        assert not self.tictactoe.make_move((0, -1))
        assert not self.tictactoe.make_move((0, 3))
        
        # 잘못된 형식
        assert not self.tictactoe.make_move("invalid")
        assert not self.tictactoe.make_move([0, 0])
    
    def test_턴_변경(self):
        """턴 변경 테스트"""
        assert self.tictactoe.current_player == 0  # 초기: X
        
        # X 이동
        self.tictactoe.make_move((0, 0))  # a1
        self.tictactoe.next_turn()
        assert self.tictactoe.current_player == 1  # O 턴
        
        # O 이동
        self.tictactoe.make_move((1, 1))  # b2
        self.tictactoe.next_turn()
        assert self.tictactoe.current_player == 0  # 다시 X 턴
    
    def test_가로줄_승리(self):
        """가로줄 승리 테스트"""
        # X가 첫 번째 가로줄 완성
        moves = [(0, 0), (0, 1), (0, 2)]  # a1, a2, a3
        for i, move in enumerate(moves):
            self.tictactoe.current_player = 0  # X 차례로 고정
            self.tictactoe.make_move(move)
        
        assert self.tictactoe._check_winner() == 0  # X 승리
        assert self.tictactoe.is_game_over()
        assert self.tictactoe.get_winner() == "플레이어X"
    
    def test_세로줄_승리(self):
        """세로줄 승리 테스트"""
        # O가 첫 번째 세로줄 완성
        moves = [(0, 0), (1, 0), (2, 0)]  # a1, b1, c1
        for i, move in enumerate(moves):
            self.tictactoe.current_player = 1  # O 차례로 고정
            self.tictactoe.make_move(move)
        
        assert self.tictactoe._check_winner() == 1  # O 승리
        assert self.tictactoe.is_game_over()
        assert self.tictactoe.get_winner() == "플레이어O"
    
    def test_대각선_승리(self):
        """대각선 승리 테스트"""
        # X가 주 대각선 완성
        moves = [(0, 0), (1, 1), (2, 2)]  # a1, b2, c3
        for i, move in enumerate(moves):
            self.tictactoe.current_player = 0  # X 차례로 고정
            self.tictactoe.make_move(move)
        
        assert self.tictactoe._check_winner() == 0  # X 승리
        assert self.tictactoe.is_game_over()
        assert self.tictactoe.get_winner() == "플레이어X"
    
    def test_역대각선_승리(self):
        """역대각선 승리 테스트"""
        # O가 역 대각선 완성
        moves = [(0, 2), (1, 1), (2, 0)]  # a3, b2, c1
        for i, move in enumerate(moves):
            self.tictactoe.current_player = 1  # O 차례로 고정
            self.tictactoe.make_move(move)        
        assert self.tictactoe._check_winner() == 1  # O 승리
        assert self.tictactoe.is_game_over()
        assert self.tictactoe.get_winner() == "플레이어O"

    def test_무승부(self):
        """무승부 테스트"""
        # 무승부 상황 설정
        # X O X
        # X O O  
        # O X O
        moves_and_players = [
            ((0, 0), 0),  # X at a1
            ((0, 1), 1),  # O at a2
            ((0, 2), 0),  # X at a3
            ((1, 0), 0),  # X at b1
            ((1, 1), 1),  # O at b2
            ((1, 2), 1),  # O at b3
            ((2, 0), 1),  # O at c1
            ((2, 1), 0),  # X at c2
            ((2, 2), 1),  # O at c3
        ]
        
        for move, player in moves_and_players:
            self.tictactoe.current_player = player
            self.tictactoe.make_move(move)
        
        assert self.tictactoe._check_winner() is None  # 승자 없음
        assert self.tictactoe.is_game_over()  # 보드 가득 참
        assert self.tictactoe.get_winner() is None  # 무승부
    
    def test_연속_게임_플레이(self):
        """연속적인 게임 진행 테스트"""
        moves = [
            (0, 0),  # X at a1
            (1, 1),  # O at b2
            (0, 1),  # X at a2
            (2, 2),  # O at c3
        ]
        
        for i, move in enumerate(moves):
            result = self.tictactoe.play(move)
            assert result["success"] is True
            
            if not result["game_over"]:
                # 턴이 바뀌었는지 확인
                expected_player = "플레이어O" if i % 2 == 0 else "플레이어X"
                assert result["next_player"] == expected_player
    
    def test_보드_출력(self):
        """보드 출력 기능 테스트"""
        board_str = self.tictactoe.print_board()
        
        # 기본적인 구조 확인
        assert "a b c" in board_str
        assert "1" in board_str and "3" in board_str
        
        # 빈 칸 심볼 확인
        assert "." in board_str
        
        # 현재 턴 정보 확인
        assert "현재 턴:" in board_str
    
    def test_게임_상태_정보(self):
        """게임 상태 정보 반환 테스트"""
        status = self.tictactoe.get_game_status()
        
        assert status["current_player"] == "플레이어X"
        assert status["is_x_turn"] is True
        assert status["valid_moves_count"] == 9  # 초기 상태에서 9개 이동 가능
        assert len(status["valid_moves"]) == 9
        assert status["board_full"] is False
        assert status["game_over"] is False
        assert status["winner"] is None
        assert status["turn_number"] == 1  # 첫 번째 턴
    
    def test_빈_위치_확인(self):
        """빈 위치 확인 테스트"""
        # 초기 상태에서 모든 위치가 비어있어야 함
        for row in range(3):
            for col in range(3):
                assert self.tictactoe.is_position_empty(row, col)
        
        # a1에 표시 후
        self.tictactoe.make_move((0, 0))
        assert not self.tictactoe.is_position_empty(0, 0)
        assert self.tictactoe.is_position_empty(0, 1)
    
    def test_위치_값_확인(self):
        """위치 값 확인 테스트"""
        # 초기 상태
        assert self.tictactoe.get_position_value(0, 0) == 0
        
        # X 표시 후
        self.tictactoe.make_move((0, 0))
        assert self.tictactoe.get_position_value(0, 0) == 1
        
        # O 표시 후
        self.tictactoe.next_turn()
        self.tictactoe.make_move((1, 1))
        assert self.tictactoe.get_position_value(1, 1) == -1
        
        # 범위 초과
        assert self.tictactoe.get_position_value(-1, 0) == 0
        assert self.tictactoe.get_position_value(3, 0) == 0
    
    def test_표시_개수_계산(self):
        """표시 개수 계산 테스트"""
        # 초기 상태
        counts = self.tictactoe.count_marks()
        assert counts["플레이어X"] == 0
        assert counts["플레이어O"] == 0
        
        # 몇 수 진행 후
        self.tictactoe.make_move((0, 0))  # X
        self.tictactoe.next_turn()
        self.tictactoe.make_move((1, 1))  # O
        
        counts = self.tictactoe.count_marks()
        assert counts["플레이어X"] == 1
        assert counts["플레이어O"] == 1
    
    def test_승리_위치_확인(self):
        """승리 위치 확인 테스트"""
        winning_positions = self.tictactoe.get_winning_positions()
        
        # 8개의 승리 조합이 있어야 함 (3가로 + 3세로 + 2대각선)
        assert len(winning_positions) == 8
        
        # 첫 번째 가로줄 확인
        assert [(0, 0), (0, 1), (0, 2)] in winning_positions
        
        # 첫 번째 세로줄 확인
        assert [(0, 0), (1, 0), (2, 0)] in winning_positions
        
        # 주 대각선 확인
        assert [(0, 0), (1, 1), (2, 2)] in winning_positions
        
        # 역 대각선 확인
        assert [(0, 2), (1, 1), (2, 0)] in winning_positions
    
    def test_다음_수_승리_가능(self):
        """다음 수로 승리 가능한 위치 테스트"""
        # X가 두 개를 놓은 상황
        self.tictactoe.board[0][0] = 1  # X at a1
        self.tictactoe.board[0][1] = 1  # X at a2
        # a3에 X를 놓으면 승리
        
        winning_moves = self.tictactoe.can_win_next_move(0)  # X 플레이어
        assert (0, 2) in winning_moves  # a3 위치
    
    def test_상대방_막기(self):
        """상대방을 막아야 하는 위치 테스트"""
        # O가 두 개를 놓은 상황
        self.tictactoe.board[1][0] = -1  # O at b1
        self.tictactoe.board[1][1] = -1  # O at b2
        # b3를 막아야 함
        
        block_moves = self.tictactoe.must_block_positions(0)  # X 플레이어 입장에서
        assert (1, 2) in block_moves  # b3 위치
    
    def test_AI_확률_배열_처리(self):
        """AI 확률 배열로 최적 수 선택 테스트"""
        # 가능한 이동들 중에서 선택
        probabilities = [0.0] * 9  # 9개 위치
        
        # a1 위치에 높은 확률 부여 (인덱스 0)
        probabilities[0] = 0.9
        
        best_move = self.tictactoe.select_best_move_from_probabilities(probabilities)
        
        # 유효한 이동이 선택되었는지 확인
        if best_move:
            assert best_move in self.tictactoe.get_valid_moves()
    
    def test_보드_상태_복사(self):
        """보드 상태 복사 테스트"""
        original_board = self.tictactoe.get_board_state()
        
        # 이동 실행
        self.tictactoe.make_move((0, 0))  # a1
        
        # 원본 보드가 변경되지 않았는지 확인
        assert original_board[0][0] == 0  # 원본에서는 여전히 빈 칸
        
        # 현재 보드 상태 확인
        current_board = self.tictactoe.get_board_state()
        assert current_board[0][0] == 1  # 현재는 X가 표시됨
    
    def test_히스토리_기록(self):
        """이동 히스토리 기록 테스트"""
        move = (0, 0)  # a1
        self.tictactoe.play(move)
        
        history = self.tictactoe.get_history()
        assert len(history) == 1
        assert history[0]["player"] == "플레이어X"
        assert history[0]["move"] == move
        assert history[0]["board"] is not None


if __name__ == "__main__":
    # 개별 테스트 실행 예시
    test_tictactoe = TestTicTacToe()
    test_tictactoe.setup_method()
    
    print("=== 틱택토 게임 테스트 시작 ===")
    
    try:
        test_tictactoe.test_초기화()
        print("✓ 보드 초기화 테스트 통과")
        
        test_tictactoe.test_유효한_이동_감지()
        print("✓ 유효한 이동 감지 테스트 통과")
        
        test_tictactoe.test_이동_문자열_변환()
        print("✓ 이동 문자열 변환 테스트 통과")
        
        test_tictactoe.test_기본_이동_실행()
        print("✓ 기본 이동 실행 테스트 통과")
        
        test_tictactoe.setup_method()  # 새로운 게임으로 리셋
        test_tictactoe.test_잘못된_이동_거부()
        print("✓ 잘못된 이동 거부 테스트 통과")
        
        test_tictactoe.setup_method()  # 새로운 게임으로 리셋
        test_tictactoe.test_가로줄_승리()
        print("✓ 가로줄 승리 테스트 통과")
        
        test_tictactoe.setup_method()  # 새로운 게임으로 리셋
        test_tictactoe.test_대각선_승리()
        print("✓ 대각선 승리 테스트 통과")
        
        test_tictactoe.setup_method()  # 새로운 게임으로 리셋
        test_tictactoe.test_무승부()
        print("✓ 무승부 테스트 통과")
        
        print("\n=== 모든 기본 테스트 통과! ===")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
