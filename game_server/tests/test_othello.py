#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
오셀로 게임 테스트 코드
"""

import pytest
import sys
import os

# 상위 디렉토리를 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Game.BoardGame.Othello import Othello


class TestOthello:
    """오셀로 게임 테스트 클래스"""
    
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        self.othello = Othello("test_game", ["흑돌_플레이어", "백돌_플레이어"])
        self.othello.initialize()
    
    def test_초기화(self):
        """보드 초기화 테스트"""
        # 보드 크기 확인
        assert len(self.othello.board) == 8
        assert len(self.othello.board[0]) == 8
        
        # 초기 돌 배치 확인
        assert self.othello.board[3][3] == -1  # d4: 백돌
        assert self.othello.board[3][4] == 1   # e4: 흑돌
        assert self.othello.board[4][3] == 1   # d5: 흑돌
        assert self.othello.board[4][4] == -1  # e5: 백돌
        
        # 나머지 칸들은 비어있어야 함
        empty_count = 0
        for row in range(8):
            for col in range(8):
                if self.othello.board[row][col] == 0:
                    empty_count += 1
        assert empty_count == 60  # 64 - 4 = 60개 빈 칸
        
        # 초기 상태 확인
        assert self.othello.current_player == 0  # 흑돌이 먼저
        assert self.othello.consecutive_passes == 0
    
    def test_유효한_이동_감지(self):
        """초기 상태에서 유효한 이동 감지 테스트"""
        valid_moves = self.othello.get_valid_moves()
        
        # 초기 상태에서 흑돌이 놓을 수 있는 위치들
        expected_moves = [(2, 3), (3, 2), (4, 5), (5, 4)]  # d3, c4, f5, e6
        
        assert len(valid_moves) == 4
        for move in expected_moves:
            assert move in valid_moves
    
    def test_이동_문자열_변환(self):
        """이동 문자열 변환 테스트"""
        # 정상적인 변환
        assert self.othello._convert_move_format("a1") == (0, 0)
        assert self.othello._convert_move_format("h8") == (7, 7)
        assert self.othello._convert_move_format("d4") == (3, 3)
        assert self.othello._convert_move_format("pass") == "pass"
        
        # 대소문자 처리
        assert self.othello._convert_move_format("A1") == (0, 0)
        assert self.othello._convert_move_format("H8") == (7, 7)
        
        # 잘못된 형식
        assert self.othello._convert_move_format("i1") is None  # 범위 초과
        assert self.othello._convert_move_format("a9") is None  # 범위 초과
        assert self.othello._convert_move_format("") is None
        assert self.othello._convert_move_format("invalid") is None
    
    def test_기본_이동_실행(self):
        """기본 이동 실행 테스트"""
        # d3에 흑돌 놓기
        move = (2, 3)  # d3
        result = self.othello.make_move(move)
        
        assert result is True
        assert self.othello.board[2][3] == 1  # 흑돌이 놓임
        assert self.othello.board[3][3] == 1  # d4의 백돌이 흑돌로 뒤집힘
        assert self.othello.consecutive_passes == 0
    
    def test_돌_뒤집기(self):
        """돌 뒤집기 로직 테스트"""
        # 초기 상태에서 d3에 흑돌 놓기
        self.othello.make_move((2, 3))  # d3
        
        # 뒤집힌 결과 확인
        assert self.othello.board[2][3] == 1   # d3: 새로 놓은 흑돌
        assert self.othello.board[3][3] == 1   # d4: 뒤집힌 흑돌
        assert self.othello.board[3][4] == 1   # e4: 원래 흑돌
        assert self.othello.board[4][3] == 1   # d5: 원래 흑돌
        assert self.othello.board[4][4] == -1  # e5: 원래 백돌 (변화 없음)
    
    def test_잘못된_이동_거부(self):
        """잘못된 이동이 거부되는지 테스트"""
        # 이미 돌이 있는 위치
        assert not self.othello.make_move((3, 3))  # d4에 이미 백돌
        
        # 뒤집을 수 없는 위치
        assert not self.othello.make_move((0, 0))  # a1
        assert not self.othello.make_move((7, 7))  # h8
        
        # 범위를 벗어난 위치
        assert not self.othello.make_move((-1, 0))
        assert not self.othello.make_move((8, 0))
        assert not self.othello.make_move((0, -1))
        assert not self.othello.make_move((0, 8))
        
        # 잘못된 형식
        assert not self.othello.make_move("invalid")
        assert not self.othello.make_move([2, 3])
    
    def test_턴_변경(self):
        """턴 변경 테스트"""
        assert self.othello.current_player == 0  # 초기: 흑돌
        
        # 흑돌 이동
        self.othello.make_move((2, 3))  # d3
        self.othello.next_turn()
        assert self.othello.current_player == 1  # 백돌 턴
        
        # 백돌 이동
        valid_moves = self.othello.get_valid_moves()
        if valid_moves:
            self.othello.make_move(valid_moves[0])
            self.othello.next_turn()
            assert self.othello.current_player == 0  # 다시 흑돌 턴
    
    def test_패스_기능(self):
        """패스 기능 테스트"""
        # 패스 실행
        result = self.othello.make_move("pass")
        assert result is True
        assert self.othello.consecutive_passes == 1
        
        # 연속 패스
        self.othello.next_turn()
        result = self.othello.make_move("pass")
        assert result is True
        assert self.othello.consecutive_passes == 2
    
    def test_점수_계산(self):
        """점수 계산 테스트"""
        scores = self.othello.get_score()
        
        # 초기 상태: 각자 2개씩
        assert scores["흑돌_플레이어"] == 2
        assert scores["백돌_플레이어"] == 2
        
        # 이동 후 점수 변화
        self.othello.make_move((2, 3))  # d3에 흑돌
        scores = self.othello.get_score()
        assert scores["흑돌_플레이어"] == 4  # 2 + 1(새로 놓음) + 1(뒤집힘)
        assert scores["백돌_플레이어"] == 1   # 2 - 1(뒤집힘)
    
    def test_게임_종료_조건(self):
        """게임 종료 조건 테스트"""
        # 초기 상태에서는 게임이 종료되지 않음
        assert not self.othello.is_game_over()
        
        # 연속 패스로 게임 종료
        self.othello.consecutive_passes = 2
        assert self.othello.is_game_over()
    
    def test_보드_가득_참_조건(self):
        """보드가 가득 찬 경우 게임 종료 테스트"""
        # 보드를 가득 채움
        for row in range(8):
            for col in range(8):
                self.othello.board[row][col] = 1 if (row + col) % 2 == 0 else -1
        
        assert self.othello._is_board_full()
        assert self.othello.is_game_over()
    
    def test_승자_결정(self):
        """승자 결정 테스트"""
        # 게임이 끝나지 않은 상태
        assert self.othello.get_winner() is None
        
        # 흑돌이 더 많은 경우
        self.othello.board = [[1 for _ in range(8)] for _ in range(8)]
        self.othello.board[0][0] = -1  # 백돌 1개
        self.othello.consecutive_passes = 2  # 게임 종료 조건
        
        assert self.othello.get_winner() == "흑돌_플레이어"
        
        # 백돌이 더 많은 경우
        self.othello.board = [[-1 for _ in range(8)] for _ in range(8)]
        self.othello.board[0][0] = 1  # 흑돌 1개
        
        assert self.othello.get_winner() == "백돌_플레이어"
        
        # 무승부인 경우
        for row in range(8):
            for col in range(8):
                self.othello.board[row][col] = 1 if row < 4 else -1
        
        assert self.othello.get_winner() is None
    
    def test_AI_확률_배열_처리(self):
        """AI 확률 배열로 최적 수 선택 테스트"""
        # 가능한 이동들 중에서 선택
        probabilities = [0.0] * 65  # 64개 위치 + 패스
        
        # d3 위치에 높은 확률 부여 (인덱스 19 = 2*8 + 3)
        probabilities[19] = 0.9
        
        best_move = self.othello.select_best_move_from_probabilities(probabilities)
        
        # 유효한 이동이 선택되었는지 확인
        if best_move:
            assert best_move in self.othello.get_valid_moves()
    
    def test_연속_게임_플레이(self):
        """연속적인 게임 진행 테스트"""
        moves = [
            (2, 3),  # d3 (흑돌)
            (2, 2),  # c3 (백돌)
            (2, 4),  # e3 (흑돌)
        ]
        
        for i, move in enumerate(moves):
            if move in self.othello.get_valid_moves():
                result = self.othello.play(move)
                assert result["success"] is True
                
                if not result["game_over"]:
                    # 턴이 바뀌었는지 확인
                    expected_player = "백돌_플레이어" if i % 2 == 0 else "흑돌_플레이어"
                    assert result["next_player"] == expected_player
    
    def test_보드_출력(self):
        """보드 출력 기능 테스트"""
        board_str = self.othello.print_board()
        
        # 기본적인 구조 확인
        assert "a b c d e f g h" in board_str
        assert "1" in board_str and "8" in board_str
        
        # 돌 심볼 확인
        assert "●" in board_str  # 흑돌
        assert "○" in board_str  # 백돌
        assert "." in board_str  # 빈 칸
        
        # 점수 정보 확인
        assert "점수:" in board_str
        assert "현재 턴:" in board_str
    
    def test_게임_상태_정보(self):
        """게임 상태 정보 반환 테스트"""
        status = self.othello.get_game_status()
        
        assert status["current_player"] == "흑돌_플레이어"
        assert status["is_black_turn"] is True
        assert status["valid_moves_count"] == 4  # 초기 상태에서 4개 이동 가능
        assert len(status["valid_moves"]) == 4
        assert status["consecutive_passes"] == 0
        assert status["board_full"] is False
        assert status["game_over"] is False
        assert status["winner"] is None
        
        # 점수 확인
        assert status["scores"]["흑돌_플레이어"] == 2
        assert status["scores"]["백돌_플레이어"] == 2
    
    def test_패스_가능_조건(self):
        """패스 가능 조건 테스트"""
        # 초기 상태에서는 유효한 수가 있으므로 패스 불가능
        assert not self.othello.can_pass()
        
        # 보드를 특수한 상태로 만들어 유효한 수가 없게 함
        # (실제로는 복잡한 상황이지만 여기서는 단순화)
        self.othello.board = [[1 for _ in range(8)] for _ in range(8)]
        self.othello.board[0][0] = 0  # 한 칸만 비움
        # 하지만 이 칸에는 유효한 수가 없도록 주변을 모두 같은 색으로
        
        # 유효한 수가 없는지 확인하고 패스 가능한지 테스트
        valid_moves = self.othello.get_valid_moves()
        if len(valid_moves) == 0:
            assert self.othello.can_pass()
    
    def test_보드_상태_복사(self):
        """보드 상태 복사 테스트"""
        original_board = self.othello.get_board_state()
        
        # 이동 실행
        self.othello.make_move((2, 3))  # d3
        
        # 원본 보드가 변경되지 않았는지 확인
        assert original_board[2][3] == 0  # 원본에서는 여전히 빈 칸
        assert original_board[3][3] == -1  # 원본에서는 여전히 백돌
        
        # 현재 보드 상태 확인
        current_board = self.othello.get_board_state()
        assert current_board[2][3] == 1   # 현재는 흑돌
        assert current_board[3][3] == 1   # 현재는 뒤집힌 흑돌
    
    def test_히스토리_기록(self):
        """이동 히스토리 기록 테스트"""
        move = (2, 3)  # d3
        self.othello.play(move)
        
        history = self.othello.get_history()
        assert len(history) == 1
        assert history[0]["player"] == "흑돌_플레이어"
        assert history[0]["move"] == move
        assert history[0]["board"] is not None
    
    def test_복잡한_게임_시나리오(self):
        """복잡한 게임 시나리오 테스트"""
        # 여러 수를 진행하여 복잡한 상황 만들기
        moves = [
            (2, 3),  # d3 (흑돌)
            (2, 2),  # c3 (백돌)
            (2, 4),  # e3 (흑돌)
            (1, 3),  # d2 (백돌)
        ]
        
        for move in moves:
            if move in self.othello.get_valid_moves():
                result = self.othello.play(move)
                assert result["success"] is True
                
                # 각 이동 후 게임 상태 검증
                assert not result["game_over"]  # 아직 게임이 끝나지 않음
                
                # 점수가 올바르게 계산되는지 확인
                scores = self.othello.get_score()
                total_stones = scores["흑돌_플레이어"] + scores["백돌_플레이어"]
                assert total_stones <= 64  # 총 돌 수는 64개를 넘지 않음


if __name__ == "__main__":
    # 개별 테스트 실행 예시
    test_othello = TestOthello()
    test_othello.setup_method()
    
    print("=== 오셀로 게임 테스트 시작 ===")
    
    try:
        test_othello.test_초기화()
        print("✓ 보드 초기화 테스트 통과")
        
        test_othello.test_유효한_이동_감지()
        print("✓ 유효한 이동 감지 테스트 통과")
        
        test_othello.test_이동_문자열_변환()
        print("✓ 이동 문자열 변환 테스트 통과")
        
        test_othello.test_기본_이동_실행()
        print("✓ 기본 이동 실행 테스트 통과")
        
        test_othello.setup_method()  # 새로운 게임으로 리셋
        test_othello.test_돌_뒤집기()
        print("✓ 돌 뒤집기 테스트 통과")
        
        test_othello.setup_method()  # 새로운 게임으로 리셋
        test_othello.test_잘못된_이동_거부()
        print("✓ 잘못된 이동 거부 테스트 통과")
        
        test_othello.setup_method()  # 새로운 게임으로 리셋
        test_othello.test_점수_계산()
        print("✓ 점수 계산 테스트 통과")
        
        test_othello.setup_method()  # 새로운 게임으로 리셋
        test_othello.test_연속_게임_플레이()
        print("✓ 연속 게임 플레이 테스트 통과")
        
        print("\n=== 모든 기본 테스트 통과! ===")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
