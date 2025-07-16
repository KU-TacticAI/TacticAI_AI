import abc
import csv
import os
from typing import Any, List, Optional, Union, Dict
import numpy as np
from copy import deepcopy
import logging


class BoardGame(abc.ABC):
    def __init__(self, game_id: str, players: List[str], max_history: Optional[int] = None):
        self.game_id = game_id
        self.players = players
        self.current_player = 0
        self.board = None
        self.index_mapping = None
        self.max_history = max_history
        self.move_history: List[Dict[str, Any]] = []
        self._load_index_mapping()

    def _load_index_mapping(self):
        """각 게임의 인덱스 맵핑 파일을 로드합니다."""
        try:
            mapping_file = self._get_mapping_file_path()
            if mapping_file and os.path.exists(mapping_file):
                self.index_mapping = {}
                with open(mapping_file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        index = int(row['index'])
                        move = row['move'].strip()
                        if move:
                            self.index_mapping[index] = move
                logging.info(f"맵핑 파일 로드 완료: {len(self.index_mapping)}개 항목")
            else:
                self.index_mapping = {}
                logging.warning(f"맵핑 파일이 존재하지 않습니다: {mapping_file}")
        except Exception as e:
            logging.error(f"맵핑 파일 로드 실패: {e}")
            self.index_mapping = {}

    @abc.abstractmethod
    def _get_mapping_file_path(self) -> Optional[str]:
        """각 게임의 맵핑 파일 경로를 반환합니다."""
        pass

    @abc.abstractmethod
    def initialize(self, **kwargs) -> None:
        """보드 및 시작 설정"""
        pass

    @abc.abstractmethod
    def get_valid_moves(self) -> List[Any]:
        """가능한 수 목록"""
        pass

    @abc.abstractmethod
    def make_move(self, move: Any) -> bool:
        """
        move 적용.
        성공 시 True, 실패 시 False
        """
        pass

    @abc.abstractmethod
    def is_game_over(self) -> bool:
        """종료 조건 만족 여부"""
        pass

    @abc.abstractmethod
    def get_winner(self) -> Optional[str]:
        """승자 ID 또는 None"""
        pass

    @abc.abstractmethod
    def _convert_move_format(self, move_str: str) -> Any:
        """문자열 형태의 수를 게임에서 사용하는 형태로 변환합니다."""
        pass

    def record_history(self, move: Any) -> None:
        """현재 턴의 상태(플레이어, 수, 보드) 기록"""
        snapshot = {
            "player": self.players[self.current_player],
            "move": move,
            "board": deepcopy(self.board)
        }
        self.move_history.append(snapshot)
        if self.max_history and len(self.move_history) > self.max_history:
            self.move_history.pop(0)

    def get_history(self) -> List[Dict[str, Any]]:
        """기록 조회"""
        return self.move_history

    def clear_history(self) -> None:
        """기록 초기화"""
        self.move_history.clear()

    def select_best_move_from_probabilities(self, probabilities: Union[List[float], np.ndarray]) -> Optional[Any]:
        if not self.index_mapping:
            logging.error("인덱스 맵핑이 로드되지 않았습니다.")
            return None
        if isinstance(probabilities, list):
            probabilities = np.array(probabilities)
        valid_moves = self.get_valid_moves()
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            if idx in self.index_mapping:
                move_str = self.index_mapping[idx]
                converted_move = self._convert_move_format(move_str)
                if converted_move in valid_moves:
                    return converted_move
        # 확률이 0이어도 가능한 수가 있다면 아무거나 반환
        if valid_moves:
            logging.warning("확률이 0이지만 가능한 수가 있어 임의로 선택합니다.")
            return valid_moves[0]
        logging.warning(f"확률 배열에서 유효한 수를 찾을 수 없습니다. 유효한 수: {valid_moves}")
        return None

    def next_turn(self) -> None:
        self.current_player = (self.current_player + 1) % len(self.players)

    def play(self, move: Any) -> Dict[str, Any]:
        """
        단일 턴 실행:
        1) make_move
        2) 기록
        3) 종료 체크
        4) 턴 전환
        """
        success = self.make_move(move)
        if success:
            self.record_history(move)
        game_over = self.is_game_over()
        winner = self.get_winner() if game_over else None
        result = {
            "success": success,
            "game_over": game_over,
            "winner": winner,
            "next_player": self.players[self.current_player],
            "board": self.board,
        }
        if success and not game_over:
            self.next_turn()
            result["next_player"] = self.players[self.current_player]
        return result
