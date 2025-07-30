#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
게임 서버 메시지 스키마 정의

RabbitMQ를 통해 주고받는 모든 메시지의 구조를 정의합니다.

주요 변경사항:
- InferenceRequest, ModelLoadRequest에 game_server_id 필드 추가
- AI 서버가 응답을 보낼 게임 서버를 식별할 수 있도록 함
- ErrorMessage 스키마 제거 (로깅으로 대체)
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class GameType(str, Enum):
    """지원하는 게임 타입"""
    CHESS = "chess"
    OTHELLO = "othello"
    TICTACTOE = "tictactoe"


class BaseMessage(BaseModel):
    """모든 메시지의 기본 클래스"""
    request_id: str
    timestamp: str


class GameRequest(BaseMessage):
    """게임 시작 요청 메시지"""
    game_id: str
    game_type: GameType
    model_ids: List[str]
    model_urls: List[str]
    players: List[str] = []


class InferenceRequest(BaseMessage):
    """AI 추론 요청 메시지"""
    game_id: str
    model_id: str
    board_state: List[int]
    current_turn: int
    turn_number: int
    game_type: GameType
    game_server_id: str  # 응답을 받을 게임 서버 ID


class InferenceResponse(BaseMessage):
    """AI 추론 응답 메시지"""
    game_id: str
    model_id: str
    success: bool
    probabilities: List[float] = []
    message: str = ""
    ai_server_id: str = ""
    response_time_ms: int = 0  


class ModelLoadRequest(BaseMessage):
    """모델 로드 요청 메시지"""
    game_id: str
    model_id: str
    model_url: str
    game_type: GameType
    game_server_id: str  # 응답을 받을 게임 서버 ID


class ModelLoadResponse(BaseMessage):
    """모델 로드 응답 메시지"""
    game_id: str
    model_id: str
    success: bool
    message: Optional[str] = None  # 로드 결과 메시지
    ai_server_id: Optional[str] = None  # 응답한 AI 서버 ID


class GameProgress(BaseMessage):
    """게임 진행 상황 메시지"""
    game_id: str
    game_type: GameType
    board_state: List[int]
    turn_number: int
    current_turn: int
    players: List[str]
    last_move: Optional[str] = None
    is_finished: bool = False
    winner: Optional[str] = None
    is_success: bool = False