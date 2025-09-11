#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai 서버 메시지 스키마 정의

RabbitMQ를 통해 주고받는 모든 메시지의 구조를 정의합니다.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class GameType(str, Enum):
    """지원하는 게임 타입"""
    CHESS = "chess"
    OTHELLO = "othello"
    TICTACTOE = "tictactoe"
    OMOK = "omok"


class BaseMessage(BaseModel):
    """모든 메시지의 기본 클래스"""
    request_id: str
    timestamp: str


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
    # 각 게임별 확률배열 길이: 틱택토 9, 오셀로 65, 체스 4672, 오목(15x15) 225 (보드 크기에 따라 달라질 수 있음)
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

