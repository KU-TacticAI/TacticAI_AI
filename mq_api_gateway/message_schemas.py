#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime

# ============================================================================
# RabbitMQ 메시지 스키마 (기존)
# ============================================================================

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
    """게임 시작 요청 메시지 (RabbitMQ용)"""
    game_id: str
    game_type: GameType
    model_ids: List[str]
    model_urls: List[str]
    players: List[str] = []
    
    class Config:
        protected_namespaces = ()


class GameProgress(BaseMessage):
    """게임 진행 상황 메시지 (RabbitMQ용)"""
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


# ============================================================================
# FastAPI 전용 스키마 (새로 추가)
# ============================================================================

class GameRequestCreate(BaseModel):
    """게임 요청 생성 (FastAPI 입력용)"""
    game_id: str = Field(..., description="게임 고유 ID")
    game_type: GameType = Field(..., description="게임 타입")
    ai_model_ids: List[str] = Field(..., description="AI 모델 ID 목록")
    ai_model_urls: List[str] = Field(..., description="AI 모델 URL 목록")
    player_names: List[str] = Field(default=[], description="플레이어 이름 목록")
    
    class Config:
        json_schema_extra = {
            "example": {
                "game_id": "game-123",
                "game_type": "tictactoe",
                "ai_model_ids": ["model-1", "model-2"],
                "ai_model_urls": ["http://example.com/model1", "http://example.com/model2"],
                "player_names": ["Player1", "Player2"]
            }
        }


class GameRequestResponse(BaseModel):
    """게임 요청 응답 (FastAPI 출력용)"""
    status: str = Field(..., description="요청 상태")
    message: str = Field(..., description="응답 메시지")
    game_id: str = Field(..., description="게임 ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="응답 시간")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "게임 요청이 성공적으로 전송되었습니다.",
                "game_id": "game-123",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class GameProgressResponse(BaseModel):
    """게임 진행상황 응답 (FastAPI 출력용)"""
    game_id: str = Field(..., description="게임 ID")
    game_type: GameType = Field(..., description="게임 타입")
    board_state: List[int] = Field(..., description="게임판 상태")
    turn_number: int = Field(..., description="현재 턴 번호")
    current_turn: int = Field(..., description="현재 플레이어 턴")
    player_names: List[str] = Field(..., description="플레이어 이름 목록")
    last_move: Optional[str] = Field(None, description="마지막 수")
    is_finished: bool = Field(False, description="게임 종료 여부")
    winner: Optional[str] = Field(None, description="승자")
    is_success: bool = Field(False, description="요청 성공 여부")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="응답 시간")
    
    class Config:
        json_schema_extra = {
            "example": {
                "game_id": "game-123",
                "game_type": "tictactoe",
                "board_state": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "turn_number": 1,
                "current_turn": 0,
                "player_names": ["Player1", "Player2"],
                "last_move": "A1",
                "is_finished": False,
                "winner": None,
                "is_success": True,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """에러 응답 (FastAPI 출력용)"""
    status: str = Field("error", description="에러 상태")
    message: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="에러 발생 시간")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "게임 요청 전송 실패",
                "detail": "RabbitMQ 연결 오류",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
