#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
게임 서버 메인 모듈 -  메트릭 자동 보고
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

from config import Config
from rabbitmq_client import RabbitMQClient
from game_manager import GameManager
from message_schemas import GameType
import db

# ============================================================
# FastAPI 매트릭 API 정의
# ============================================================

app = FastAPI(title="Game Server Metrics API", version="1.0.0")

# GameManager 인스턴스를 저장할 전역 변수
_game_manager: Optional[GameManager] = None


class GameStats(BaseModel):
    """게임 타입별 통계"""
    type: str
    activeRooms: int
    waitingUsers: int
    status: str  # HEALTHY, BUSY, CRITICAL


class MetricsResponse(BaseModel):
    """매트릭 응답 스키마"""
    timestamp: str
    totalActiveMatches: int
    games: List[GameStats]


def get_status_by_active_rooms(active_rooms: int) -> str:
    """활성 방 수에 따라 상태 판단
    
    기준:
    - HEALTHY: activeRooms < 100
    - BUSY: 100 <= activeRooms < 300
    - CRITICAL: activeRooms >= 300
    """
    if active_rooms < 100:
        return "HEALTHY"
    elif active_rooms < 300:
        return "BUSY"
    else:
        return "CRITICAL"


@app.get("/metric/realtime/game-stats")
async def get_game_stats():
    """게임별 실시간 매트릭 조회 (Prometheus 형식)"""
    global _game_manager
    
    output = []
    
    # 지원하는 모든 게임 타입 (기본값 설정)
    all_game_types = ["CHESS", "GOMOKU", "OTHELLO", "TICTACTOE"]
    game_stats: Dict[str, Dict] = {
        gt: {"activeRooms": 0, "waitingUsers": 0, "status": "HEALTHY"}
        for gt in all_game_types
    }
    
    total_active_matches = 0
    
    if _game_manager is not None:
        total_active_matches = len(_game_manager.games)
        
        # 게임 타입별 활성 방 수 집계
        for game_id, meta in _game_manager.game_meta.items():
            game_type = meta.get("game_type")
            if game_type is not None:
                if hasattr(game_type, 'name'):
                    type_name = game_type.name
                else:
                    type_name = str(game_type)
                
                if type_name in game_stats:
                    game_stats[type_name]["activeRooms"] += 1
                    # 상태 판단
                    rooms = game_stats[type_name]["activeRooms"]
                    if rooms >= 300:
                        game_stats[type_name]["status"] = "CRITICAL"
                    elif rooms >= 100:
                        game_stats[type_name]["status"] = "BUSY"
    
    # ===== Prometheus 메트릭 출력 =====
    
    # total_active_matches (Gauge)
    output.append("# HELP game_server_total_active_matches Total number of active game matches.")
    output.append("# TYPE game_server_total_active_matches gauge")
    output.append(f"game_server_total_active_matches {total_active_matches}")
    
    # 게임 타입별 activeRooms (Gauge)
    output.append("")
    output.append("# HELP game_server_active_rooms Number of active rooms by game type.")
    output.append("# TYPE game_server_active_rooms gauge")
    for game_type, stats in game_stats.items():
        output.append(f'game_server_active_rooms{{game_type="{game_type}"}} {stats["activeRooms"]}')
    
    # 게임 타입별 waitingUsers (Gauge)
    output.append("")
    output.append("# HELP game_server_waiting_users Number of waiting users by game type.")
    output.append("# TYPE game_server_waiting_users gauge")
    for game_type, stats in game_stats.items():
        output.append(f'game_server_waiting_users{{game_type="{game_type}"}} {stats["waitingUsers"]}')
    
    # 게임 타입별 status (Gauge)
    output.append("")
    output.append("# HELP game_server_status Game server status by type (0=HEALTHY, 1=BUSY, 2=CRITICAL).")
    output.append("# TYPE game_server_status gauge")
    status_map = {"HEALTHY": 0, "BUSY": 1, "CRITICAL": 2}
    for game_type, stats in game_stats.items():
        status_value = status_map.get(stats["status"], 0)
        output.append(f'game_server_status{{game_type="{game_type}",status="{stats["status"]}"}} {status_value}')
    
    response_content = "\n".join(output) + "\n"
    
    return Response(
        content=response_content,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "ok"}


def run_metrics_server(host: str, port: int):
    """FastAPI 서버를 별도 쓰레드에서 실행"""
    uvicorn.run(app, host=host, port=port, log_level="info")


# ============================================================
# 메인 서버 로직
# ============================================================

async def main():
    global _game_manager
    
    # 1. 설정 및 로깅
    config = Config()
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
    logger = logging.getLogger(__name__)
    logger.info("게임 서버 시작")

    # 2. 매트릭 API 서버 시작 (별도 쓰레드)
    metrics_host = getattr(config, 'METRICS_API_HOST', '0.0.0.0')
    metrics_port = getattr(config, 'METRICS_API_PORT', 8002)
    metrics_thread = threading.Thread(
        target=run_metrics_server,
        args=(metrics_host, metrics_port),
        daemon=True
    )
    metrics_thread.start()
    logger.info(f"매트릭 API 서버 시작: http://{metrics_host}:{metrics_port}")

    # 3. DB 연결 먼저 (무한 재시도)
    logger.info("DB 연결 시도 (MongoDB + MySQL)")
    while True:
        try:
            mongodb_ok, mysql_ok = db.init_db_connections()
            if mongodb_ok:
                logger.info("MongoDB 연결 성공")
            else:
                logger.warning("MongoDB 연결 실패")
            if mysql_ok:
                logger.info("MySQL 연결 성공")
            else:
                logger.warning("MySQL 연결 실패")
            # proceed even if one DB is down; ensure game_manager decides availability
            break
        except Exception as e:
            logger.exception(f"DB 초기화 중 오류: {e}")
        logger.info("DB 연결 실패. 5초 후 재시도합니다.")
        await asyncio.sleep(5)

    # 4. RabbitMQ 연결 (무한 재시도)
    rabbitmq = RabbitMQClient(config, config.SERVER_ID)
    while True:
        connected = await rabbitmq.connect()
        if connected:
            break
        logger.error("RabbitMQ 연결 실패. 5초 후 재시도")
        await asyncio.sleep(5)

    await rabbitmq.setup_topology()

    # 5. GameManager 생성 및 전역 변수에 저장 (매트릭 API에서 사용)
    manager = GameManager(rabbitmq)
    _game_manager = manager

    # 6. 콜백 등록
    callbacks = {
        "game_request": manager.handle_game_request,
        "inference_response": manager.handle_inference_response,
        "model_load_response": manager.handle_model_load_response,
    }
    await rabbitmq.start_consuming(callbacks)

    # 7. graceful shutdown 처리
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("서버 종료 신호 수신. 연결 해제 중...")
        await rabbitmq.disconnect()
        logger.info("서버 정상 종료.")


if __name__ == "__main__":
    asyncio.run(main())