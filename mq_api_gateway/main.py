#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway 메인 모듈 - Redis 메트릭 조회 + Prometheus 통합
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
import uuid
import logging
import json
import redis
import requests
from datetime import datetime
from contextlib import asynccontextmanager

from message_schemas import (
  GameRequest,
  GameRequestCreate, GameRequestResponse, GameProgressResponse,
  ErrorResponse
)
from config import Config
from rabbitmq_client import get_rabbitmq_client, close_rabbitmq_client
from gateway_metrics import (
  setup_metrics_endpoint,
  metrics_middleware,
  wrap_rabbitmq_client_with_metrics,
  update_rabbitmq_connection_status,
  record_game_session_completed,
  get_realtime_gateway_metrics
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
)
logger = logging.getLogger("Gateway")

# Redis 클라이언트
redis_client = None

# Admin 토큰 검증
security = HTTPBearer()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin_secret_token_12345")

# 메트릭이 적용된 RabbitMQ 클라이언트
_metrics_client = None


def get_metrics_rabbitmq_client():
  """메트릭이 적용된 RabbitMQ 클라이언트 반환"""
  global _metrics_client
  if _metrics_client is None:
    original_client = get_rabbitmq_client()
    _metrics_client = wrap_rabbitmq_client_with_metrics(original_client)
  return _metrics_client


def verify_admin_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)):
  """Admin 토큰 검증"""
  if credentials.credentials != ADMIN_TOKEN:
    raise HTTPException(status_code=401, detail="Invalid admin token")
  return credentials.credentials


def init_redis():
  """Redis 초기화"""
  global redis_client
  redis_host = os.getenv("REDIS_HOST", "redis")
  redis_port = int(os.getenv("REDIS_PORT", 6379))

  try:
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    redis_client.ping()
    logger.info(f"Redis connected: {redis_host}:{redis_port}")
  except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    logger.warning("Metrics endpoints will return empty data")
    redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
  """앱 시작/종료 시 실행"""
  # 시작
  logger.info("MQ API Gateway 시작...")
  init_redis()

  # RabbitMQ 연결 상태 초기화
  logger.info("RabbitMQ 연결 시도 중...")
  try:
    client = get_rabbitmq_client()
    is_connected = client.is_connected()
    update_rabbitmq_connection_status(is_connected)
    if is_connected:
      logger.info("RabbitMQ 연결 성공")
    else:
      logger.warning("RabbitMQ 연결 실패 (is_connected=False)")
  except Exception as e:
    logger.error(f"RabbitMQ 연결 상태 확인 실패: {e}", exc_info=True)
    update_rabbitmq_connection_status(False)

  yield

  # 종료
  logger.info("MQ API Gateway 종료...")
  close_rabbitmq_client()


# FastAPI 앱 생성
app = FastAPI(
    title="MQ API Gateway",
    description="게임 서버와 RabbitMQ 간의 API Gateway",
    lifespan=lifespan
)

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus 메트릭 미들웨어
app.middleware("http")(metrics_middleware)

# Prometheus 메트릭 엔드포인트 추가
setup_metrics_endpoint(app)


# ============================================================================
# 헬스 체크
# ============================================================================

@app.get("/health")
def health_check():
  """헬스 체크"""
  rabbitmq_connected = False
  redis_connected = False
  rabbitmq_error = None

  try:
    client = get_rabbitmq_client()
    rabbitmq_connected = client.is_connected()
    if not rabbitmq_connected:
      logger.warning("RabbitMQ client exists but is_connected() returned False")
  except Exception as e:
    rabbitmq_error = str(e)
    logger.error(f"RabbitMQ health check failed: {e}", exc_info=True)

  try:
    if redis_client:
      redis_client.ping()
      redis_connected = True
  except Exception as e:
    logger.error(f"Redis health check failed: {e}")

  response = {
    "status": "ok",
    "timestamp": datetime.utcnow().isoformat(),
    "rabbitmq_connected": rabbitmq_connected,
    "redis_connected": redis_connected
  }

  if rabbitmq_error:
    response["rabbitmq_error"] = rabbitmq_error

  return response


# ============================================================================
# Redis 기반 메트릭 엔드포인트
# ============================================================================

@app.get("/metric/realtime")
def get_gateway_realtime_metrics(token: str = Depends(verify_admin_token)):
  """
  게이트웨이 자체 실시간 메트릭 조회
  - 실시간 사용자 수 (활성 연결)
  - 최근 10초 요청 수
  - CPU 사용률
  - 메모리 사용률
  - RabbitMQ 연결 상태
  - 업타임
  """
  try:
    metrics = get_realtime_gateway_metrics()
    return {
      "timestamp": datetime.utcnow().isoformat(),
      "activeUsers": metrics.get("activeUsers", 0),
      "recentRequests": metrics.get("recentRequests", 0),
      "cpuUsagePercent": metrics.get("cpuUsagePercent", 0.0),
      "memoryUsagePercent": metrics.get("memoryUsagePercent", 0.0),
      "rabbitmqConnected": metrics.get("rabbitmqConnected", False),
      "uptimeSeconds": metrics.get("uptimeSeconds", 0)
    }
  except Exception as e:
    logger.error(f"Failed to get gateway realtime metrics: {e}")
    raise HTTPException(status_code=500, detail=str(e))


@app.get("/metric/realtime/ai-performance")
def get_ai_performance_metrics(token: str = Depends(verify_admin_token)):
  """
  AI 서버 성능 메트릭 조회 (Redis)
  """
  try:
    if not redis_client:
      return {
        "timestamp": datetime.utcnow().isoformat(),
        "averageInferenceTime": 0.0,
        "pods": [],
        "message": "Redis not available"
      }

    # Redis에서 모든 AI 서버 메트릭 조회
    keys = redis_client.keys("metrics:ai:*")

    all_pods = []
    total_time = 0

    for key in keys:
      try:
        data_json = redis_client.get(key)
        if data_json:
          data = json.loads(data_json)
          all_pods.append(data)
          total_time += data.get("inferenceTime", 0)
      except Exception as e:
        logger.error(f"Failed to parse AI metrics from {key}: {e}")

    avg_time = total_time / len(all_pods) if all_pods else 0

    return {
      "timestamp": datetime.utcnow().isoformat(),
      "averageInferenceTime": round(avg_time, 2),
      "pods": all_pods
    }

  except Exception as e:
    logger.error(f"Failed to get AI metrics: {e}")
    raise HTTPException(status_code=500, detail=str(e))


@app.get("/metric/realtime/game-stats")
def get_game_stats_metrics(token: str = Depends(verify_admin_token)):
  """
  게임별 통계 메트릭 조회 (Redis)
  """
  try:
    if not redis_client:
      return {
        "timestamp": datetime.utcnow().isoformat(),
        "totalActiveMatches": 0,
        "games": [],
        "message": "Redis not available"
      }

    # Redis에서 모든 게임 서버 메트릭 조회
    keys = redis_client.keys("metrics:game:*")

    # 게임 타입별 집계
    aggregated = {
      "CHESS": {"activeRooms": 0, "waitingUsers": 0, "status": "HEALTHY"},
      "GOMOKU": {"activeRooms": 0, "waitingUsers": 0, "status": "HEALTHY"},
      "OTHELLO": {"activeRooms": 0, "waitingUsers": 0, "status": "HEALTHY"},
      "TICTACTOE": {"activeRooms": 0, "waitingUsers": 0, "status": "HEALTHY"}
    }

    for key in keys:
      try:
        data_json = redis_client.get(key)
        if data_json:
          server_metrics = json.loads(data_json)
          game_stats = server_metrics.get("game_stats", {})

          for game_type, stats in game_stats.items():
            if game_type in aggregated:
              aggregated[game_type]["activeRooms"] += stats.get("activeRooms",
                                                                0)
              aggregated[game_type]["waitingUsers"] += stats.get("waitingUsers",
                                                                 0)

              # 상태는 가장 높은 부하 기준
              if stats.get("status") == "OVERLOADED":
                aggregated[game_type]["status"] = "OVERLOADED"
              elif stats.get("status") == "BUSY" and aggregated[game_type][
                "status"] != "OVERLOADED":
                aggregated[game_type]["status"] = "BUSY"
      except Exception as e:
        logger.error(f"Failed to parse game metrics from {key}: {e}")

    # 전체 활성 매치 수
    total_matches = sum(stats["activeRooms"] for stats in aggregated.values())

    # 리스트 형태로 변환
    games_list = [
      {"type": game_type, **stats}
      for game_type, stats in aggregated.items()
    ]

    return {
      "timestamp": datetime.utcnow().isoformat(),
      "totalActiveMatches": total_matches,
      "games": games_list
    }

  except Exception as e:
    logger.error(f"Failed to get game stats: {e}")
    raise HTTPException(status_code=500, detail=str(e))


@app.get("/metric/realtime/queue-status")
def get_queue_status_metrics(token: str = Depends(verify_admin_token)):
  """
  RabbitMQ 큐 상태 메트릭 조회
  """
  try:
    rabbitmq_host = os.getenv("RABBITMQ_HOST", "rabbitmq")
    rabbitmq_port = os.getenv("RABBITMQ_MGMT_PORT", "15672")
    rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
    rabbitmq_pass = os.getenv("RABBITMQ_PASS", "guest")

    api_url = f"http://{rabbitmq_host}:{rabbitmq_port}/api/queues"

    try:
      response = requests.get(
          api_url,
          auth=(rabbitmq_user, rabbitmq_pass),
          timeout=5
      )
      response.raise_for_status()
      queues_data = response.json()
    except requests.exceptions.RequestException as e:
      logger.warning(f"RabbitMQ Management API 호출 실패: {e}")
      queues_data = []

    queue_metrics = {}
    total_messages = 0

    for queue in queues_data:
      queue_name = queue.get("name", "")
      messages = queue.get("messages", 0)
      consumers = queue.get("consumers", 0)

      if any(keyword in queue_name for keyword in
             ["game_request", "game_log", "inference", "model_load"]):
        queue_metrics[queue_name] = {
          "messages": messages,
          "consumers": consumers
        }
        total_messages += messages

    if not queue_metrics:
      queue_metrics = {
        "game_request_queue": {"messages": 0, "consumers": 0},
        "game_log_queue": {"messages": 0, "consumers": 0}
      }

    if total_messages > 1000:
      queue_status = "CRITICAL"
    elif total_messages > 500:
      queue_status = "CONGESTED"
    else:
      queue_status = "STABLE"

    return {
      "timestamp": datetime.utcnow().isoformat(),
      "totalMessages": total_messages,
      "queues": queue_metrics,
      "status": queue_status
    }
  except Exception as e:
    logger.error(f"Failed to get queue status: {e}")
    raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 게임 요청 엔드포인트
# ============================================================================

@app.post("/game-request", response_model=GameRequestResponse)
@app.post("/ai/game-request", response_model=GameRequestResponse)
def create_game_request(request: GameRequestCreate):
  """게임 요청을 RabbitMQ로 전송"""
  try:
    rabbitmq_request = GameRequest(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        game_id=request.game_id,
        game_type=request.game_type,
        model_ids=request.ai_model_ids,
        model_urls=request.ai_model_urls,
        players=request.player_ids
    )

    client = get_metrics_rabbitmq_client()

    if not client.create_game_progress_queue(request.game_id,
                                             request.player_ids):
      logger.warning(f"게임 진행상황 큐 생성 실패: {request.game_id}")

    if not client.publish_game_request(rabbitmq_request):
      raise RuntimeError("게임 요청 발행 실패")

    logger.info(f"게임 요청 전송 완료: {request.game_id}")
    return GameRequestResponse(
        status="success",
        message="게임 요청이 성공적으로 전송되었습니다.",
        game_id=request.game_id
    )

  except Exception as e:
    logger.error(f"게임 요청 전송 실패: {e}")
    raise HTTPException(
        status_code=500,
        detail=ErrorResponse(
            message="게임 요청 전송 실패",
            detail=str(e)
        ).model_dump()
    )


@app.get("/ai/progress/{game_id}/{player_id}",
         response_model=List[GameProgressResponse])
def get_game_progress(game_id: str, player_id: str, n: int = Query(10, gt=0)):
  """게임 진행상황 조회"""
  try:
    client = get_metrics_rabbitmq_client()
    messages = client.get_game_progress_messages(game_id, player_id, n)

    progress_responses = []
    game_finished = False

    for msg in messages:
      data = msg['data']
      try:
        player_ids = data.get('players', [])
        avg_response_times_dict = data.get('avg_response_times')
        avg_response_times = None
        if avg_response_times_dict is not None and isinstance(
            avg_response_times_dict, dict):
          avg_response_times = [avg_response_times_dict.get(pid, 0) for pid in
                                player_ids]
        elif isinstance(avg_response_times_dict, list):
          avg_response_times = avg_response_times_dict
        else:
          avg_response_times = [0 for _ in player_ids]

        progress_response = GameProgressResponse(
            game_id=data.get('game_id'),
            game_type=data.get('game_type'),
            board_state=data.get('board_state', []),
            turn_number=data.get('turn_number', 0),
            current_turn=data.get('current_turn', 0),
            player_ids=player_ids,
            last_move=data.get('last_move'),
            is_finished=data.get('is_finished', False),
            winner=data.get('winner'),
            is_success=data.get('is_success', False),
            avg_response_times=avg_response_times
        )
        progress_responses.append(progress_response)

        if progress_response.is_finished:
          game_finished = True
          game_type = data.get('game_type', 'unknown')
          result = "finished" if progress_response.is_success else "error"
          record_game_session_completed(game_type, result)

      except Exception as e:
        logger.error(f"게임 진행상황 응답 변환 실패: {e}")
        continue

    if game_finished:
      logger.info(f"게임 종료 확인됨: {game_id}")

    return progress_responses

  except Exception as e:
    logger.error(f"게임 진행상황 조회 실패: {e}")
    raise HTTPException(
        status_code=500,
        detail=ErrorResponse(
            message="게임 진행상황 조회 실패",
            detail=str(e)
        ).model_dump()
    )


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("gateway_main:app", host="0.0.0.0", port=8000, reload=True)
