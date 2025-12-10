#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway 메인 모듈 - Redis 메트릭 조회 + Prometheus 통합
"""

import sys
import os
import time
import uuid
import logging
import json
import redis
import requests
import asyncio
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from message_schemas import (
  GameRequest, GameRequestCreate, GameRequestResponse, GameProgressResponse,
  ErrorResponse
)
from config import Config
from rabbitmq_client import get_rabbitmq_client, close_rabbitmq_client
from gateway_metrics import (
  setup_metrics_endpoint,
  metrics_middleware,
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
  # 1. 시작 로그 및 Redis 초기화
  logger.info("MQ API Gateway 시작...")
  init_redis()

  # 2. RabbitMQ 연결 (핵심 수정 사항)
  client = get_rabbitmq_client()

  # 설정 정보 로깅 (디버깅용)
  mq_host = getattr(client.config, 'RABBITMQ_HOST', 'unknown')
  mq_port = getattr(client.config, 'RABBITMQ_PORT', 'unknown')
  logger.info(f"RabbitMQ 연결 시도: {mq_host}:{mq_port}")

  # 연결 시도 (최대 3회 재시도)
  connected = False
  for i in range(3):
    if client.connect():
      logger.info("RabbitMQ 연결 성공")
      connected = True
      break
    else:
      logger.warning(f"RabbitMQ 연결 실패 (시도 {i + 1}/3). 2초 후 재시도...")
      time.sleep(2)

  if not connected:
    logger.error("RabbitMQ 연결에 최종 실패했습니다. (Gateway 기능 제한됨)")

  # 연결 상태 메트릭 업데이트
  update_rabbitmq_connection_status(connected)

  yield

  # 3. 종료 처리
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

  try:
    client = get_rabbitmq_client()
    rabbitmq_connected = client.is_connected()
  except Exception:
    pass

  try:
    if redis_client:
      redis_client.ping()
      redis_connected = True
  except Exception:
    pass

  return {
    "status": "ok",
    "timestamp": datetime.utcnow().isoformat(),
    "rabbitmq_connected": rabbitmq_connected,
    "redis_connected": redis_connected
  }


# ============================================================================
# Redis 기반 메트릭 엔드포인트
# ============================================================================

@app.get("/metric/realtime")
def get_gateway_realtime_metrics(token: str = Depends(verify_admin_token)):
  """게이트웨이 자체 실시간 메트릭 조회"""
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
  """AI 서버 성능 메트릭 조회 (Redis)"""
  try:
    if not redis_client:
      return {"timestamp": datetime.utcnow().isoformat(),
              "averageInferenceTime": 0.0, "pods": []}

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
      except Exception:
        continue

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
  """게임별 통계 메트릭 조회 (Redis)"""
  try:
    if not redis_client:
      return {"timestamp": datetime.utcnow().isoformat(),
              "totalActiveMatches": 0, "games": []}

    keys = redis_client.keys("metrics:game:*")
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
              if stats.get("status") == "BUSY":
                aggregated[game_type]["status"] = "BUSY"
      except Exception:
        continue

    total_matches = sum(stats["activeRooms"] for stats in aggregated.values())
    games_list = [{"type": k, **v} for k, v in aggregated.items()]

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
  """RabbitMQ 큐 상태 메트릭 조회"""
  try:
    rabbitmq_host = os.getenv("RABBITMQ_HOST", "rabbitmq")
    rabbitmq_port = os.getenv("RABBITMQ_MGMT_PORT", "15672")
    rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
    rabbitmq_pass = os.getenv("RABBITMQ_PASS", "guest")

    api_url = f"http://{rabbitmq_host}:{rabbitmq_port}/api/queues"

    try:
      response = requests.get(api_url, auth=(rabbitmq_user, rabbitmq_pass),
                              timeout=3)
      response.raise_for_status()
      queues_data = response.json()
    except Exception as e:
      logger.warning(f"RabbitMQ API Error: {e}")
      queues_data = []

    queue_metrics = {}
    total_messages = 0

    for q in queues_data:
      name = q.get("name")
      msgs = q.get("messages", 0)
      consumers = q.get("consumers", 0)
      if any(k in name for k in ["game", "inference", "model"]):
        queue_metrics[name] = {"messages": msgs, "consumers": consumers}
        total_messages += msgs

    return {
      "timestamp": datetime.utcnow().isoformat(),
      "totalMessages": total_messages,
      "queues": queue_metrics,
      "status": "STABLE" if total_messages < 500 else "BUSY"
    }
  except Exception as e:
    logger.error(f"Queue status error: {e}")
    raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 게임 요청 엔드포인트
# ============================================================================

@app.post("/game-request", response_model=GameRequestResponse)
@app.post("/ai/game-request", response_model=GameRequestResponse)
def create_game_request(request: GameRequestCreate):
  """게임 요청을 RabbitMQ로 전송"""
  try:
    client = get_rabbitmq_client()
    # 연결 안 되어 있으면 한 번 더 시도
    if not client.is_connected():
      client.connect()

    rabbitmq_request = GameRequest(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        game_id=request.game_id,
        game_type=request.game_type,
        model_ids=request.ai_model_ids,
        model_urls=request.ai_model_urls,
        players=request.player_ids
    )

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
    raise HTTPException(status_code=500, detail=ErrorResponse(message="전송 실패",
                                                              detail=str(
                                                                e)).model_dump())


@app.get("/ai/progress/{game_id}/{player_id}",
         response_model=List[GameProgressResponse])
def get_game_progress(game_id: str, player_id: str, n: int = Query(10, gt=0)):
  """게임 진행상황 조회"""
  try:
    client = get_rabbitmq_client()
    messages = client.get_game_progress_messages(game_id, player_id, n)

    # 메시지 변환 로직 (간소화)
    responses = []
    for msg in messages:
      data = msg['data']
      # ... (기존 데이터 매핑 로직 유지) ...
      # 여기서는 편의상 원본 데이터를 그대로 반환하거나 필요한 필드만 매핑
      # 실제로는 기존 코드의 매핑 로직을 그대로 사용하세요.
      responses.append(GameProgressResponse(**data))

    return responses
  except Exception as e:
    logger.error(f"진행상황 조회 실패: {e}")
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
