#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
게임 서버 메인 모듈 - Redis 메트릭 자동 보고
"""

import asyncio
import logging
import time
import json
import os
import redis
from typing import Optional
from datetime import datetime

from config import Config
from rabbitmq_client import RabbitMQClient
from game_manager import GameManager
import db

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
)
logger = logging.getLogger("GameServer")


class GameServerMetrics:
  """게임 서버 메트릭 수집 클래스"""

  def __init__(self):
    self.game_stats = {
      "CHESS": {"activeRooms": 0, "waitingUsers": 0, "status": "HEALTHY"},
      "GOMOKU": {"activeRooms": 0, "waitingUsers": 0, "status": "HEALTHY"},
      "OTHELLO": {"activeRooms": 0, "waitingUsers": 0, "status": "HEALTHY"},
      "TICTACTOE": {"activeRooms": 0, "waitingUsers": 0, "status": "HEALTHY"}
    }
    self.total_active_matches = 0
    self.last_update = datetime.utcnow()

  def update_game_stats(self, game_type: str, active_rooms: int,
      waiting_users: int):
    """게임 통계 업데이트"""
    if game_type in self.game_stats:
      self.game_stats[game_type]["activeRooms"] = active_rooms
      self.game_stats[game_type]["waitingUsers"] = waiting_users

      # 상태 결정
      if active_rooms > 200:
        self.game_stats[game_type]["status"] = "BUSY"
      elif active_rooms > 300:
        self.game_stats[game_type]["status"] = "OVERLOADED"
      else:
        self.game_stats[game_type]["status"] = "HEALTHY"

    self.total_active_matches = sum(
        stats["activeRooms"] for stats in self.game_stats.values()
    )
    self.last_update = datetime.utcnow()

  def get_metrics(self) -> dict:
    """현재 메트릭 반환"""
    return {
      "timestamp": self.last_update.isoformat(),
      "totalActiveMatches": self.total_active_matches,
      "game_stats": self.game_stats
    }


class GameServerWrapper:
  """게임 서버 래퍼"""

  def __init__(self, config: Config):
    self.config = config
    self.server_id = config.SERVER_ID or "game-server-01"

    # 메트릭 수집기
    self.metrics = GameServerMetrics()

    # Redis 클라이언트
    self.redis_client = self._init_redis()

    # RabbitMQ 및 GameManager (나중에 초기화)
    self.rabbitmq: Optional[RabbitMQClient] = None
    self.manager: Optional[GameManager] = None
    self.is_initialized = False
    self.running = True

  def _init_redis(self) -> Optional[redis.Redis]:
    """Redis 클라이언트 초기화"""
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", 6379))

    try:
      client = redis.Redis(
          host=redis_host,
          port=redis_port,
          db=0,
          decode_responses=True,
          socket_connect_timeout=5,
          socket_timeout=5
      )
      client.ping()
      logger.info(f"Redis connected: {redis_host}:{redis_port}")
      return client
    except Exception as e:
      logger.error(f"Redis connection failed: {e}")
      logger.warning("Continuing without Redis (metrics will not be reported)")
      return None

  async def initialize(self):
    """비동기 초기화"""
    logger.info("게임 서버 초기화 시작")

    # DB 연결
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
        break
      except Exception as e:
        logger.exception(f"DB 초기화 중 오류: {e}")
      logger.info("DB 연결 실패. 5초 후 재시도")
      await asyncio.sleep(5)

    # RabbitMQ 연결
    self.rabbitmq = RabbitMQClient(self.config, self.server_id)
    while True:
      connected = await self.rabbitmq.connect()
      if connected:
        break
      logger.error("RabbitMQ 연결 실패. 5초 후 재시도")
      await asyncio.sleep(5)

    await self.rabbitmq.setup_topology()

    # GameManager 생성 (메트릭 전달)
    self.manager = GameManager(self.rabbitmq, self.metrics)

    # 콜백 등록
    callbacks = {
      "game_request": self.manager.handle_game_request,
      "inference_response": self.manager.handle_inference_response,
      "model_load_response": self.manager.handle_model_load_response,
    }
    await self.rabbitmq.start_consuming(callbacks)

    self.is_initialized = True
    logger.info("게임 서버 초기화 완료")

  async def report_metrics_loop(self):
    """주기적으로 Redis에 메트릭 보고"""
    logger.info("Metrics reporter loop started")

    consecutive_failures = 0
    max_failures = 5

    while self.running:
      try:
        # Redis 연결 확인
        if not self.redis_client:
          logger.warning("Redis client not initialized, retrying...")
          self.redis_client = self._init_redis()
          if not self.redis_client:
            await asyncio.sleep(5)
            continue

        # 메트릭 조회
        metrics_data = self.metrics.get_metrics()

        # Redis에 저장 (TTL 10초)
        key = f"metrics:game:{self.server_id}"
        self.redis_client.setex(key, 10, json.dumps(metrics_data))

        if consecutive_failures > 0:
          logger.info("Redis connection recovered")
          consecutive_failures = 0

        logger.debug(f"Metrics reported: {metrics_data}")

      except redis.ConnectionError as e:
        consecutive_failures += 1
        logger.error(
          f"Redis connection error ({consecutive_failures}/{max_failures}): {e}")

        if consecutive_failures >= max_failures:
          logger.warning("Too many Redis failures, attempting reconnect...")
          self.redis_client = self._init_redis()
          consecutive_failures = 0
          await asyncio.sleep(5)
          continue

      except Exception as e:
        logger.error(f"Unexpected error in metrics reporter: {e}",
                     exc_info=True)

      await asyncio.sleep(5)

    logger.info("Metrics reporter loop stopped")

  async def shutdown(self):
    """종료"""
    logger.info("게임 서버 종료 중")
    self.running = False

    # Redis에서 메트릭 제거
    if self.redis_client:
      try:
        key = f"metrics:game:{self.server_id}"
        self.redis_client.delete(key)
        logger.info("Metrics removed from Redis")
      except Exception as e:
        logger.error(f"Failed to remove metrics: {e}")

    if self.rabbitmq:
      await self.rabbitmq.disconnect()
    logger.info("게임 서버 종료 완료")


async def main():
  """메인 함수"""
  logger.info("=" * 60)
  logger.info("Game Server Starting")
  logger.info("=" * 60)

  config = Config()
  server = GameServerWrapper(config)

  try:
    # 초기화
    await server.initialize()

    # 메트릭 리포터 태스크 시작
    metrics_task = asyncio.create_task(server.report_metrics_loop())

    # graceful shutdown 처리
    try:
      while True:
        await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
      logger.info("서버 종료 신호 수신")

  except Exception as e:
    logger.exception(f"Fatal error: {e}")
  finally:
    await server.shutdown()
    logger.info("서버 정상 종료")


if __name__ == "__main__":
  asyncio.run(main())