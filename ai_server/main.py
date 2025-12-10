#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 서버 메인 모듈 - Redis 메트릭 자동 보고
"""

import time
import json
import os
import uuid
import threading
import redis
import logging
import signal
import sys
from typing import Optional

from config import Config
from rabbitmq_client import RabbitMQClient
from ai_manager import AIManager
from message_schemas import (
  InferenceRequest, InferenceResponse,
  ModelLoadRequest, ModelLoadResponse
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
)
logger = logging.getLogger("AIServer")


class AIServerWrapper:
  def __init__(self):
    self.server_id = Config.SERVER_ID or f"ai-{uuid.uuid4().hex[:6]}"
    self.manager = AIManager()
    self.mq = RabbitMQClient()
    self.running = True

    # Redis 클라이언트 초기화
    self.redis_client = self._init_redis()

    # 메트릭 보고 스레드 참조
    self.reporter_thread: Optional[threading.Thread] = None

    # 시그널 핸들러 설정
    signal.signal(signal.SIGINT, self._signal_handler)
    signal.signal(signal.SIGTERM, self._signal_handler)

    logger.info(f"AI Server initialized: {self.server_id}")

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

  def _signal_handler(self, signum, frame):
    """시그널 핸들러"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    self.stop()
    sys.exit(0)

  def start(self):
    """서버 시작"""
    logger.info("Starting AI Server...")

    # 1. RabbitMQ 연결 및 Consumer 시작
    if self.mq.connect():
      logger.info("RabbitMQ connected")
      try:
        self.mq.setup_exchanges_and_queues()
        self.mq.consume_model_load_requests(self.handle_load)
        self.mq.start_inference_consumer(self.handle_inference)
        self.mq.start_consuming_thread()
        self.mq.start_reconnect_loop()
        logger.info("RabbitMQ consumers started")
      except Exception as e:
        logger.error(f"Failed to setup RabbitMQ: {e}")
    else:
      logger.warning(
        "Initial RabbitMQ connection failed - reconnect loop will handle it")
      self.mq.start_reconnect_loop()

    # 2. Redis 리포터 스레드 시작
    if self.redis_client:
      self.reporter_thread = threading.Thread(
          target=self.report_metrics_loop,
          name="MetricsReporter",
          daemon=True
      )
      self.reporter_thread.start()
      logger.info("Metrics reporter thread started")
    else:
      logger.warning("Redis not available - metrics reporting disabled")

    logger.info("AI Server started successfully")

    # 3. 메인 스레드 유지
    try:
      while self.running:
        time.sleep(1)
    except KeyboardInterrupt:
      logger.info("KeyboardInterrupt received")
      self.stop()

  def report_metrics_loop(self):
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
            time.sleep(5)
            continue

        # 메트릭 조회
        metrics = self.manager.get_metrics()
        avg_time = metrics.get("avg_inference_time", 0)
        total_inferences = metrics.get("total_inferences", 0)

        # 상태 판단
        if avg_time > 1.0:
          status = "OVERLOADED"
        elif avg_time > 0.5:
          status = "BUSY"
        else:
          status = "HEALTHY"

        # 보고할 데이터
        data = {
          "podId": self.server_id,
          "inferenceTime": round(avg_time, 4),
          "status": status,
          "totalInferences": total_inferences,
          "timestamp": time.time()
        }

        # Redis에 저장 (TTL 10초)
        key = f"metrics:ai:{self.server_id}"
        self.redis_client.setex(key, 10, json.dumps(data))

        if consecutive_failures > 0:
          logger.info("Redis connection recovered")
          consecutive_failures = 0

        logger.debug(f"Metrics reported: {data}")

      except redis.ConnectionError as e:
        consecutive_failures += 1
        logger.error(
          f"Redis connection error ({consecutive_failures}/{max_failures}): {e}")

        if consecutive_failures >= max_failures:
          logger.warning("Too many Redis failures, attempting reconnect...")
          self.redis_client = self._init_redis()
          consecutive_failures = 0
          time.sleep(5)
          continue

      except Exception as e:
        logger.error(f"Unexpected error in metrics reporter: {e}",
                     exc_info=True)

      time.sleep(5)

    logger.info("Metrics reporter loop stopped")

  def stop(self):
    """서버 종료"""
    if not self.running:
      return

    logger.info("Stopping AI Server...")
    self.running = False

    # Redis에서 메트릭 제거
    if self.redis_client:
      try:
        key = f"metrics:ai:{self.server_id}"
        self.redis_client.delete(key)
        logger.info("Metrics removed from Redis")
      except Exception as e:
        logger.error(f"Failed to remove metrics: {e}")

    # RabbitMQ 종료
    try:
      self.mq.stop_consuming()

      if hasattr(self.mq, 'consuming_thread') and self.mq.consuming_thread:
        logger.info("Waiting for RabbitMQ consumer thread...")
        self.mq.consuming_thread.join(timeout=5)

      self.mq.disconnect()
      logger.info("RabbitMQ disconnected")
    except Exception as e:
      logger.error(f"Error stopping RabbitMQ: {e}")

    # Reporter 스레드 대기
    if self.reporter_thread and self.reporter_thread.is_alive():
      logger.info("Waiting for metrics reporter thread...")
      self.reporter_thread.join(timeout=5)

    logger.info("AI Server stopped")

  def handle_load(self, req: ModelLoadRequest):
    """모델 로드 요청 처리"""
    logger.info(
      f"[ModelLoad] request_id={req.request_id} model_id={req.model_id}")

    start_time = time.time()
    res = self.manager.load_model(req)

    if res.success:
      self.mq.bind_inference_model(req.model_id)
      logger.info(f"Model loaded successfully: {req.model_id}")
    else:
      logger.error(f"Model load failed: {req.model_id}")

    self.mq.publish_model_load_response(res, req.game_server_id)

    elapsed = time.time() - start_time
    logger.info(f"[ModelLoad] completed in {elapsed:.3f}s")

  def handle_inference(self, req: InferenceRequest):
    """추론 요청 처리"""
    logger.info(
      f"[Inference] request_id={req.request_id} model_id={req.model_id}")

    start_time = time.time()
    res = self.manager.inference(req)

    # 응답에 서버 ID 추가
    res.ai_server_id = self.server_id

    # 추론 시간 추가
    inference_time_ms = int((time.time() - start_time) * 1000)
    res.response_time_ms = inference_time_ms

    self.mq.publish_inference_response(res, req.game_server_id)

    logger.info(f"[Inference] completed in {inference_time_ms}ms")


def main():
  """메인 함수"""
  logger.info("=" * 60)
  logger.info("AI Server Starting")
  logger.info("=" * 60)

  server = AIServerWrapper()

  try:
    server.start()
  except Exception as e:
    logger.exception(f"Fatal error: {e}")
    server.stop()
    sys.exit(1)


if __name__ == "__main__":
  main()