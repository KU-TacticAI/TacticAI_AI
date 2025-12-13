#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 서버 메인 모듈 - Redis 메트릭 자동 보고
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, Set, Optional, List
from collections import deque
import uuid
import threading
import redis
import logging
import signal
import sys
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

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


# ============================================================
# FastAPI 매트릭 API 정의
# ============================================================

metrics_app = FastAPI(title="AI Server Metrics API", version="1.0.0")

# AIServer 인스턴스를 저장할 전역 변수
_ai_server: Optional['AIServer'] = None


class PodStats(BaseModel):
    """Pod별 통계"""
    podId: str
    inferenceTime: float  # 초 단위
    status: str  # HEALTHY, BUSY, CRITICAL


class AIPerformanceResponse(BaseModel):
    """AI 성능 매트릭 응답 스키마"""
    timestamp: str
    averageInferenceTime: float  # 전체 평균 (초 단위)
    pods: List[PodStats]


def get_status_by_inference_time(inference_time: float) -> str:
    """추론 시간에 따라 상태 판단
    
    기준:
    - HEALTHY: inferenceTime < 0.5초
    - BUSY: 0.5초 <= inferenceTime < 1.0초
    - CRITICAL: inferenceTime >= 1.0초
    """
    if inference_time < 0.5:
        return "HEALTHY"
    elif inference_time < 1.0:
        return "BUSY"
    else:
        return "CRITICAL"


@metrics_app.get("/metric/realtime/ai-performance", response_model=AIPerformanceResponse)
async def get_ai_performance():
    """AI 서버별 실시간 성능 매트릭 조회"""
    global _ai_server
    
    timestamp = datetime.now().isoformat(timespec='seconds')
    
    if _ai_server is None:
        return AIPerformanceResponse(
            timestamp=timestamp,
            averageInferenceTime=0.0,
            pods=[]
        )
    
    # 최근 추론 시간들의 평균 계산
    inference_times = list(_ai_server.recent_inference_times)
    if inference_times:
        avg_inference_time = sum(inference_times) / len(inference_times)
    else:
        avg_inference_time = 0.0
    
    # 현재 Pod 정보
    pod_stats = PodStats(
        podId=_ai_server.server_id,
        inferenceTime=round(avg_inference_time, 3),
        status=get_status_by_inference_time(avg_inference_time)
    )
    
    return AIPerformanceResponse(
        timestamp=timestamp,
        averageInferenceTime=round(avg_inference_time, 3),
        pods=[pod_stats]
    )


@metrics_app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "ok"}


def run_metrics_server(host: str, port: int):
    """FastAPI 서버를 별도 쓰레드에서 실행"""
    uvicorn.run(metrics_app, host=host, port=port, log_level="info")


# ============================================================
# AI 서버 클래스
# ============================================================

class AIServer:
    """AI 서버 클래스"""
    
    # 최근 추론 시간을 저장할 최대 개수
    MAX_INFERENCE_HISTORY = 100
    
    def __init__(self):
        """AI 서버 초기화"""
        self.server_id = Config.SERVER_ID or f"ai_server_{uuid.uuid4().hex[:8]}"
        self.logger = self._setup_logging()
        
        # 컴포넌트 초기화
        self.rabbitmq_client = RabbitMQClient()
        self.ai_manager = AIManager()
        
        # 상태 관리
        self.is_running = False
        self.loaded_models: Set[str] = set()
        self.processing_requests: Set[str] = set()
        
        # 추론 시간 추적 (최근 N개의 추론 시간을 초 단위로 저장)
        self.recent_inference_times: deque = deque(maxlen=self.MAX_INFERENCE_HISTORY)
        
        # 시그널 핸들러 설정
        self._setup_signal_handlers()
        
    def _setup_logging(self) -> logging.Logger:
        """
        로깅 설정
        
        Returns:
            logging.Logger: 설정된 로거
        """
        logger = logging.getLogger("AIServer")
        logger.setLevel(Config.LOG_LEVEL)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger
        
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def handle_signal(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
            sys.exit(0)
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
    def start(self):
        """서버 시작"""
        global _ai_server
        _ai_server = self
        
        self.logger.info("Starting AI Server...")
        
        # 매트릭 API 서버 시작 (별도 쓰레드)
        metrics_host = getattr(Config, 'METRICS_API_HOST', '0.0.0.0')
        metrics_port = getattr(Config, 'METRICS_API_PORT', 8001)
        metrics_thread = threading.Thread(
            target=run_metrics_server,
            args=(metrics_host, metrics_port),
            daemon=True
        )
        metrics_thread.start()
        self.logger.info(f"매트릭 API 서버 시작: http://{metrics_host}:{metrics_port}")
        
        connected = self.rabbitmq_client.connect()
        if not connected:
            self.logger.warning("Initial RabbitMQ connect failed - will continue and let reconnect loop handle it")
            # reconnect 루프 시작
            self.rabbitmq_client.start_reconnect_loop()
        else:
            try:
                self.rabbitmq_client.setup_exchanges_and_queues()
            except Exception as e:
                self.logger.warning(f"Failed to setup exchanges/queues: {e}")
        self.is_running = True
        self.logger.info("AI Server started (may be waiting for RabbitMQ reconnection).")
        # 모델 로드 요청 소비 등록 (등록 실패 시 콜백은 client 내부에 저장되어 재연결 후 등록됨)
        self.rabbitmq_client.consume_model_load_requests(self.handle_model_load_request)
        # 추론 요청 소비 등록 (등록 실패 시 내부에 저장됨)
        self.rabbitmq_client.start_inference_consumer(self.handle_inference_request)
        # consuming 스레드는 단 한 번만 실행
        self.rabbitmq_client.start_consuming_thread()
        # reconnect loop도 항상 시작해 두어 재연결을 시도
        self.rabbitmq_client.start_reconnect_loop()
        # 모델별 바인딩은 모델 로드 성공 시마다 추가
        
    def stop(self):
        """서버 종료"""
        if not self.is_running:
            return
        self.logger.info("Stopping AI Server...")
        self.is_running = False
        self.rabbitmq_client.stop_consuming()
        # consuming 스레드가 있으면 종료를 기다림
        if hasattr(self.rabbitmq_client, 'consuming_thread') and self.rabbitmq_client.consuming_thread is not None:
            self.logger.info("Waiting for consuming thread to finish...")
            self.rabbitmq_client.consuming_thread.join(timeout=5)
        self.rabbitmq_client.disconnect()
        self.logger.info("AI Server stopped.")
        
    def handle_model_load_request(self, request: ModelLoadRequest):
        """
        모델 로드 요청 처리
        Args:
            request: 모델 로드 요청
        """
        self.logger.info(f"[ModelLoad] request_id={request.request_id} model_id={request.model_id}")
        start_time = time.time()
        response = self.ai_manager.load_model(request)
        # 모델 로드 성공 시 해당 모델의 라우팅키만 바인딩
        if response.success:
            self.rabbitmq_client.bind_inference_model(request.model_id)
        self.rabbitmq_client.publish_model_load_response(response, request.game_server_id)
        self._log_request_metrics(request.request_id, "model_load", start_time, response.success)
        
    def handle_inference_request(self, request: InferenceRequest):
        """
        추론 요청 처리
        Args:
            request: 추론 요청
        """
        self.logger.info(f"[Inference] request_id={request.request_id} model_id={request.model_id}")
        start_time = time.time()
        
        # AI 추론 수행
        response = self.ai_manager.inference(request)
        
        # 추론 시간 계산 (밀리초)
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        # 추론 시간 기록 (초 단위로 변환하여 저장)
        inference_time_sec = inference_time_ms / 1000.0
        self.recent_inference_times.append(inference_time_sec)
        
        # 응답에 추론 시간과 AI 서버 ID 추가
        response.response_time_ms = inference_time_ms
        response.ai_server_id = self.server_id
        
        # 응답 발송
        self.rabbitmq_client.publish_inference_response(response, request.game_server_id)
        self._log_request_metrics(request.request_id, "inference", start_time, response.success)
        
    def _create_inference_response(self, request: InferenceRequest, probabilities: list, success: bool = True, message: str = "", response_time_ms: int = 0) -> InferenceResponse:
        """
        추론 응답 생성
        Args:
            request: 원본 요청
            probabilities: 확률 배열
            success: 성공 여부
            message: 응답 메시지
            response_time_ms: 응답 시간 (밀리초)
        Returns:
            InferenceResponse: 생성된 응답
        """
        return InferenceResponse(
            request_id=request.request_id,
            timestamp=datetime.utcnow().isoformat(),
            game_id=request.game_id,
            model_id=request.model_id,
            success=success,
            probabilities=probabilities,
            message=message,
            ai_server_id="",  # handle_inference_request에서 설정
            response_time_ms=response_time_ms
        )
        
    def _create_model_load_response(self, request: ModelLoadRequest, success: bool, message: str = None) -> ModelLoadResponse:
        """
        모델 로드 응답 생성
        Args:
            request: 원본 요청
            success: 성공 여부
            message: 메시지
        Returns:
            ModelLoadResponse: 생성된 응답
        """
        return ModelLoadResponse(
            request_id=request.request_id,
            timestamp=datetime.utcnow().isoformat(),
            game_id=request.game_id,
            model_id=request.model_id,
            success=success,
            message=message,
            ai_server_id=self.server_id
        )
        
    def _log_request_metrics(self, request_id: str, request_type: str, start_time: float, success: bool):
        """
        요청 메트릭 로깅
        Args:
            request_id: 요청 ID
            request_type: 요청 타입
            start_time: 시작 시간
            success: 성공 여부
        """
        latency = time.time() - start_time
        self.logger.info(f"[Metrics] {request_type} request_id={request_id} latency={latency:.3f}s success={success}")
        
    def run(self):
        """서버 실행"""
        try:
            self.start()
            while self.is_running:
                time.sleep(1)
        except Exception as e:
            self.logger.exception(f"Exception in server run: {e}")
            self.stop()
        finally:
            self.stop()

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

