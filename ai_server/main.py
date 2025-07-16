#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 서버 메인 모듈

AI 추론 서버의 메인 진입점입니다.
RabbitMQ를 통한 메시지 처리와 AI 모델 관리를 담당합니다.
"""

import asyncio
import signal
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Set, Optional
import uuid

from config import Config
from rabbitmq_client import RabbitMQClient
from ai_manager import AIManager
from message_schemas import (
    InferenceRequest, InferenceResponse, 
    ModelLoadRequest, ModelLoadResponse
)


class AIServer:
    """AI 서버 클래스"""
    
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
        self.logger.info("Starting AI Server...")
        if not self.rabbitmq_client.connect():
            self.logger.error("Failed to connect to RabbitMQ. Exiting.")
            sys.exit(1)
        self.rabbitmq_client.setup_exchanges_and_queues()
        self.is_running = True
        self.logger.info("AI Server started and ready to receive messages.")
        # 모델 로드 요청 소비 등록 (스레드X)
        self.rabbitmq_client.consume_model_load_requests(self.handle_model_load_request)
        # 추론 요청 소비 등록 (스레드X)
        self.rabbitmq_client.start_inference_consumer(self.handle_inference_request)
        # consuming 스레드는 단 한 번만 실행
        self.rabbitmq_client.start_consuming_thread()
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
        response = self.ai_manager.inference(request)
        self.rabbitmq_client.publish_inference_response(response, request.game_server_id)
        self._log_request_metrics(request.request_id, "inference", start_time, response.success)
        
    def _create_inference_response(self, request: InferenceRequest, probabilities: list, success: bool = True) -> InferenceResponse:
        """
        추론 응답 생성
        Args:
            request: 원본 요청
            probabilities: 확률 배열
            success: 성공 여부
        Returns:
            InferenceResponse: 생성된 응답
        """
        return InferenceResponse(
            request_id=request.request_id,
            timestamp=datetime.utcnow().isoformat(),
            game_id=request.game_id,
            model_id=request.model_id,
            probabilities=probabilities,
            success=success
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


def main():
    """메인 함수"""
    server = AIServer()
    try:
        server.run()
    except KeyboardInterrupt:
        server.logger.info("KeyboardInterrupt received. Exiting...")
        server.stop()
    except Exception as e:
        server.logger.exception(f"Unhandled exception: {e}")
        server.stop()


if __name__ == "__main__":
    main()
