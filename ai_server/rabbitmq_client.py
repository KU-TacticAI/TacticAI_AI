#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RabbitMQ 클라이언트

AI 서버에서 RabbitMQ와의 통신을 담당하는 클라이언트입니다.
메시지 발행, 구독, 큐 관리 등의 기능을 제공합니다.
"""

import json
import logging
from typing import Callable, Optional, Dict, Any
import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from config import Config
from message_schemas import (
    InferenceRequest, InferenceResponse, 
    ModelLoadRequest, ModelLoadResponse
)


class RabbitMQClient:
    """RabbitMQ 클라이언트 클래스"""
    
    def __init__(self):
        """RabbitMQ 클라이언트 초기화"""
        self.connection = None
        self.channel = None
        self.server_id = Config.SERVER_ID
        self.logger = logging.getLogger(__name__)
        
        # 큐 및 exchange 이름들
        self.ai_inference_exchange = Config.AI_INFERENCE_EXCHANGE
        self.model_load_exchange = Config.MODEL_LOAD_EXCHANGE
        self.model_load_queue = Config.MODEL_LOAD_QUEUE
        self.response_exchange = Config.RESPONSE_EXCHANGE
        
    def connect(self) -> bool:
        """
        RabbitMQ 서버에 연결
        Returns:
            bool: 연결 성공 여부
        """
        try:
            credentials = pika.PlainCredentials(Config.RABBITMQ_USER, Config.RABBITMQ_PASSWORD)
            params = pika.ConnectionParameters(
                host=Config.RABBITMQ_HOST,
                port=Config.RABBITMQ_PORT,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()
            self.logger.info("Connected to RabbitMQ.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
        
    def delete_own_queue(self):
        """자신의 inference_request 큐를 별도 연결로 안전하게 삭제"""
        import pika
        queue_name = f"inference_request_{self.server_id}"
        try:
            credentials = pika.PlainCredentials(Config.RABBITMQ_USER, Config.RABBITMQ_PASSWORD)
            params = pika.ConnectionParameters(
                host=Config.RABBITMQ_HOST,
                port=Config.RABBITMQ_PORT,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_delete(queue=queue_name)
            self.logger.info(f"[SAFE] Deleted queue: {queue_name} (via new connection)")
            channel.close()
            connection.close()
        except Exception as e:
            self.logger.warning(f"[SAFE] Failed to delete queue {queue_name} via new connection: {e}")

    def disconnect(self):
        """RabbitMQ 연결 종료"""
        try:
            # 기존 연결 종료
            if self.channel and getattr(self.channel, 'is_open', True):
                self.channel.close()
            if self.connection and getattr(self.connection, 'is_open', True):
                self.connection.close()
            self.logger.info("Disconnected from RabbitMQ.")
            # 별도 연결로 큐 삭제 시도
            self.delete_own_queue()
            self.channel = None
            self.connection = None
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
        
    def setup_exchanges_and_queues(self):
        """Exchange와 Queue 설정"""
        # Exchanges
        self.channel.exchange_declare(exchange=self.ai_inference_exchange, exchange_type='topic', durable=True)
        self.channel.exchange_declare(exchange=self.model_load_exchange, exchange_type='direct', durable=True)
        self.channel.exchange_declare(exchange=self.response_exchange, exchange_type='topic', durable=True)
        # Model load queue
        self.channel.queue_declare(queue=self.model_load_queue, durable=True)
        self.channel.queue_bind(queue=self.model_load_queue, exchange=self.model_load_exchange, routing_key='model_load')
        # AI 서버 전용 inference request 큐 미리 생성
        queue_name = f"inference_request_{self.server_id}"
        self.channel.queue_declare(queue=queue_name, durable=True)
        self.logger.info(f"Pre-created inference request queue: {queue_name}")
        self.logger.info("Exchanges and queues set up.")
        
    def publish_inference_response(self, response: InferenceResponse, game_server_id: str):
        """
        추론 결과 발행
        Args:
            response: 추론 응답 메시지
            game_server_id: 게임 서버 ID
        """
        try:
            routing_key = f"ai_response.{game_server_id}"
            body = response.model_dump_json()
            self.channel.basic_publish(
                exchange=self.response_exchange,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(content_type='application/json', delivery_mode=2)
            )
            self.logger.info(f"Published inference response to {routing_key}")
        except Exception as e:
            self.logger.error(f"Failed to publish inference response: {e}")
        
    def publish_model_load_response(self, response: ModelLoadResponse, game_server_id: str):
        """
        모델 로드 응답 발행
        Args:
            response: 모델 로드 응답 메시지
            game_server_id: 게임 서버 ID
        """
        try:
            routing_key = f"model_load_response.{game_server_id}"
            body = response.model_dump_json()
            self.channel.basic_publish(
                exchange=self.response_exchange,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(content_type='application/json', delivery_mode=2)
            )
            self.logger.info(f"Published model load response to {routing_key}")
        except Exception as e:
            self.logger.error(f"Failed to publish model load response: {e}")
        
    def consume_model_load_requests(self, callback: Callable[[ModelLoadRequest], None]):
        """
        model_load_queue에 대해 basic_consume만 등록 (consuming 스레드는 만들지 않음)
        """
        def on_message(ch, method, properties, body):
            self.logger.info(f"[DEBUG] Received model load message: {body}")
            print("[DEBUG] Received model load message:", body)
            try:
                data = json.loads(body)
                self.logger.info(f"[DEBUG] Parsed JSON: {data}")
                print("[DEBUG] Parsed JSON:", data)
                request = ModelLoadRequest(**data)
                self.logger.info(f"[DEBUG] Parsed ModelLoadRequest: {request}")
                print("[DEBUG] Parsed ModelLoadRequest:", request)
                callback(request)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                self.logger.info("[DEBUG] Acked model load message.")
                print("[DEBUG] Acked model load message.")
            except Exception as e:
                self.logger.error(f"Error processing model load request: {e}")
                print("[DEBUG] Error processing model load request:", e)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                self.logger.info("[DEBUG] Nacked model load message.")
                print("[DEBUG] Nacked model load message.")
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.model_load_queue, on_message_callback=on_message)
        self.logger.info(f"Registered consumer for model load queue: {self.model_load_queue}")

    def start_inference_consumer(self, callback: Callable[[InferenceRequest], None]):
        """
        inference_request_{server_id} 큐에 대해 basic_consume만 등록 (consuming 스레드는 만들지 않음)
        """
        queue_name = f"inference_request_{self.server_id}"
        self.channel.queue_declare(queue=queue_name, durable=True)
        def on_message(ch, method, properties, body):
            try:
                data = json.loads(body)
                request = InferenceRequest(**data)
                callback(request)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                self.logger.error(f"Error processing inference request: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=queue_name, on_message_callback=on_message)
        self.logger.info(f"Registered consumer for inference queue: {queue_name}")

    def bind_inference_model(self, model_id: str):
        """
        inference_request_{server_id} 큐에 모델별 라우팅키 바인딩만 추가
        """
        queue_name = f"inference_request_{self.server_id}"
        binding_key = f"ai_request.{model_id}"
        self.channel.queue_bind(queue=queue_name, exchange=self.ai_inference_exchange, routing_key=binding_key)
        self.logger.info(f"Bound routing_key {binding_key} to queue {queue_name}")

    def start_consuming(self):
        """메시지 소비 시작"""
        self.logger.info("[DEBUG] start_consuming called")
        self.channel.start_consuming()
        
    def stop_consuming(self):
        """메시지 소비 중단"""
        try:
            if self.channel:
                self.channel.stop_consuming()
            self.logger.info("Stopped consuming messages.")
        except Exception as e:
            self.logger.error(f"Error stopping consuming: {e}")

    def start_consuming_thread(self):
        """
        consuming 스레드를 단 한 번만 생성하여 start_consuming() 실행
        """
        import threading
        self.consuming_thread = threading.Thread(target=self.start_consuming, daemon=True)
        self.consuming_thread.start()
        self.logger.info("Started single consuming thread for all queues.")
