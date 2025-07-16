#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RabbitMQ 클라이언트 구현
게임 서버와 AI 서버 간의 메시지 통신을 담당합니다.
"""

import asyncio
import json
import logging
from typing import Dict, Callable, Optional, Any, List
from uuid import uuid4
import aio_pika
from aio_pika import connect_robust, Message, DeliveryMode, ExchangeType, IncomingMessage 
from aio_pika.abc import AbstractConnection, AbstractChannel, AbstractExchange, AbstractQueue

from config import Config
from message_schemas import (
    InferenceRequest, ModelLoadRequest, GameProgress, 
    InferenceResponse, ModelLoadResponse, GameRequest
)

logger = logging.getLogger(__name__)

# 콜백 함수 이름 상수
class CallbackNames:
    GAME_REQUEST = "game_request"
    INFERENCE_RESPONSE = "inference_response"
    MODEL_LOAD_RESPONSE = "model_load_response"
    UNROUTABLE_MESSAGE = "unroutable_message"


class RabbitMQClient:
    def __init__(self, config: Config, server_id: str):
        """RabbitMQ 클라이언트 초기화"""
        self.config = config
        self.server_id = server_id
        self.connection: Optional[AbstractConnection] = None
        self.channel: Optional[AbstractChannel] = None
        
        # Exchanges
        self.game_request_exchange: Optional[AbstractExchange] = None
        self.model_load_exchange: Optional[AbstractExchange] = None
        self.ai_inference_exchange: Optional[AbstractExchange] = None
        self.response_exchange: Optional[AbstractExchange] = None
        self.game_progress_exchange: Optional[AbstractExchange] = None
        
        # Queues
        self.game_request_queue: Optional[AbstractQueue] = None
        self.response_queue: Optional[AbstractQueue] = None
        
        # Callbacks
        self.callbacks: Dict[str, Callable] = {}
        
        # Consumer tasks for graceful shutdown
        self.consumer_tasks: List[asyncio.Task] = []
        
        logger.info(f"RabbitMQ 클라이언트 초기화 완료: {server_id}")
    
    async def connect(self) -> bool:
        """RabbitMQ 서버에 연결"""
        try:
            connection_url = f"amqp://{self.config.RABBITMQ_USER}:{self.config.RABBITMQ_PASSWORD}@{self.config.RABBITMQ_HOST}:{self.config.RABBITMQ_PORT}/"
            
            self.connection = await connect_robust(
                connection_url,
                client_properties={"server_id": self.server_id}
            )
            
            self.channel = await self.connection.channel(
                publisher_confirms=True,
                on_return_raises=True
            )
            await self.channel.set_qos(prefetch_count=self.config.RABBITMQ_PREFETCH_COUNT)
            
            logger.info(f"RabbitMQ 연결 성공: {self.config.RABBITMQ_HOST}:{self.config.RABBITMQ_PORT}")
            return True
            
        except Exception as e:
            logger.error(f"RabbitMQ 연결 실패: {e}")
            return False
    
    async def _on_reconnect(self, connection: AbstractConnection):
        """재연결 시 콜백"""
        logger.info("RabbitMQ 재연결됨, 토폴로지 재설정")
        await self.setup_topology()
    
    async def setup_topology(self):
        """RabbitMQ 토폴로지 설정 (Exchange 및 Queue 생성)"""
        if not self.channel:
            raise RuntimeError("RabbitMQ 채널이 연결되지 않았습니다.")
        
        try:
            # Exchange 생성
            self.game_request_exchange = await self.channel.declare_exchange(
                self.config.GAME_REQUEST_EXCHANGE, ExchangeType.DIRECT, durable=True
            )
            
            self.model_load_exchange = await self.channel.declare_exchange(
                self.config.MODEL_LOAD_EXCHANGE , ExchangeType.DIRECT, durable=True
            )
            
            self.ai_inference_exchange = await self.channel.declare_exchange(
                self.config.AI_INFERENCE_EXCHANGE, ExchangeType.TOPIC, durable=True
            )
            
            self.response_exchange = await self.channel.declare_exchange(
                self.config.RESPONSE_EXCHANGE, ExchangeType.TOPIC, durable=True
            )
            
            self.game_progress_exchange = await self.channel.declare_exchange(
                self.config.GAME_PROGRESS_EXCHANGE, ExchangeType.TOPIC, durable=True
            )
            
            # Queue 생성 및 바인딩
            # 게임 요청 큐 (경쟁 소비)
            self.game_request_queue = await self.channel.declare_queue(
                self.config.GAME_REQUEST_QUEUE, durable=True
            )
            await self.game_request_queue.bind(
                self.game_request_exchange, routing_key=self.config.GAME_REQUEST_QUEUE
            )

            # 모델 로드 큐 (경쟁 소비)
            self.model_load_queue = await self.channel.declare_queue(
                self.config.MODEL_LOAD_QUEUE, durable=True
            )
            await self.model_load_queue.bind(
                self.model_load_exchange, routing_key=self.config.MODEL_LOAD_QUEUE
            )
            
            # 서버별 응답 큐 (exclusive=True이므로 durable=False)
            response_queue_name = f"{self.config.RESPONSE_QUEUE_PREFIX}{self.server_id}"
            self.response_queue = await self.channel.declare_queue(
                response_queue_name, durable=False, exclusive=True
            )
            # 응답 큐 바인딩
            await self.response_queue.bind(
                self.response_exchange, routing_key=f"{self.config.RESPONSE_INFERENCE_PREFIX}.{self.server_id}"
            )
            await self.response_queue.bind(
                self.response_exchange, routing_key=f"{self.config.RESPONSE_MODEL_LOAD_PREFIX}.{self.server_id}"
            )
            
            logger.info("RabbitMQ 토폴로지 설정 완료")
            
        except Exception as e:
            logger.error(f"RabbitMQ 토폴로지 설정 실패: {e}")
            raise
    
    async def publish_inference_request(self, request: InferenceRequest, model_url: str = None):
        """AI 추론 요청 발행"""
        if not self.ai_inference_exchange:
            raise RuntimeError("AI 추론 Exchange가 설정되지 않았습니다.")
        try:
            message_body = request.model_dump_json()
            routing_key = f"{self.config.AI_INFERENCE_REQUEST_PREFIX}.{request.model_id}"
            message = Message(
                message_body.encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                content_type="application/json",
                correlation_id=str(uuid4()),
                headers={"request_type": "inference", "model_id": request.model_id}
            )
            import aio_pika
            try:
                await self.ai_inference_exchange.publish(
                    message, 
                    routing_key=routing_key,
                    mandatory=True
                )
                logger.debug(f"AI 추론 요청 발행: {request.game_id}, 모델: {request.model_id}")
            except aio_pika.exceptions.DeliveryError:
                logger.warning(f"Unroutable: 모델 미로딩 상태, 모델 로드 요청 발행: {request.model_id}")
                if not model_url:
                    logger.error(f"모델 URL을 찾을 수 없습니다: {request.model_id}, game_id: {request.game_id}")
                    return
                from message_schemas import ModelLoadRequest
                import datetime
                model_load_req = ModelLoadRequest(
                    request_id=request.request_id,
                    timestamp=datetime.datetime.utcnow().isoformat(),
                    game_id=request.game_id,
                    model_id=request.model_id,
                    model_url=model_url,
                    game_type=request.game_type,
                    game_server_id=request.game_server_id
                )
                await self.publish_model_load_request(model_load_req)
        except Exception as e:
            logger.error(f"AI 추론 요청 발행 실패: {e}")
            raise
    
    async def publish_model_load_request(self, request: ModelLoadRequest):
        """모델 로드 요청 발행"""
        if not self.model_load_exchange:
            raise RuntimeError("모델 로드 Exchange가 설정되지 않았습니다.")
        
        try:
            message_body = request.model_dump_json()
            routing_key=f"{self.config.MODEL_LOAD_QUEUE}"

            message = Message(
                message_body.encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                content_type="application/json",
                correlation_id=str(uuid4()),
                headers={"request_type": "model_load", "model_id": request.model_id}
            )
            
            await self.model_load_exchange.publish(
                message, 
                routing_key=routing_key,
                mandatory=True
            )
            
            logger.info(f"모델 로드 요청 발행: {request.model_id}")
            
        except Exception as e:
            logger.error(f"모델 로드 요청 발행 실패: {e}")
            raise
    
    async def publish_game_progress(self, progress: GameProgress):
        """게임 진행상황 발행"""
        if not self.game_progress_exchange:
            raise RuntimeError("게임 진행상황 Exchange가 설정되지 않았습니다.")
        
        try:
            message_body = progress.model_dump_json()
            routing_key = f"{self.config.GAME_PROGRESS_QUEUE_PREFIX}.{progress.game_id}"
            
            message = Message(
                message_body.encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                content_type="application/json",
                correlation_id=str(uuid4()),
                headers={"request_type": "game_progress", "game_id": progress.game_id}
            )
            
            await self.game_progress_exchange.publish(
                message, 
                routing_key=routing_key,
                mandatory=True
            )
            
            logger.debug(f"게임 진행상황 발행: {progress.game_id}")
            
        except Exception as e:
            logger.error(f"게임 진행상황 발행 실패: {e}")
            raise
    async def start_consuming(self, callbacks: Dict[str, Callable]):
        """메시지 소비 시작"""
        if not self.game_request_queue or not self.response_queue:
            raise RuntimeError("큐가 설정되지 않았습니다.")
        
        self.callbacks = callbacks
        
        try:
            # 게임 요청 소비
            game_request_task = asyncio.create_task(
                self.game_request_queue.consume(
                    self._handle_game_request,
                    consumer_tag=f"game_request_consumer_{self.server_id}"
                )
            )
            self.consumer_tasks.append(game_request_task)
            
            # 응답 소비
            response_task = asyncio.create_task(
                self.response_queue.consume(
                    self._handle_response,
                    consumer_tag=f"response_consumer_{self.server_id}"
                )
            )
            self.consumer_tasks.append(response_task)
            
            logger.info("메시지 소비 시작")
            
        except Exception as e:
            logger.error(f"메시지 소비 시작 실패: {e}")
            raise
    
    async def _handle_game_request(self, message: IncomingMessage):
        """게임 요청 메시지 처리"""
        async with message.process(ignore_processed=True):
            try:
                data = json.loads(message.body.decode())
                request = GameRequest(**data)
                callback = self.callbacks.get(CallbackNames.GAME_REQUEST)
                if callback:
                    await callback(request)
                else:
                    logger.warning("게임 요청 콜백이 설정되지 않았습니다.")
            except json.JSONDecodeError as e:
                logger.error(f"게임 요청 JSON 파싱 실패: {e}")
                # 실패로 간주하고 ack만 처리 (재처리X)
            except Exception as e:
                logger.error(f"게임 요청 처리 실패: {e}")
                # 실패로 간주하고 ack만 처리 (재처리X)
        # async with 블록을 벗어나면 무조건 ack 처리됨 (ignore_processed=True)
    
    async def _handle_response(self, message: IncomingMessage):
        """응답 메시지 처리"""
        async with message.process():
            try:
                data = json.loads(message.body.decode())
                routing_key = message.routing_key
                
                # 라우팅 키로 메시지 타입 구분
                if routing_key.startswith(f"{self.config.RESPONSE_INFERENCE_PREFIX}.{self.server_id}"):
                    # AI 추론 응답
                    response = InferenceResponse(**data)
                    callback = self.callbacks.get(CallbackNames.INFERENCE_RESPONSE)
                    if callback:
                        await callback(response)
                    else:
                        logger.warning("AI 추론 응답 콜백이 설정되지 않았습니다.")
                elif routing_key.startswith(f"{self.config.RESPONSE_MODEL_LOAD_PREFIX}.{self.server_id}"):
                    # 모델 로드 응답
                    response = ModelLoadResponse(**data)
                    callback = self.callbacks.get(CallbackNames.MODEL_LOAD_RESPONSE)
                    if callback:
                        await callback(response)
                    else:
                        logger.warning("모델 로드 응답 콜백이 설정되지 않았습니다.")
                else:
                    logger.warning(f"알 수 없는 라우팅 키: {routing_key}, 데이터: {data}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"응답 JSON 파싱 실패: {e}")
                await message.nack(requeue=False)
            except Exception as e:
                logger.error(f"응답 처리 실패: {e}")
                await message.nack(requeue=True)
    
    async def handle_returned_message(self, message: IncomingMessage):
        """반환된 메시지 처리 (라우팅 실패)"""
        try:
            data = json.loads(message.body.decode())
            headers = message.headers or {}
            
            # 헤더에서 요청 타입 확인
            request_type = headers.get("request_type")
            model_id = headers.get("model_id")
            
            if request_type == "inference" and model_id:
                # AI 추론 요청이 라우팅되지 않음 - 모델 로드 필요
                try:
                    request = InferenceRequest(**data)
                    callback = self.callbacks.get(CallbackNames.UNROUTABLE_MESSAGE)
                    if callback:
                        await callback(request)
                    else:
                        logger.warning(f"라우팅되지 않은 추론 요청: 모델 {model_id}")
                except Exception as e:
                    logger.error(f"라우팅되지 않은 메시지 처리 실패: {e}")
            
            await message.ack()
            
        except Exception as e:
            logger.error(f"반환된 메시지 처리 실패: {e}")
            try:
                await message.nack(requeue=False)
            except:
                pass
    
    async def disconnect(self):
        """연결 종료"""
        try:
            # Consumer 태스크 취소
            for task in self.consumer_tasks:
                if not task.done():
                    task.cancel()
            
            # 모든 태스크가 완료될 때까지 대기
            if self.consumer_tasks:
                await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
            
            # 채널 및 연결 종료 
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
            
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
                
            logger.info("RabbitMQ 연결 종료")
            
        except Exception as e:
            logger.error(f"RabbitMQ 연결 종료 실패: {e}")