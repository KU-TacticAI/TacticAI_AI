#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQ API Gateway용 RabbitMQ 클라이언트
API Gateway와 RabbitMQ 간의 메시지 통신을 담당합니다.
"""

import json
import logging
from typing import Optional, Dict, Any
import pika
from pika import BlockingConnection, ConnectionParameters, BasicProperties
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from config import Config
from message_schemas import GameRequest, GameProgress

logger = logging.getLogger(__name__)


class RabbitMQClient:
    """API Gateway용 RabbitMQ 클라이언트"""
    
    def __init__(self, config: Config):
        """RabbitMQ 클라이언트 초기화"""
        self.config = config
        self.connection: Optional[BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None
        
        logger.info("RabbitMQ 클라이언트 초기화 완료")
    
    def connect(self) -> bool:
        """RabbitMQ 서버에 연결"""
        try:
            # 연결 파라미터 설정
            if self.config.RABBITMQ_USER and self.config.RABBITMQ_PASSWORD:
                credentials = pika.PlainCredentials(
                    self.config.RABBITMQ_USER, 
                    self.config.RABBITMQ_PASSWORD
                )
                parameters = ConnectionParameters(
                    host=self.config.RABBITMQ_HOST,
                    port=self.config.RABBITMQ_PORT,
                    credentials=credentials
                )
            else:
                parameters = ConnectionParameters(
                    host=self.config.RABBITMQ_HOST,
                    port=self.config.RABBITMQ_PORT
                )
            
            # 연결 생성
            self.connection = BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # QoS 설정
            self.channel.basic_qos(prefetch_count=1)
            
            logger.info(f"RabbitMQ 연결 성공: {self.config.RABBITMQ_HOST}:{self.config.RABBITMQ_PORT}")
            return True
            
        except Exception as e:
            logger.error(f"RabbitMQ 연결 실패: {e}")
            return False
    
    def setup_topology(self):
        """RabbitMQ 토폴로지 설정 (Exchange 및 Queue 생성)"""
        if not self.channel:
            raise RuntimeError("RabbitMQ 채널이 연결되지 않았습니다.")
        
        try:
            # 게임 요청 Exchange 및 Queue 설정
            self.channel.exchange_declare(
                exchange=self.config.GAME_REQUEST_EXCHANGE,
                exchange_type='direct',
                durable=True
            )
            
            self.channel.queue_declare(
                queue=self.config.GAME_REQUEST_QUEUE,
                durable=True
            )
            
            self.channel.queue_bind(
                exchange=self.config.GAME_REQUEST_EXCHANGE,
                queue=self.config.GAME_REQUEST_QUEUE,
                routing_key=self.config.GAME_REQUEST_QUEUE
            )
            
            # 게임 진행상황 Exchange 설정
            self.channel.exchange_declare(
                exchange=self.config.GAME_PROGRESS_EXCHANGE,
                exchange_type='topic',
                durable=True
            )
            
            # 게임별 진행상황 큐는 동적으로 생성되므로 여기서는 Exchange만 설정
            # 실제 큐는 게임 요청 시 game_id별로 생성됨
            
            # 응답 Exchange는 game_server에서 사용하므로 여기서는 생성하지 않음
            # mq_api_gateway는 주로 게임 요청 발행과 진행상황 조회만 담당
            
            logger.info("RabbitMQ 토폴로지 설정 완료")
            
        except Exception as e:
            logger.error(f"RabbitMQ 토폴로지 설정 실패: {e}")
            raise
    
    def publish_game_request(self, request: GameRequest) -> bool:
        """게임 요청 메시지 발행"""
        try:
            # 연결 상태 확인 및 재연결
            if not self.is_connected():
                if not self.connect():
                    logger.error("RabbitMQ 재연결 실패")
                    return False
                self.setup_topology()
            
            message_body = request.model_dump_json()
            
            properties = BasicProperties(
                delivery_mode=2,  # persistent message
                content_type='application/json'
            )
            
            self.channel.basic_publish(
                exchange=self.config.GAME_REQUEST_EXCHANGE,
                routing_key=self.config.GAME_REQUEST_QUEUE,
                body=message_body,
                properties=properties
            )
            
            logger.info(f"게임 요청 발행 성공: {request.game_id}")
            return True
            
        except Exception as e:
            logger.error(f"게임 요청 발행 실패: {e}")
            return False
    

    

    
    def create_game_progress_queue(self, game_id: str) -> bool:
        """특정 게임의 진행상황 큐 생성"""
        try:
            # 연결 상태 확인 및 재연결
            if not self.is_connected():
                if not self.connect():
                    logger.error("RabbitMQ 재연결 실패")
                    return False
                self.setup_topology()
            
            queue_name = f"{self.config.GAME_PROGRESS_QUEUE_PREFIX}_{game_id}"
            routing_key = f"{self.config.GAME_PROGRESS_QUEUE_PREFIX}.{game_id}"
            
            # 큐 생성
            self.channel.queue_declare(
                queue=queue_name,
                durable=True
            )
            
            # Exchange와 바인딩
            self.channel.queue_bind(
                exchange=self.config.GAME_PROGRESS_EXCHANGE,
                queue=queue_name,
                routing_key=routing_key
            )
            
            logger.info(f"게임 진행상황 큐 생성 완료: {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"게임 진행상황 큐 생성 실패: {e}")
            return False
    
    def get_game_progress_messages(self, game_id: str, limit: int = 10) -> list:
        """특정 게임의 진행상황 메시지 조회"""
        if not game_id:
            logger.error("game_id가 필요합니다.")
            return []
        
        try:
            # 연결 상태 확인 및 재연결
            if not self.is_connected():
                if not self.connect():
                    logger.error("RabbitMQ 재연결 실패")
                    return []
                self.setup_topology()
            
            queue_name = f"{self.config.GAME_PROGRESS_QUEUE_PREFIX}_{game_id}"
            
            # 큐가 존재하는지 확인하고 없으면 바로 빈 리스트 반환
            try:
                self.channel.queue_declare(queue=queue_name, passive=True)
            except:
                # 큐가 없으면 바로 빈 리스트 반환
                logger.info(f"게임 진행상황 큐가 존재하지 않음: {queue_name}")
                return []
            
            messages = []
            
            # 게임 진행상황 큐에서 메시지 가져오기
            for i in range(limit):
                method_frame, header_frame, body = self.channel.basic_get(
                    queue_name, 
                    auto_ack=False
                )
                
                if method_frame:
                    try:
                        message_data = json.loads(body.decode())
                        messages.append({
                            'index': len(messages) + 1,
                            'queue': queue_name,
                            'data': message_data,
                            'delivery_tag': method_frame.delivery_tag
                        })
                        
                        # 메시지 처리 완료 시 ACK
                        self.channel.basic_ack(method_frame.delivery_tag)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"게임 진행상황 메시지 JSON 파싱 실패: {e}")
                        self.channel.basic_nack(method_frame.delivery_tag, requeue=False)
                        
                else:
                    break
                    
            return messages
                    
        except Exception as e:
            logger.error(f"게임 진행상황 메시지 조회 실패: {e}")
            return []
    
    def delete_game_progress_queue(self, game_id: str) -> bool:
        """특정 게임의 진행상황 큐 삭제"""
        if not game_id:
            logger.error("game_id가 필요합니다.")
            return False
        
        try:
            # 연결 상태 확인 및 재연결
            if not self.is_connected():
                if not self.connect():
                    logger.error("RabbitMQ 재연결 실패")
                    return False
                self.setup_topology()
            
            queue_name = f"{self.config.GAME_PROGRESS_QUEUE_PREFIX}_{game_id}"
            
            # 큐 삭제
            self.channel.queue_delete(queue=queue_name)
            
            logger.info(f"게임 진행상황 큐 삭제 완료: {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"게임 진행상황 큐 삭제 실패: {e}")
            return False
    
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return (
            self.connection is not None and 
            not self.connection.is_closed and
            self.channel is not None and 
            not self.channel.is_closed
        )
    
    def disconnect(self):
        """연결 종료"""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()
            
            if self.connection and not self.connection.is_closed:
                self.connection.close()
                
            logger.info("RabbitMQ 연결 종료")
            
        except Exception as e:
            logger.error(f"RabbitMQ 연결 종료 실패: {e}")
    
    def __enter__(self):
        """Context manager 진입"""
        if not self.connect():
            raise RuntimeError("RabbitMQ 연결 실패")
        self.setup_topology()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.disconnect()


# 싱글톤 인스턴스
_rabbitmq_client: Optional[RabbitMQClient] = None


def get_rabbitmq_client() -> RabbitMQClient:
    """RabbitMQ 클라이언트 싱글톤 인스턴스 반환"""
    global _rabbitmq_client
    
    if _rabbitmq_client is None:
        config = Config()
        _rabbitmq_client = RabbitMQClient(config)
    
    return _rabbitmq_client


def close_rabbitmq_client():
    """RabbitMQ 클라이언트 연결 종료"""
    global _rabbitmq_client
    
    if _rabbitmq_client:
        _rabbitmq_client.disconnect()
        _rabbitmq_client = None 