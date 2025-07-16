# .env 파일 로드

import os
import socket
from typing import Optional, Union
from dotenv import load_dotenv

# .env 파일 로드 (있는 경우) - override=True로 환경변수보다 .env 우선
load_dotenv(override=True)


class Config:
    # 서버 식별
    SERVER_ID = os.getenv('SERVER_ID')

    # RabbitMQ 설정
    RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
    RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
    RABBITMQ_USER = os.getenv('RABBITMQ_USER')
    RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD')

    # MongoDB 설정
    MONGO_URI = os.getenv('MONGO_URI')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')

    # 외부 시스템 설정
    EXTERNAL_API_URL = os.getenv('EXTERNAL_API_URL')
    WEBSOCKET_BROADCAST_ENABLED = os.getenv('WEBSOCKET_BROADCAST_ENABLED', 'false').lower() == 'true'

    # 환경 설정
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

    # 큐 및 Exchange 이름
    GAME_REQUEST_EXCHANGE = os.getenv('GAME_REQUEST_EXCHANGE', 'game_request_exchange')
    GAME_REQUEST_QUEUE = os.getenv('GAME_REQUEST_QUEUE', 'game_request_queue')
    GAME_PROGRESS_EXCHANGE = os.getenv('GAME_PROGRESS_EXCHANGE', 'game_progress_exchange')
    GAME_PROGRESS_QUEUE_PREFIX = os.getenv('GAME_PROGRESS_QUEUE_PREFIX', 'game_progress')
