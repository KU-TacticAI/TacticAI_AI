#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
게임 서버 설정 관리 모듈

환경변수(.env 파일)를 읽어와서 서버 설정을 관리합니다.
기본값을 제공하고 타입 변환을 수행합니다.
"""

import os
import socket
from typing import Optional, Union
from dotenv import load_dotenv

# .env 파일 로드 (있는 경우) - override=True로 환경변수보다 .env 우선
load_dotenv(override=True)


class Config:
    """게임 서버 설정 클래스"""
    
    def __init__(self):
        """설정 초기화"""
        self._load_config()
    
    def _load_config(self):
        """환경변수에서 설정값들을 로드"""
        
        # 서버 식별
        self.SERVER_ID = self._get_server_id()
        
        # RabbitMQ 설정
        self.RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
        self.RABBITMQ_PORT = self._get_int('RABBITMQ_PORT', 5672)
        self.RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'testuser')
        self.RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', 'testuserpw')
        
        # MongoDB 설정
        self.MONGO_HOST = os.getenv('MONGO_HOST', 'localhost')
        self.MONGO_PORT = self._get_int('MONGO_PORT', 27017)
        self.MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'database')
        self.MONGO_USER = os.getenv('MONGO_USER', 'testuser')
        self.MONGO_PASSWORD = os.getenv('MONGO_PASSWORD', 'testuserpw')
        
        # MySQL 설정
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
        self.MYSQL_PORT = self._get_int('MYSQL_PORT', 3306)
        self.MYSQL_USER = os.getenv('MYSQL_USER', 'testuser')
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'testuserpw')
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'database')
        
        # 성능 설정
        self.MAX_CONCURRENT_GAMES = self._get_int('MAX_CONCURRENT_GAMES', 500000)
        self.AI_REQUEST_TIMEOUT = self._get_int('AI_REQUEST_TIMEOUT', 30)
        self.EXTERNAL_SYNC_BATCH_SIZE = self._get_int('EXTERNAL_SYNC_BATCH_SIZE', 100)
        
        # RabbitMQ 동시 처리량 제어
        self.GAME_REQUEST_PREFETCH_COUNT = self._get_int('GAME_REQUEST_PREFETCH_COUNT', 10)
        self.RESPONSE_PREFETCH_COUNT = self._get_int('RESPONSE_PREFETCH_COUNT', 15)
        self.RABBITMQ_PREFETCH_COUNT = int(os.environ.get("RABBITMQ_PREFETCH_COUNT", 10))
        
        # 외부 시스템 설정
        self.EXTERNAL_API_URL = os.getenv('EXTERNAL_API_URL', '')
        self.WEBSOCKET_BROADCAST_ENABLED = self._get_bool('WEBSOCKET_BROADCAST_ENABLED', False)
        
        # 환경 설정
        self.ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.DEBUG = self._get_bool('DEBUG', False)
        

        
        # 큐 및 Exchange 이름 설정
        self._load_queue_config()
        
        # 모델 로드 설정
        self.MODEL_LOAD_TIMEOUT = self._get_int('MODEL_LOAD_TIMEOUT', 120)
        self.AI_REQUEST_RETRY_DELAY = self._get_int('AI_REQUEST_RETRY_DELAY', 5)
        # DB 재연결 설정
        # 최대 재시도 횟수 (0 = 재시도 안함, -1 = 무한 재시도)
        self.DB_RECONNECT_MAX_ATTEMPTS = self._get_int('DB_RECONNECT_MAX_ATTEMPTS', -1)
        # 재시도 간격(초)
        self.DB_RECONNECT_DELAY = self._get_float('DB_RECONNECT_DELAY', 2.0)
        # 영속적 로컬 큐 설정 (DB가 다운되었을 때 기록을 로컬에 쌓고 복구 시 전송)
        self.DB_USE_PERSISTENT_QUEUE = self._get_bool('DB_USE_PERSISTENT_QUEUE', True)
        # 큐 파일 경로 (기본: game_server/persistent_db_queue.jsonl)
        self.DB_QUEUE_PATH = os.getenv('DB_QUEUE_PATH', os.path.join(os.path.dirname(__file__), 'persistent_db_queue.jsonl'))
        # 큐 flush 주기(초)
        self.DB_QUEUE_FLUSH_INTERVAL = self._get_float('DB_QUEUE_FLUSH_INTERVAL', 5.0)
        
        # 디버깅을 위한 로드된 값 확인
        if self.DEBUG:
            print(f"[DEBUG] Config loaded - DEBUG value from env: '{os.getenv('DEBUG')}' -> parsed as: {self.DEBUG}")
    
    def _load_queue_config(self):
        """큐 및 Exchange 설정 로드"""
        
        # 게임 요청 전용 Exchange 설정
        self.GAME_REQUEST_EXCHANGE = os.getenv('GAME_REQUEST_EXCHANGE', 'game_request_exchange')
        self.GAME_REQUEST_QUEUE = os.getenv('GAME_REQUEST_QUEUE', 'game_request_queue')
        
        # AI 추론 요청 전용 Exchange 및 Prefix
        self.AI_INFERENCE_EXCHANGE = os.getenv('AI_INFERENCE_EXCHANGE', 'ai_inference_exchange')
        self.AI_INFERENCE_REQUEST_PREFIX = os.getenv('AI_INFERENCE_REQUEST_PREFIX', 'ai_request')
        
        # 모델 로드 요청 전용 Exchange 및 Queue/Prefix
        self.MODEL_LOAD_EXCHANGE = os.getenv('MODEL_LOAD_EXCHANGE', 'model_load_exchange')
        self.MODEL_LOAD_QUEUE = os.getenv('MODEL_LOAD_QUEUE', 'model_load_queue')
        
        # 통합 응답 Exchange 설정
        self.RESPONSE_EXCHANGE = os.getenv('RESPONSE_EXCHANGE', 'response_exchange')
        self.RESPONSE_QUEUE_PREFIX = os.getenv('RESPONSE_QUEUE_PREFIX', 'response_')
        self.RESPONSE_INFERENCE_PREFIX = os.getenv('RESPONSE_INFERENCE_PREFIX', 'ai_response')
        self.RESPONSE_MODEL_LOAD_PREFIX = os.getenv('RESPONSE_MODEL_LOAD_PREFIX', 'model_load_response')
        
        
        # 게임 진행상황 전용 Exchange 설정
        self.GAME_PROGRESS_EXCHANGE = os.getenv('GAME_PROGRESS_EXCHANGE', 'game_progress_exchange')
        self.GAME_PROGRESS_QUEUE_PREFIX = os.getenv('GAME_PROGRESS_QUEUE_PREFIX', 'game_progress')
    
    def _get_server_id(self) -> str:
        """서버 ID 생성 또는 가져오기"""
        server_id = os.getenv('SERVER_ID')
        if not server_id:
            # 자동 생성: game_server_hostname_pid
            hostname = socket.gethostname()
            pid = os.getpid()
            server_id = f"game_server_{hostname}_{pid}"
        return server_id
    
    def _get_int(self, key: str, default: int) -> int:
        """환경변수에서 정수값 가져오기"""
        try:
            value = os.getenv(key)
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _get_float(self, key: str, default: float) -> float:
        """환경변수에서 실수값 가져오기"""
        try:
            value = os.getenv(key)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """환경변수에서 불린값 가져오기"""
        value = os.getenv(key, '').lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off', ''):
            return False
        else:
            # 알 수 없는 값의 경우 로깅하고 기본값 사용
            print(f"Warning: Unknown boolean value for {key}='{value}', using default: {default}")
            return default
    
    def _get_list(self, key: str, default: list, separator: str = ',') -> list:
        """환경변수에서 리스트값 가져오기"""
        value = os.getenv(key)
        if value:
            return [item.strip() for item in value.split(separator) if item.strip()]
        return default
    
    def is_development(self) -> bool:
        """개발 환경인지 확인"""
        return self.ENVIRONMENT.lower() == 'development'
    
    def is_production(self) -> bool:
        """프로덕션 환경인지 확인"""
        return self.ENVIRONMENT.lower() == 'production'
    
    def get_rabbitmq_url(self) -> str:
        """RabbitMQ 연결 URL 생성"""
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/"
    
    def get_mongodb_url(self) -> str:
        """MongoDB 연결 URL 반환"""
        if self.MONGO_USER and self.MONGO_PASSWORD:
            return (
                f"mongodb://{self.MONGO_USER}:{self.MONGO_PASSWORD}" 
                f"@{self.MONGO_HOST}:{self.MONGO_PORT}/{self.MONGO_DB_NAME}?authSource={self.MONGO_DB_NAME}"
            )
        else:
            return f"mongodb://{self.MONGO_HOST}:{self.MONGO_PORT}/{self.MONGO_DB_NAME}"
    
    def validate_config(self) -> list:
        """설정 검증 및 문제점 반환"""
        issues = []
        
        # 필수 설정 검증
        if not self.RABBITMQ_HOST:
            issues.append("RABBITMQ_HOST가 설정되지 않았습니다.")
        
        if not self.RABBITMQ_USER:
            issues.append("RABBITMQ_USER가 설정되지 않았습니다.")
        
        if not self.RABBITMQ_PASSWORD:
            issues.append("RABBITMQ_PASSWORD가 설정되지 않았습니다.")
        
        if not self.MONGO_HOST:
            issues.append("MONGO_HOST가 설정되지 않았습니다.")
        
        # 포트 범위 검증
        if not (1 <= self.MONGO_PORT <= 65535):
            issues.append(f"MONGO_PORT({self.MONGO_PORT})가 유효하지 않습니다.")
        
        # 성능 설정 검증
        if self.MAX_CONCURRENT_GAMES <= 0:
            issues.append("MAX_CONCURRENT_GAMES는 0보다 커야 합니다.")
        
        if self.AI_REQUEST_TIMEOUT <= 0:
            issues.append("AI_REQUEST_TIMEOUT은 0보다 커야 합니다.")
        
        if self.GAME_REQUEST_PREFETCH_COUNT <= 0:
            issues.append("GAME_REQUEST_PREFETCH_COUNT는 0보다 커야 합니다.")
        
        if self.RESPONSE_PREFETCH_COUNT <= 0:
            issues.append("RESPONSE_PREFETCH_COUNT는 0보다 커야 합니다.")
        
        if self.MODEL_LOAD_TIMEOUT <= 0:
            issues.append("MODEL_LOAD_TIMEOUT은 0보다 커야 합니다.")
        
        if self.AI_REQUEST_RETRY_DELAY <= 0:
            issues.append("AI_REQUEST_RETRY_DELAY는 0보다 커야 합니다.")
        
        return issues
    
    def print_config_summary(self):
        """현재 설정 요약 출력"""
        print("=" * 50)
        print("게임 서버 설정 요약")
        print("=" * 50)
        print(f"서버 ID: {self.SERVER_ID}")
        print(f"환경: {self.ENVIRONMENT}")
        print(f"디버그 모드: {self.DEBUG}")
        print(f"로그 레벨: {self.LOG_LEVEL}")
        print()
        print("RabbitMQ 설정:")
        print(f"  호스트: {self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}")
        print(f"  사용자: {self.RABBITMQ_USER}")
        print(f"  게임 요청 Prefetch: {self.GAME_REQUEST_PREFETCH_COUNT}")
        print(f"  응답 Prefetch: {self.RESPONSE_PREFETCH_COUNT}")
        print()
        print("MongoDB 설정:")
        print(f"  호스트: {self.MONGO_HOST}:{self.MONGO_PORT}")
        print(f"  데이터베이스: {self.MONGO_DB_NAME}")
        print()
        print("MySQL 설정:")
        print(f"  호스트: {self.MYSQL_HOST}:{self.MYSQL_PORT}")
        print(f"  데이터베이스: {self.MYSQL_DATABASE}")
        print()
        print("성능 설정:")
        print(f"  최대 동시 게임: {self.MAX_CONCURRENT_GAMES:,}")
        print(f"  AI 요청 타임아웃: {self.AI_REQUEST_TIMEOUT}초")
        print(f"  외부 동기화 배치 크기: {self.EXTERNAL_SYNC_BATCH_SIZE}")
        print()

        print("큐 설정:")
        print(f"  게임 요청 Exchange: {self.GAME_REQUEST_EXCHANGE}")
        print(f"  AI 추론 Exchange: {self.AI_INFERENCE_EXCHANGE}")
        print(f"  모델 로드 Exchange: {self.MODEL_LOAD_EXCHANGE}")
        print(f"  응답 Exchange: {self.RESPONSE_EXCHANGE}")
        print(f"  게임 진행 Exchange: {self.GAME_PROGRESS_EXCHANGE}")
        print()
        print("모델 로드 설정:")
        print(f"  모델 로드 타임아웃: {self.MODEL_LOAD_TIMEOUT}초")
        print(f"  AI 요청 재시도 지연: {self.AI_REQUEST_RETRY_DELAY}초")
        print("=" * 50)
    
    def __repr__(self) -> str:
        """설정 객체 문자열 표현"""
        return f"Config(SERVER_ID={self.SERVER_ID}, ENVIRONMENT={self.ENVIRONMENT}, DEBUG={self.DEBUG})"
    



# 전역 설정 인스턴스 (지연 초기화)
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """설정 인스턴스 반환 (싱글톤)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reload_config() -> Config:
    """설정 다시 로드"""
    global _config_instance
    # .env 파일 다시 로드
    load_dotenv(override=True)
    _config_instance = Config()
    return _config_instance


if __name__ == "__main__":
    # 설정 테스트
    print("환경변수 직접 확인:")
    print(f"  DEBUG (os.getenv): '{os.getenv('DEBUG')}'")
    print(f"  DEBUG (os.environ.get): '{os.environ.get('DEBUG')}'")
    print()
    
    cfg = get_config()
    cfg.print_config_summary()

    # 설정 검증
    issues = cfg.validate_config()
    if issues:
        print("\n설정 문제점:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 모든 설정이 유효합니다.")
