import os
import uuid
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    # 서버 식별
    _server_id = os.getenv("SERVER_ID")
    if not _server_id or _server_id.strip() == "":
        _server_id = f"ai_server_{uuid.uuid4().hex[:8]}"
    SERVER_ID = _server_id

    # RabbitMQ 설정
    RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
    RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
    RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
    RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

    # 성능 설정
    MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", 10))
    MAX_SAVED_MODELS = int(os.getenv("MAX_SAVED_MODELS", 20))

    # 외부 시스템 설정
    EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL")
    WEBSOCKET_BROADCAST_ENABLED = os.getenv("WEBSOCKET_BROADCAST_ENABLED", "false").lower() == "true"

    # 환경 설정
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # 큐 및 Exchange 이름
    AI_INFERENCE_EXCHANGE=os.getenv("AI_INFERENCE_EXCHANGE", "ai_inference_exchange")
    AI_INFERENCE_REQUEST_PREFIX=os.getenv("AI_INFERENCE_REQUEST_PREFIX", "ai_request")
    MODEL_LOAD_EXCHANGE=os.getenv("MODEL_LOAD_EXCHANGE", "model_load_exchange")
    MODEL_LOAD_QUEUE=os.getenv("MODEL_LOAD_QUEUE", "model_load_queue")
    RESPONSE_EXCHANGE=os.getenv("RESPONSE_EXCHANGE", "response_exchange")
    RESPONSE_INFERENCE_PREFIX=os.getenv("RESPONSE_INFERENCE_PREFIX", "ai_response")
    RESPONSE_MODEL_LOAD_PREFIX=os.getenv("RESPONSE_MODEL_LOAD_PREFIX", "model_load_response")

    # 디바이스 설정 (CUDA/CPU)
    USE_CUDA = os.getenv("USE_CUDA", "auto").lower()  # 'auto', 'true', 'false'
    if USE_CUDA == "true":
        DEVICE = "cuda"
    elif USE_CUDA == "false":
        DEVICE = "cpu"
    else:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
