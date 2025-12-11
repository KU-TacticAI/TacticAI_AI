import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Deque, List, Optional
from collections import deque
from message_schemas import (
    GameRequest, GameProgress,  # RabbitMQ용
    GameRequestCreate, GameRequestResponse, GameProgressResponse, ErrorResponse  # FastAPI용
)
from config import Config
from rabbitmq_client import get_rabbitmq_client, close_rabbitmq_client
import uuid
import logging
from datetime import datetime, timedelta
import time
import psutil

logger = logging.getLogger(__name__)

app = FastAPI(title="MQ API Gateway", description="게임 서버와 RabbitMQ 간의 API Gateway")

# 앱 시작 시간 (헬스체크용)
START_TIME = None

# 최근 게임 요청 타임스탬프 저장 (최근 10초간 요청 추적용)
GAME_REQUEST_TIMESTAMPS: Deque[float] = deque()

# 현재 활성 사용자 수 추적 (game_id 기반)
ACTIVE_GAMES: Dict[str, float] = {}  # game_id -> 마지막 요청 시간

# Admin 토큰 (Config에서 가져오거나 환경변수 사용)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "admin_secret_token")

# ✅ CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적인 도메인 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행"""
    print("MQ API Gateway 시작...")
    global START_TIME
    START_TIME = datetime.utcnow()

@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 실행"""
    print("MQ API Gateway 종료...")
    close_rabbitmq_client()


@app.get("/health")
def health_check():
    """간단한 헬스체크 엔드포인트
    반환: status, timestamp, uptime(초)
    """
    now = datetime.utcnow()
    uptime = None
    if START_TIME is not None:
        uptime = (now - START_TIME).total_seconds()
    return {
        "status": "ok",
        "timestamp": now.isoformat(),
        "uptime": uptime,
    }


def _cleanup_old_timestamps():
    """10초보다 오래된 타임스탬프 제거"""
    current_time = time.time()
    while GAME_REQUEST_TIMESTAMPS and (current_time - GAME_REQUEST_TIMESTAMPS[0]) > 10:
        GAME_REQUEST_TIMESTAMPS.popleft()


def _cleanup_inactive_games():
    """60초 이상 활동이 없는 게임 제거"""
    current_time = time.time()
    inactive_games = [game_id for game_id, last_time in ACTIVE_GAMES.items() 
                      if (current_time - last_time) > 60]
    for game_id in inactive_games:
        del ACTIVE_GAMES[game_id]


def _get_system_status(request_count: int) -> str:
    """요청 수에 따라 시스템 상태 결정"""
    if request_count < 500:
        return "STABLE"
    elif request_count < 1000:
        return "WARN"
    else:
        return "CRITICAL"


@app.get("/metric/realtime")
def get_realtime_metrics(authorization: Optional[str] = Header(None)):
    """
    실시간 사용자 수 (게임요청 최근 10초)
    Header: Authorization: Bearer {admin_token}
    """
    # # 토큰 검증
    # if not authorization:
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    # parts = authorization.split(" ")
    # if len(parts) != 2 or parts[0].lower() != "bearer":
    #     raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    # token = parts[1]
    # if token != ADMIN_TOKEN:
    #     raise HTTPException(status_code=403, detail="Invalid admin token")
    
    # 오래된 타임스탬프 정리
    _cleanup_old_timestamps()
    _cleanup_inactive_games()
    
    # 메트릭 계산
    total_requests_last_10s = len(GAME_REQUEST_TIMESTAMPS)
    active_user_count = len(ACTIVE_GAMES)
    
    # CPU 사용률 가져오기 (선택)
    try:
        gateway_cpu_usage = psutil.cpu_percent(interval=0.1)
    except Exception:
        gateway_cpu_usage = 0.0
    
    system_status = _get_system_status(total_requests_last_10s)
    
    return {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "systemStatus": system_status,
        "metrics": {
            "activeUserCount": active_user_count,
            "totalRequestsLast10s": total_requests_last_10s,
            "gatewayCpuUsage": round(gateway_cpu_usage, 1)
        }
    }


@app.post("/game-request", response_model=GameRequestResponse)
@app.post("/ai/game-request", response_model=GameRequestResponse)
def create_game_request(request: GameRequestCreate):
    """게임 요청을 RabbitMQ로 전송"""
    try:
        # 메트릭 추적: 요청 타임스탬프 기록
        current_time = time.time()
        GAME_REQUEST_TIMESTAMPS.append(current_time)
        ACTIVE_GAMES[request.game_id] = current_time
        # 1. FastAPI 요청을 RabbitMQ 메시지로 변환
        rabbitmq_request = GameRequest(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            game_id=request.game_id,
            game_type=request.game_type,
            model_ids=request.ai_model_ids,
            model_urls=request.ai_model_urls,
            players=request.player_ids
        )
        
        # 2. RabbitMQ 클라이언트 사용
        client = get_rabbitmq_client()
        
        # 3. 연결 및 토폴로지 설정 (클라이언트 내부에서 자동 처리)
        # 연결 상태 확인 및 재연결은 각 메서드 내부에서 처리됨
        
        # 4. 해당 게임의 진행상황 큐 생성
        if not client.create_game_progress_queue(request.game_id,request.player_ids):
            logger.warning(f"게임 진행상황 큐 생성 실패: {request.game_id}")
        
        # 5. 메시지 발행
        if not client.publish_game_request(rabbitmq_request):
            raise RuntimeError("게임 요청 발행 실패")
        
        print(f"게임 요청 전송 완료: {request.game_id}, 게임 타입: {request.game_type}")
        return GameRequestResponse(
            status="success",
            message="게임 요청이 성공적으로 전송되었습니다.",
            game_id=request.game_id
        )
        
    except Exception as e:
        print(f"게임 요청 전송 실패: {e}")
        raise HTTPException(
            status_code=500, 
            detail=ErrorResponse(
                message="게임 요청 전송 실패",
                detail=str(e)
            ).model_dump()
        )


@app.get("/ai/progress/{game_id}/{player_id}", response_model=List[GameProgressResponse])
def get_game_progress(game_id: str,player_id:str, n: int = Query(10, gt=0)):
    """게임 진행상황 조회"""
    try:
        client = get_rabbitmq_client()
        
        # 연결 및 토폴로지 설정 (클라이언트 내부에서 자동 처리)
        # 연결 상태 확인 및 재연결은 각 메서드 내부에서 처리됨
        
        # 게임 진행상황 메시지 조회
        messages = client.get_game_progress_messages(game_id,player_id, n)
        
        # GameProgressResponse 형태로 변환
        progress_responses = []
        game_finished = False
        
        for msg in messages:
            data = msg['data']
            try:
                player_ids = data.get('players', [])
                avg_response_times_dict = data.get('avg_response_times')
                avg_response_times = None
                if avg_response_times_dict is not None and isinstance(avg_response_times_dict, dict):
                    # dict -> list (player_ids 순서)
                    avg_response_times = [avg_response_times_dict.get(pid, 0) for pid in player_ids]
                elif isinstance(avg_response_times_dict, list):
                    avg_response_times = avg_response_times_dict
                else:
                    avg_response_times = [0 for _ in player_ids]
                progress_response = GameProgressResponse(
                    game_id=data.get('game_id'),
                    game_type=data.get('game_type'),
                    board_state=data.get('board_state', []),
                    turn_number=data.get('turn_number', 0),
                    current_turn=data.get('current_turn', 0),
                    player_ids=player_ids,
                    last_move=data.get('last_move'),
                    is_finished=data.get('is_finished', False),
                    winner=data.get('winner'),
                    is_success=data.get('is_success', False),
                    avg_response_times=avg_response_times
                )
                progress_responses.append(progress_response)
                # 게임이 종료되었는지 확인
                if progress_response.is_finished:
                    game_finished = True
            except Exception as e:
                logger.error(f"게임 진행상황 응답 변환 실패: {e}")
                continue
        
        # 게임이 종료되었어도 큐 삭제는 하지 않음 (게이트웨이에서 관리하지 않음)
        if game_finished:
            logger.info(f"게임 종료 확인됨 (큐 삭제는 수행하지 않음): {game_id}")
        
        return progress_responses
        
    except Exception as e:
        logger.error(f"게임 진행상황 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                message="게임 진행상황 조회 실패",
                detail=str(e)
            ).model_dump()
        )





if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)