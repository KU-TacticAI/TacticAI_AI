import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Deque, List
from collections import deque
from message_schemas import (
    GameRequest, GameProgress,  # RabbitMQ용
    GameRequestCreate, GameRequestResponse, GameProgressResponse, ErrorResponse  # FastAPI용
)
from config import Config
from rabbitmq_client import get_rabbitmq_client, close_rabbitmq_client
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

app = FastAPI(title="MQ API Gateway", description="게임 서버와 RabbitMQ 간의 API Gateway")

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행"""
    print("MQ API Gateway 시작...")

@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 실행"""
    print("MQ API Gateway 종료...")
    close_rabbitmq_client()

@app.post("/game-request", response_model=GameRequestResponse)
def create_game_request(request: GameRequestCreate):
    """게임 요청을 RabbitMQ로 전송"""
    try:
        # 1. FastAPI 요청을 RabbitMQ 메시지로 변환
        rabbitmq_request = GameRequest(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            game_id=request.game_id,
            game_type=request.game_type,
            model_ids=request.ai_model_ids,
            model_urls=request.ai_model_urls,
            players=request.player_names
        )
        
        # 2. RabbitMQ 클라이언트 사용
        client = get_rabbitmq_client()
        
        # 3. 연결 및 토폴로지 설정 (클라이언트 내부에서 자동 처리)
        # 연결 상태 확인 및 재연결은 각 메서드 내부에서 처리됨
        
        # 4. 해당 게임의 진행상황 큐 생성
        if not client.create_game_progress_queue(request.game_id):
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


@app.get("/progress/{game_id}", response_model=List[GameProgressResponse])
def get_game_progress(game_id: str, n: int = Query(10, gt=0)):
    """게임 진행상황 조회"""
    try:
        client = get_rabbitmq_client()
        
        # 연결 및 토폴로지 설정 (클라이언트 내부에서 자동 처리)
        # 연결 상태 확인 및 재연결은 각 메서드 내부에서 처리됨
        
        # 게임 진행상황 메시지 조회
        messages = client.get_game_progress_messages(game_id, n)
        
        # GameProgressResponse 형태로 변환
        progress_responses = []
        game_finished = False
        
        for msg in messages:
            data = msg['data']
            try:
                progress_response = GameProgressResponse(
                    game_id=data.get('game_id'),
                    game_type=data.get('game_type'),
                    board_state=data.get('board_state', []),
                    turn_number=data.get('turn_number', 0),
                    current_turn=data.get('current_turn', 0),
                    player_names=data.get('players', []),
                    last_move=data.get('last_move'),
                    is_finished=data.get('is_finished', False),
                    winner=data.get('winner'),
                    is_success=data.get('is_success', False)
                )
                progress_responses.append(progress_response)
                
                # 게임이 종료되었는지 확인
                if progress_response.is_finished:
                    game_finished = True
                    
            except Exception as e:
                logger.error(f"게임 진행상황 응답 변환 실패: {e}")
                continue
        
        # 게임이 종료되었으면 해당 큐 삭제
        if game_finished and progress_responses:
            last_progress = progress_responses[-1]
            if client.delete_game_progress_queue(last_progress.game_id):
                logger.info(f"게임 종료로 인한 큐 삭제 완료: {last_progress.game_id}")
            else:
                logger.warning(f"게임 종료 큐 삭제 실패: {last_progress.game_id}")
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)