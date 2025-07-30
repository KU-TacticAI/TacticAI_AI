import os
from pymongo import MongoClient
import mysql.connector
from config import get_config
from typing import Any, List, Dict
from dataclasses import dataclass, asdict

# MongoDB 컬렉션: GameInfo, GameDetailLog, AI_ExecutionLog, GameResult
# MySQL 테이블: AI_Statistics

# --- DB 연결 객체 전역 선언 및 초기화 ---
mongodb_client = None
mysql_conn = None

def init_db_connections():
    """MongoDB와 MySQL 연결 객체를 전역 변수로 초기화"""
    global mongodb_client, mysql_conn
    mongodb_client = init_mongodb()
    mysql_conn = init_mysql()

# --- 데이터 클래스 정의 ---

@dataclass
class GameInfoSchema:
    player_ids: List[int]
    ai_ids: List[int]
    # 필요한 경우 추가 필드 정의

@dataclass
class GameDetailLogSchema:
    response_time_ms: int
    board_snapshot: Dict[str, Any]
    turn_count: int
    move_data: str
    ai_id: int  # MySQL에서는 BIGINT, Python에서는 int
    gameinfo_id: str  # ObjectId를 str로 저장
    # 필요한 경우 추가 필드 정의

@dataclass
class AIExecutionLogSchema:
    execution_time_ms: int
    status: str  # 예: 'success', 'failed', 'timeout'
    log_output: dict
    ai_id: int  # MySQL에서는 BIGINT, Python에서는 int
    gameinfo_id: str  # ObjectId를 str로 저장
    # 필요한 경우 추가 필드 정의

@dataclass
class GameResultSchema:
    created_at: str  # ISO8601 datetime string 등
    winner_ai_id: int  # MySQL에서는 BIGINT, Python에서는 int
    gameinfo_id: str  # ObjectId를 str로 저장
    # 필요한 경우 추가 필드 정의

@dataclass
class AIStatisticsSchema:
    ai_id: int  # MySQL에서는 BIGINT, Python에서는 int
    game_count: int
    wins: int
    losses: int
    draws: int
    avg_turns: float
    avg_response_time_ms: int
    win_rate: float

# --- MongoDB ---

def insert_game_info(data: GameInfoSchema):
    """
    GameInfo 컬렉션에 게임 정보 추가
    """
    try:
        db = mongodb_client[get_config().MONGO_DB_NAME]
        result = db.GameInfo.insert_one(asdict(data))
        return str(result.inserted_id)
    except Exception as e:
        return None

def insert_game_detail_log(data: GameDetailLogSchema):
    """
    GameDetailLog 컬렉션에 게임 상세 정보 추가
    """
    try:
        db = mongodb_client[get_config().MONGO_DB_NAME]
        result = db.GameDetailLog.insert_one(asdict(data))
        return str(result.inserted_id)
    except Exception as e:
        return None

def insert_ai_execution_log(data: AIExecutionLogSchema):
    """
    AI_ExecutionLog 컬렉션에 AI 실행 기록 추가
    """
    try:
        db = mongodb_client[get_config().MONGO_DB_NAME]
        result = db.AI_ExecutionLog.insert_one(asdict(data))
        return str(result.inserted_id)
    except Exception as e:
        return None

def insert_game_result(data: GameResultSchema):
    """
    GameResult 컬렉션에 게임 결과 추가
    """
    try:
        db = mongodb_client[get_config().MONGO_DB_NAME]
        result = db.GameResult.insert_one(asdict(data))
        return str(result.inserted_id)
    except Exception as e:
        return None

# --- MySQL ---
def insert_ai_statistics(data: AIStatisticsSchema):
    """
    AI_Statistics 테이블에 AI 통계 추가
    """
    try:
        cursor = mysql_conn.cursor()
        query = """
            INSERT INTO AI_Statistics 
            (ai_id, game_count, wins, losses, draws, avg_turns, avg_response_time_ms, win_rate, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        values = (
            data.ai_id, data.game_count, data.wins, data.losses, data.draws,
            data.avg_turns, data.avg_response_time_ms, data.win_rate
        )
        cursor.execute(query, values)
        mysql_conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"Error inserting AI statistics: {e}")
        return False

def update_ai_statistics(ai_id: int, win: int, draw: int, loss: int, turns: int, response_time_ms: int):
    """
    AI_Statistics 테이블의 특정 AI 통계 갱신 (평균 계산 후 카운트 업데이트)
    입력: (ai_id, win, draw, loss, turns, response_time_ms)
    예시: (1, 1, 0, 0, 63, 455) - 승리, 63턴, 455ms 응답시간
    """
    try:
        cursor = mysql_conn.cursor()
        
        update_query = """
            UPDATE AI_Statistics
            SET
              avg_turns            = (avg_turns * game_count + %s) / (game_count + 1),
              avg_response_time_ms = (avg_response_time_ms * game_count + %s) / (game_count + 1),
              win_rate             = (win_rate * game_count + %s) / (game_count + 1),
              
              game_count = game_count + 1,
              wins       = wins   + %s,
              draws      = draws  + %s,
              losses     = losses + %s,
              
              updated_at = NOW()
            WHERE ai_id = %s
        """
        values = (turns, response_time_ms, win, win, draw, loss, ai_id)
        cursor.execute(update_query, values)
        
        mysql_conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"Error updating AI statistics: {e}")
        return False

# --- DB 연결 함수 ---
def init_mongodb():
    """MongoDB 연결 클라이언트 반환"""
    cfg = get_config()
    client = MongoClient(cfg.get_mongodb_url())
    # 연결 테스트 (ping)
    client.admin.command('ping')
    return client


def init_mysql():
    """MySQL 연결 커넥션 반환"""
    cfg = get_config()
    conn = mysql.connector.connect(
        host=cfg.MYSQL_HOST,
        port=cfg.MYSQL_PORT,
        user=cfg.MYSQL_USER,
        password=cfg.MYSQL_PASSWORD,
        database=cfg.MYSQL_DATABASE
    )
    # 연결 테스트 (ping)
    conn.ping(reconnect=True, attempts=1, delay=0)
    return conn

