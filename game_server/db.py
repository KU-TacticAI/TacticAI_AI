import os
import time
import logging
from pymongo import MongoClient
from bson import ObjectId
import mysql.connector
from config import get_config
from typing import Any, List, Dict
from dataclasses import dataclass, asdict

# MongoDB 컬렉션: GameInfo, GameDetailLog, AI_ExecutionLog, GameResult
# MySQL 테이블: AI_Statistics

# --- DB 연결 객체 전역 선언 및 초기화 ---
mongodb_client = None
mysql_conn = None

# 로거 설정 (상위에서 로거 핸들러가 설정되어 있어야 함)
logger = logging.getLogger("game_server.db")

def init_db_connections():
    """MongoDB와 MySQL 연결 객체를 전역 변수로 초기화"""
    global mongodb_client, mysql_conn
    mongodb_ok = False
    mysql_ok = False
    # try mongodb first
    try:
        mongodb_client = init_mongodb()
        mongodb_ok = True
        logger.info("MongoDB connection initialized")
    except Exception as e:
        mongodb_client = None
        logger.exception("Failed to initialize MongoDB: %s", e)
    # try mysql
    try:
        mysql_conn = init_mysql()
        mysql_ok = True
        logger.info("MySQL connection initialized")
    except Exception as e:
        mysql_conn = None
        logger.exception("Failed to initialize MySQL: %s", e)

    return mongodb_ok, mysql_ok

# --- 데이터 클래스 정의 ---

@dataclass
class GameInfoSchema:
    player_ids: List[int]
    ai_ids: List[int]
    created_at: str = None
    winner_ai_id: int = None
    game_type: str = None
    # 필요한 경우 추가 필드 정의

@dataclass
class GameDetailLogSchema:
    response_time_ms: int
    board_snapshot: Dict[str, Any]
    turn_count: int
    move_data: str
    execution_time_ms: int
    status: str
    log_output: Dict[str, Any]
    ai_id: int  # MySQL에서는 BIGINT, Python에서는 int
    gameinfo_id: str  # ObjectId를 str로 저장
    # 필요한 경우 추가 필드 정의

@dataclass
class AIExecutionLogSchema:
    # Removed: AI_ExecutionLog
    pass


@dataclass
class GameResultSchema:
    # Removed: GameResult
    pass


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

def update_game_info_winner(gameinfo_id: str, winner_ai_id: int) -> bool:
    """
    Update the GameInfo document's winner_ai_id field.
    """
    try:
        db = mongodb_client[get_config().MONGO_DB_NAME]
        if not ObjectId.is_valid(gameinfo_id):
            logger.error(f"Invalid gameinfo_id for winner update: {gameinfo_id}")
            return False
        oid = ObjectId(gameinfo_id)
        result = db.GameInfo.update_one({"_id": oid}, {"$set": {"winner_ai_id": winner_ai_id}})
        if result.modified_count > 0:
            logger.info(f"GameInfo winner updated: gameinfo_id={gameinfo_id}, winner_ai_id={winner_ai_id}")
            return True
        else:
            if result.matched_count == 0:
                logger.warning(f"GameInfo document not found for id: {gameinfo_id}")
                return False
            logger.info(f"GameInfo winner not changed (possibly same value): gameinfo_id={gameinfo_id}")
            return True
    except Exception as e:
        logger.exception(f"Failed to update GameInfo winner: gameinfo_id={gameinfo_id}, winner_ai_id={winner_ai_id}, error={e}")
        return False

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

# --- MySQL ---
def insert_ai_statistics(data: AIStatisticsSchema):
    """
    AI_Statistics 테이블에 AI 통계 추가
    """
    start = time.time()
    cursor = None
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
        affected = cursor.rowcount
        mysql_conn.commit()
        duration = time.time() - start
        logger.info(f"Inserted AI_Statistics ai_id={data.ai_id}, affected={affected}, duration={duration:.3f}s")
        cursor.close()
        return True
    except Exception as e:
        duration = time.time() - start
        if cursor:
            try:
                mysql_conn.rollback()
            except Exception:
                pass
        logger.exception("Error inserting AI statistics: ai_id=%s, params=%s, duration=%.3fs", data.ai_id, values if 'values' in locals() else None, duration)
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        return False

def update_ai_statistics(ai_id: int, win: int, draw: int, loss: int, turns: int, response_time_ms: int):
    """
    AI_Statistics 테이블의 특정 AI 통계 갱신 (평균 계산 후 카운트 업데이트)
    입력: (ai_id, win, draw, loss, turns, response_time_ms)
    예시: (1, 1, 0, 0, 63, 455) - 승리, 63턴, 455ms 응답시간
    """
    start = time.time()
    cursor = None
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
        affected = cursor.rowcount
        if affected == 0:
            # 해당 ai_id가 없으면 insert 시도 (기본 형태)
            logger.info(f"No rows updated for ai_id={ai_id}, attempting insert")
            from datetime import datetime
            insert_query = """
                INSERT INTO AI_Statistics
                (ai_id, game_count, wins, losses, draws, avg_turns, avg_response_time_ms, win_rate, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
            # game_count is 1 for first game
            insert_values = (
                ai_id, 1, win, loss, draw, turns, response_time_ms, win
            )
            cursor.execute(insert_query, insert_values)
            affected = cursor.rowcount

        mysql_conn.commit()
        duration = time.time() - start
        logger.info(f"Updated AI_Statistics ai_id={ai_id}, affected={affected}, duration={duration:.3f}s")
        cursor.close()
        return True
    except Exception as e:
        duration = time.time() - start
        if cursor:
            try:
                mysql_conn.rollback()
            except Exception:
                pass
        logger.exception("Error updating AI statistics: ai_id=%s, params=%s, duration=%.3fs", ai_id, values if 'values' in locals() else None, duration)
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        return False

# --- DB 연결 함수 ---
def init_mongodb():
    """MongoDB 연결 클라이언트 반환"""
    cfg = get_config()
    client = MongoClient(cfg.get_mongodb_url())
    # 연결 테스트 (ping)
    client.admin.command('ping')
    try:
        ensure_mongodb_schema(client)
    except Exception as e:
        logger.exception(f"Failed to ensure MongoDB schema: {e}")
    return client


def ensure_mongodb_schema(client: MongoClient):
    """
    Ensure required MongoDB collections and indexes exist. Creates collections if missing and adds helpful indexes.
    """
    cfg = get_config()
    dbname = cfg.MONGO_DB_NAME
    db = client[dbname]
    try:
        existing = db.list_collection_names()
    except Exception:
        existing = []

    # Ensure GameInfo collection
    try:
        if 'GameInfo' not in existing:
            db.create_collection('GameInfo')
            logger.info('Created MongoDB collection: GameInfo')
        # indexes
        try:
            db.GameInfo.create_index('created_at')
            db.GameInfo.create_index('winner_ai_id')
            db.GameInfo.create_index('game_type')
        except Exception as ie:
            logger.debug(f'Could not create GameInfo indexes: {ie}')
    except Exception as e:
        logger.exception(f'Error ensuring GameInfo collection: {e}')

    # Ensure GameDetailLog collection
    try:
        if 'GameDetailLog' not in existing:
            db.create_collection('GameDetailLog')
            logger.info('Created MongoDB collection: GameDetailLog')
        # indexes
        try:
            db.GameDetailLog.create_index('gameinfo_id')
            db.GameDetailLog.create_index([('ai_id', 1), ('turn_count', 1)])
        except Exception as ie:
            logger.debug(f'Could not create GameDetailLog indexes: {ie}')
    except Exception as e:
        logger.exception(f'Error ensuring GameDetailLog collection: {e}')

    # Note: AI_ExecutionLog and GameResult collections were removed/merged; we do not recreate them.


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
    try:
        ensure_mysql_schema(conn)
    except Exception as e:
        logger.exception(f"Failed to ensure MySQL schema: {e}")
    return conn


def ensure_mysql_schema(conn):
    """
    Ensure required MySQL tables exist; create them if missing.
    Currently ensures AI_Statistics table.
    """
    cursor = conn.cursor()
    try:
        # Create AI_Statistics if not exists
        create_ai_stats = """
        CREATE TABLE IF NOT EXISTS AI_Statistics (
            ai_id BIGINT PRIMARY KEY,
            game_count INT NOT NULL DEFAULT 0,
            wins INT NOT NULL DEFAULT 0,
            losses INT NOT NULL DEFAULT 0,
            draws INT NOT NULL DEFAULT 0,
            avg_turns DOUBLE NOT NULL DEFAULT 0,
            avg_response_time_ms INT NOT NULL DEFAULT 0,
            win_rate DOUBLE NOT NULL DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """
        cursor.execute(create_ai_stats)
        conn.commit()
        logger.info("Ensured MySQL schema: AI_Statistics table ready")
    finally:
        try:
            cursor.close()
        except Exception:
            pass

