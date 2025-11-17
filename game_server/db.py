import os
import time
import logging
from pymongo import MongoClient
from bson import ObjectId
import mysql.connector
from config import get_config
from typing import Any, List, Dict
from dataclasses import dataclass, asdict
try:
    # local persistent queue (optional)
    from db_queue import enqueue as _enqueue_db
except Exception:
    _enqueue_db = None

# MongoDB 컬렉션: GameInfo, GameDetailLog, AI_ExecutionLog, GameResult
# MySQL 테이블: ai_statistics

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


def _should_retry(attempt: int, max_attempts: int) -> bool:
    if max_attempts < 0:
        return True
    return attempt < max_attempts


def ensure_mongodb_connected() -> bool:
    """Ensure the global `mongodb_client` is connected. Attempt to reconnect according to config."""
    global mongodb_client
    cfg = get_config()
    attempt = 0
    delay = cfg.DB_RECONNECT_DELAY
    max_attempts = cfg.DB_RECONNECT_MAX_ATTEMPTS

    while not mongodb_client:
        attempt += 1
        try:
            logger.info(f"Attempting MongoDB connect (attempt {attempt})")
            mongodb_client = init_mongodb()
            logger.info("MongoDB reconnected")
            return True
        except Exception as e:
            mongodb_client = None
            logger.exception(f"MongoDB reconnect attempt {attempt} failed: {e}")
            if not _should_retry(attempt, max_attempts):
                logger.error("MongoDB reconnect max attempts reached, giving up")
                return False
            time.sleep(delay)
    # already connected
    try:
        mongodb_client.admin.command('ping')
        return True
    except Exception:
        # try reconnect loop once
        mongodb_client = None
        return ensure_mongodb_connected()


def ensure_mysql_connected() -> bool:
    """Ensure the global `mysql_conn` is connected. Attempt to reconnect according to config."""
    global mysql_conn
    cfg = get_config()
    attempt = 0
    delay = cfg.DB_RECONNECT_DELAY
    max_attempts = cfg.DB_RECONNECT_MAX_ATTEMPTS

    while True:
        attempt += 1
        if mysql_conn:
            try:
                mysql_conn.ping(reconnect=True, attempts=1, delay=0)
                return True
            except Exception:
                try:
                    mysql_conn.close()
                except Exception:
                    pass
                mysql_conn = None

        try:
            logger.info(f"Attempting MySQL connect (attempt {attempt})")
            mysql_conn = init_mysql()
            logger.info("MySQL reconnected")
            return True
        except Exception as e:
            mysql_conn = None
            logger.exception(f"MySQL reconnect attempt {attempt} failed: {e}")
            if not _should_retry(attempt, max_attempts):
                logger.error("MySQL reconnect max attempts reached, giving up")
                return False
            time.sleep(delay)


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
        if not ensure_mongodb_connected():
            logger.error("Cannot update GameInfo winner because MongoDB is not connected")
            # enqueue for later if configured
            try:
                cfg = get_config()
                if cfg.DB_USE_PERSISTENT_QUEUE and _enqueue_db:
                    _enqueue_db('update_game_info_winner', {'gameinfo_id': gameinfo_id, 'winner_ai_id': winner_ai_id})
            except Exception:
                pass
            return False
        db = mongodb_client[get_config().MONGO_DB_NAME]
        if not ObjectId.is_valid(gameinfo_id):
            logger.error(f"Invalid gameinfo_id for winner update: {gameinfo_id}")
            return False
        oid = ObjectId(gameinfo_id)
        result = db['game_info'].update_one({"_id": oid}, {"$set": {"winner_ai_id": winner_ai_id}})
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

def insert_game_info(data: GameInfoSchema, client_gameinfo_id: str = None):
    """
    GameInfo 컬렉션에 게임 정보 추가
    """
    try:
        if not ensure_mongodb_connected():
            logger.error("Cannot insert GameInfo because MongoDB is not connected")
            try:
                cfg = get_config()
                if cfg.DB_USE_PERSISTENT_QUEUE and _enqueue_db:
                    payload = asdict(data)
                    if client_gameinfo_id:
                        payload['client_gameinfo_id'] = client_gameinfo_id
                    _enqueue_db('insert_game_info', payload)
            except Exception:
                pass
            return None
        db = mongodb_client[get_config().MONGO_DB_NAME]
        doc = asdict(data)
        if client_gameinfo_id:
            # store the client id so queued logs can be resolved later
            doc['client_gameinfo_id'] = client_gameinfo_id
        result = db['game_info'].insert_one(doc)
        return str(result.inserted_id)
    except Exception as e:
        try:
            cfg = get_config()
            if cfg.DB_USE_PERSISTENT_QUEUE and _enqueue_db:
                payload = asdict(data)
                if client_gameinfo_id:
                    payload['client_gameinfo_id'] = client_gameinfo_id
                _enqueue_db('insert_game_info', payload)
        except Exception:
            pass
        logger.exception(f"Failed to insert GameInfo: {e}")
        return None

def insert_game_detail_log(data: GameDetailLogSchema):
    """
    GameDetailLog 컬렉션에 게임 상세 정보 추가
    """
    try:
        if not ensure_mongodb_connected():
            logger.error("Cannot insert GameDetailLog because MongoDB is not connected")
            try:
                cfg = get_config()
                if cfg.DB_USE_PERSISTENT_QUEUE and _enqueue_db:
                    _enqueue_db('insert_game_detail_log', asdict(data))
            except Exception:
                pass
            return None
        db = mongodb_client[get_config().MONGO_DB_NAME]
        result = db['game_detail_log'].insert_one(asdict(data))
        return str(result.inserted_id)
    except Exception as e:
        try:
            cfg = get_config()
            if cfg.DB_USE_PERSISTENT_QUEUE and _enqueue_db:
                _enqueue_db('insert_game_detail_log', asdict(data))
        except Exception:
            pass
        logger.exception(f"Failed to insert GameDetailLog: {e}")
        return None

# --- MySQL ---
def insert_ai_statistics(data: AIStatisticsSchema):
    """
    ai_statistics 테이블에 AI 통계 추가
    """
    start = time.time()
    cursor = None
    try:
        if not ensure_mysql_connected():
            logger.error("Cannot insert AI statistics because MySQL is not connected")
            try:
                cfg = get_config()
                if cfg.DB_USE_PERSISTENT_QUEUE and _enqueue_db:
                    _enqueue_db('insert_ai_statistics', asdict(data))
            except Exception:
                pass
            return False
        cursor = mysql_conn.cursor()
        query = """
            INSERT INTO ai_statistics 
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
        logger.info(f"Inserted ai_statistics ai_id={data.ai_id}, affected={affected}, duration={duration:.3f}s")
        cursor.close()
        return True
    except Exception as e:
        duration = time.time() - start
        if cursor:
            try:
                mysql_conn.rollback()
            except Exception:
                pass
        try:
            cfg = get_config()
            if cfg.DB_USE_PERSISTENT_QUEUE and _enqueue_db:
                _enqueue_db('insert_ai_statistics', asdict(data))
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
    ai_statistics 테이블의 특정 AI 통계 갱신 (평균 계산 후 카운트 업데이트)
    입력: (ai_id, win, draw, loss, turns, response_time_ms)
    예시: (1, 1, 0, 0, 63, 455) - 승리, 63턴, 455ms 응답시간
    """
    start = time.time()
    cursor = None
    try:
        if not ensure_mysql_connected():
            logger.error("Cannot update AI statistics because MySQL is not connected")
            try:
                cfg = get_config()
                if cfg.DB_USE_PERSISTENT_QUEUE and _enqueue_db:
                    _enqueue_db('update_ai_statistics', {
                        'ai_id': ai_id,
                        'win': win,
                        'draw': draw,
                        'loss': loss,
                        'turns': turns,
                        'response_time_ms': response_time_ms
                    })
            except Exception:
                pass
            return False
        cursor = mysql_conn.cursor()
        
        update_query = """
            UPDATE ai_statistics
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
                INSERT INTO ai_statistics
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
        logger.info(f"Updated ai_statistics ai_id={ai_id}, affected={affected}, duration={duration:.3f}s")
        cursor.close()
        return True
    except Exception as e:
        duration = time.time() - start
        if cursor:
            try:
                mysql_conn.rollback()
            except Exception:
                pass
        try:
            cfg = get_config()
            if cfg.DB_USE_PERSISTENT_QUEUE and _enqueue_db:
                _enqueue_db('update_ai_statistics', {
                    'ai_id': ai_id,
                    'win': win,
                    'draw': draw,
                    'loss': loss,
                    'turns': turns,
                    'response_time_ms': response_time_ms
                })
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
        if 'game_info' not in existing:
            db.create_collection('game_info')
            logger.info('Created MongoDB collection: game_info')
        # indexes
        try:
            db.game_info.create_index('created_at')
            db.game_info.create_index('winner_ai_id')
            db.game_info.create_index('game_type')
        except Exception as ie:
            logger.debug(f'Could not create GameInfo indexes: {ie}')
    except Exception as e:
        logger.exception(f'Error ensuring GameInfo collection: {e}')

    # Ensure GameDetailLog collection
    try:
        if 'game_detail_log' not in existing:
            db.create_collection('game_detail_log')
            logger.info('Created MongoDB collection: game_detail_log')
        # indexes
        try:
            db.game_detail_log.create_index('gameinfo_id')
            db.game_detail_log.create_index([('ai_id', 1), ('turn_count', 1)])
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
    Currently ensures ai_statistics table.
    """
    cursor = conn.cursor()
    try:
        # Create ai_statistics if not exists
        create_ai_stats = """
        CREATE TABLE IF NOT EXISTS ai_statistics (
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
        logger.info("Ensured MySQL schema: ai_statistics table ready")
    finally:
        try:
            cursor.close()
        except Exception:
            pass

