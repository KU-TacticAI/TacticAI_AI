import sys
import os
import uuid
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import pytest
import db
from bson import ObjectId

# 테스트에서 공통으로 사용할 ai_id와 gameinfo_id
AI_ID = 1
GAMEINFO_ID = "988617ce809acc3a88186e16"  # 32자리 소문자 hex 문자열


def setup_module(module):
    db.init_db_connections()

def teardown_module(module):
    # 테스트로 삽입된 GameInfo 문서 삭제
    mongo_db = db.mongodb_client[db.get_config().MONGO_DB_NAME]
    mongo_db.GameInfo.delete_many({"test_flag": True})
    
    # 테스트용 AI 통계 데이터 삭제
    cursor = db.mysql_conn.cursor()
    cursor.execute("DELETE FROM AI_Statistics WHERE ai_id = %s", (AI_ID,))
    db.mysql_conn.commit()
    cursor.close()

def test_insert_game_info():
    test_data = db.GameInfoSchema(
        player_ids=[674232352, 676734312421],
        ai_ids=[12456243, 45645734523]
    )
    inserted_id = db.insert_game_info(test_data)
    assert inserted_id is not None
    # 실제로 DB에 잘 들어갔는지 확인
    mongo_db = db.mongodb_client[db.get_config().MONGO_DB_NAME]
    doc = mongo_db.GameInfo.find_one({"_id": ObjectId(inserted_id)})
    assert doc is not None
    assert doc["player_ids"] == [674232352, 676734312421]
    assert doc["ai_ids"] == [12456243, 45645734523]
    return inserted_id


def test_insert_game_detail_log():
    test_data = db.GameDetailLogSchema(
        response_time_ms=123,
        board_snapshot={"board": [[1, 0, 0],[0, 0, 0],[0, 0, 0]]},
        turn_count=1,
        move_data="a1",
        ai_id=AI_ID,  # int
        gameinfo_id=GAMEINFO_ID  # uuid4 hex 문자열
    )
    inserted_id = db.insert_game_detail_log(test_data)
    assert inserted_id is not None
    # 실제로 DB에 잘 들어갔는지 확인
    mongo_db = db.mongodb_client[db.get_config().MONGO_DB_NAME]
    doc = mongo_db.GameDetailLog.find_one({"_id": ObjectId(inserted_id)})
    assert doc is not None
    assert doc["response_time_ms"] == 123
    assert doc["board_snapshot"] == {"board": [[1, 0, 0],[0, 0, 0],[0, 0, 0]]}
    assert doc["turn_count"] == 1
    assert doc["move_data"] == "a1"
    assert doc["ai_id"] == AI_ID
    assert doc["gameinfo_id"] == GAMEINFO_ID


def test_insert_ai_execution_log():
    test_data = db.AIExecutionLogSchema(
        execution_time_ms=456,
        status="success",
        log_output={"msg": "ok"},
        ai_id=AI_ID,
        gameinfo_id=GAMEINFO_ID
    )
    inserted_id = db.insert_ai_execution_log(test_data)
    assert inserted_id is not None
    mongo_db = db.mongodb_client[db.get_config().MONGO_DB_NAME]
    doc = mongo_db.AI_ExecutionLog.find_one({"_id": ObjectId(inserted_id)})
    assert doc is not None
    assert doc["execution_time_ms"] == 456
    assert doc["status"] == "success"
    assert doc["log_output"] == {"msg": "ok"}
    assert doc["ai_id"] == AI_ID
    assert doc["gameinfo_id"] == GAMEINFO_ID


def test_insert_game_result():
    test_data = db.GameResultSchema(
        created_at="2024-06-01T12:00:00Z",
        winner_ai_id=AI_ID,
        gameinfo_id=GAMEINFO_ID
    )
    inserted_id = db.insert_game_result(test_data)
    assert inserted_id is not None
    mongo_db = db.mongodb_client[db.get_config().MONGO_DB_NAME]
    doc = mongo_db.GameResult.find_one({"_id": ObjectId(inserted_id)})
    assert doc is not None
    assert doc["created_at"] == "2024-06-01T12:00:00Z"
    assert doc["winner_ai_id"] == AI_ID
    assert doc["gameinfo_id"] == GAMEINFO_ID

@pytest.mark.mysql_test
def test_insert_ai_statistics():
    """AI_Statistics 테이블에 초기 데이터 삽입 테스트"""
    test_data = db.AIStatisticsSchema(
        ai_id=AI_ID,
        game_count=0,
        wins=0,
        losses=0,
        draws=0,
        avg_turns=0.0,
        avg_response_time_ms=0,
        win_rate=0.0
    )
    result = db.insert_ai_statistics(test_data)
    assert result is True

@pytest.mark.mysql_test
def test_update_ai_statistics():
    """AI_Statistics 테이블 업데이트 테스트"""
    # 첫 번째 게임: 승리, 50턴, 300ms
    result1 = db.update_ai_statistics(AI_ID, 1, 0, 0, 50, 300)
    assert result1 is True
    
    # 두 번째 게임: 패배, 75턴, 450ms  
    result2 = db.update_ai_statistics(AI_ID, 0, 0, 1, 75, 450)
    assert result2 is True
    
    # 세 번째 게임: 무승부, 60턴, 350ms
    result3 = db.update_ai_statistics(AI_ID, 0, 1, 0, 60, 350)
    assert result3 is True
    
    # 결과 검증
    cursor = db.mysql_conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM AI_Statistics WHERE ai_id = %s", (AI_ID,))
    data = cursor.fetchone()
    cursor.close()
    
    assert data is not None
    assert data['game_count'] == 3
    assert data['wins'] == 1
    assert data['losses'] == 1  
    assert data['draws'] == 1
    # avg_turns = (50 + 75 + 60) / 3 = 61.67 (decimal(5,2)로 반올림)
    assert abs(float(data['avg_turns']) - 61.67) < 0.01
    # avg_response_time_ms = (300 + 450 + 350) / 3 = 366.67 → 367 (int로 반올림)
    assert data['avg_response_time_ms'] == 367
    # win_rate = 1 / 3 = 0.33 (decimal(5,2)로 반올림)
    assert abs(float(data['win_rate']) - 0.33) < 0.01