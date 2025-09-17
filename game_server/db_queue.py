import os
import json
import time
import threading
import logging
from typing import Dict, Any
from config import get_config

logger = logging.getLogger("game_server.db_queue")

_lock = threading.Lock()
_worker_thread = None
_worker_running = False


def _ensure_queue_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass


def enqueue(item_type: str, payload: Dict[str, Any]):
    """Append an item to the persistent JSONL queue.
    item_type: string identifying handler (e.g. 'insert_game_info', 'insert_game_detail_log', 'update_ai_statistics', 'insert_ai_statistics', 'update_game_info_winner')
    payload: JSON-serializable payload
    """
    cfg = get_config()
    if not cfg.DB_USE_PERSISTENT_QUEUE:
        return False

    path = cfg.DB_QUEUE_PATH
    _ensure_queue_dir(path)

    line = json.dumps({"type": item_type, "payload": payload, "ts": time.time()}, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ensure worker is running
    _start_worker_if_needed()
    logger.debug(f"Enqueued DB item: {item_type}")
    return True


def _start_worker_if_needed():
    global _worker_thread, _worker_running
    if _worker_running:
        return
    _worker_running = True
    _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
    _worker_thread.start()
    logger.info("Started DB queue background worker")


def _worker_loop():
    cfg = get_config()
    interval = max(0.1, float(cfg.DB_QUEUE_FLUSH_INTERVAL))
    while True:
        try:
            _flush_once()
        except Exception as e:
            logger.exception(f"DB queue flush error: {e}")
        time.sleep(interval)


def _flush_once():
    cfg = get_config()
    path = cfg.DB_QUEUE_PATH
    if not os.path.exists(path):
        return

    # read all items
    with _lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
        except FileNotFoundError:
            return

    if not lines:
        return

    items = []
    for ln in lines:
        try:
            items.append(json.loads(ln))
        except Exception:
            # malformed line -> skip
            continue

    remaining = []
    # mapping from client_gameinfo_id -> real inserted id (string)
    client_map = {}

    # process items sequentially
    for item in items:
        t = item.get("type")
        payload = item.get("payload")
        success = False
        try:
            # import here to avoid circular import at module load
            import game_server.db as db
            # substitute client_gameinfo_id in payload if available
            if isinstance(payload, dict) and 'client_gameinfo_id' in payload:
                cid = payload.get('client_gameinfo_id')
                if cid in client_map:
                    payload['gameinfo_id'] = client_map[cid]
                    # remove client_gameinfo_id to avoid double storage
                    payload.pop('client_gameinfo_id', None)
            if t == 'insert_game_info':
                # payload is a dict matching GameInfoSchema
                try:
                    gi = db.GameInfoSchema(**{k: v for k, v in payload.items() if k != 'client_gameinfo_id'})
                    # pass client_gameinfo_id through so the DB doc keeps it
                    client_id = payload.get('client_gameinfo_id')
                    rid = db.insert_game_info(gi, client_gameinfo_id=client_id)
                    success = bool(rid)
                    if success and client_id:
                        client_map[client_id] = rid
                except Exception:
                    success = False
            elif t == 'insert_game_detail_log':
                try:
                    # payload expected to contain gameinfo_id (may have been substituted above)
                    gdl = db.GameDetailLogSchema(**payload)
                    rid = db.insert_game_detail_log(gdl)
                    success = bool(rid)
                except Exception:
                    success = False
            elif t == 'insert_ai_statistics':
                try:
                    ais = db.AIStatisticsSchema(**payload)
                    success = db.insert_ai_statistics(ais)
                except Exception:
                    success = False
            elif t == 'update_ai_statistics':
                # payload expected to be dict of args for update_ai_statistics
                try:
                    success = db.update_ai_statistics(**payload)
                except Exception:
                    success = False
            elif t == 'update_game_info_winner':
                try:
                    # payload may contain client_gameinfo_id substituted earlier
                    success = db.update_game_info_winner(payload.get('gameinfo_id'), payload.get('winner_ai_id'))
                except Exception:
                    success = False
            else:
                # unknown type -> skip
                success = True

        except Exception as e:
            logger.debug(f"DB queue processing import or call error: {e}")
            success = False

        if not success:
            remaining.append(item)

    # write back remaining (overwrite file)
    with _lock:
        if remaining:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    for it in remaining:
                        f.write(json.dumps(it, ensure_ascii=False) + "\n")
            except Exception:
                # if we cannot write remaining, don't lose data -> try appending back
                try:
                    with open(path, "a", encoding="utf-8") as f:
                        for it in remaining:
                            f.write(json.dumps(it, ensure_ascii=False) + "\n")
                except Exception:
                    pass
        else:
            try:
                os.remove(path)
            except Exception:
                # if removal fails, truncate file
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        pass
                except Exception:
                    pass


if __name__ == '__main__':
    # quick manual test
    enqueue('insert_game_info', {'player_ids': [1], 'ai_ids': [2], 'created_at': 'now', 'winner_ai_id': None, 'game_type': 't'})
    print('enqueued')
