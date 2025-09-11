from typing import Dict, Any
from Game.BoardGame.BoardGame import BoardGame
from message_schemas import GameRequest, InferenceRequest, InferenceResponse, ModelLoadRequest, ModelLoadResponse, GameProgress, GameType
import logging
import db
import datetime
import time
import numpy as np

class GameManager:
    def __init__(self, rabbitmq_client):
        self.rabbitmq_client = rabbitmq_client
        self.games: Dict[str, BoardGame] = {}  # game_id -> BoardGame 인스턴스
        self.game_meta: Dict[str, Dict[str, Any]] = {}  # game_id -> 메타정보(플레이어, 모델 등)
        self.logger = logging.getLogger(__name__)
        
        # DB 연결 초기화
        try:
            db.init_db_connections()
            self.db_available = True
            self.logger.info("DB 연결 초기화 성공")
            
            # MySQL 연결 상태 확인
            if hasattr(db, 'mysql_conn') and db.mysql_conn is not None:
                self.mysql_available = True
                self.logger.info("MySQL 연결 사용 가능 - AI 통계 업데이트 기능 활성화")
            else:
                self.mysql_available = False
                self.logger.warning("MySQL 연결 불가 - AI 통계 업데이트 기능 비활성화")
                
        except Exception as e:
            self.logger.error(f"DB 연결 실패: {e}")
            self.db_available = False
            self.mysql_available = False

    async def handle_game_request(self, msg: GameRequest):
        """게임 시작 요청 처리"""
        game_id = msg.game_id
        game_type = msg.game_type
        model_ids = msg.model_ids
        model_urls = msg.model_urls

        players = msg.players or [f"model_{i}" for i in range(len(msg.model_ids))]

        # 게임 타입별 클래스 매핑
        game_class_map = {
            GameType.CHESS: 'Chess',
            GameType.OTHELLO: 'Othello',
            GameType.TICTACTOE: 'TicTacToe',
            GameType.OMOK: 'Omok',
        }
        try:
            class_name = game_class_map[game_type]
            module = __import__(f"Game.BoardGame.{class_name}", fromlist=[class_name])
            game_class = getattr(module, class_name)
        except Exception as e:
            self.logger.error(f"지원하지 않는 게임 타입 또는 클래스 로드 실패: {game_type}, {e}")
            return
            
        # 게임 인스턴스 생성 및 초기화
        game = game_class(game_id, players)
        game.initialize()
        self.games[game_id] = game
        
        # GameInfo 기록 (players를 player_ids로 활용)
        gameinfo_id = await self._record_game_info(game_id, model_ids, players)
        if gameinfo_id is None:
            self.logger.warning(f"GameInfo 기록 실패했지만 게임 계속 진행: {game_id}")
            gameinfo_id = "fallback_id"  # 기본값 설정
        
        # 초기 보드 상태 기록
        if self.db_available:
            await self._record_initial_board(game_id, gameinfo_id)

        # AI별 응답 시간 누적 초기화
        ai_response_times = {}
        for model_id in model_ids:
            ai_response_times[str(model_id)] = []

        self.game_meta[game_id] = {
            "players": players,
            "model_ids": model_ids,
            "model_urls": model_urls,
            "turn_number": 0,
            "current_turn": 0,
            "game_type": game_type,
            "gameinfo_id": gameinfo_id,
            "ai_response_times": ai_response_times,  # 추가
        }
        self.logger.info(f"게임 생성 및 초기화 완료: {game_id}, 타입: {game_type}")

        # 초기 진행상황 발행
        await self.broadcast_progress(game_id)
        # 첫 턴 AI 요청
        await self.proceed_turn(game_id)

    async def handle_inference_response(self, msg: InferenceResponse):
        """AI 추론 응답 처리"""
        game_id = msg.game_id
        game = self.games.get(game_id)
        meta = self.game_meta.get(game_id)
        
        if not game or not meta:
            self.logger.error(f"AI 응답 처리 실패: 게임이 존재하지 않음: {game_id}")
            await self.broadcast_progress(game_id, error="게임이 존재하지 않음")
            return

        # Validate pending request id and model to avoid applying stale/out-of-order responses
        pending_req = meta.get("pending_request_id")
        pending_model = meta.get("pending_model_id")
        if pending_req is not None and hasattr(msg, 'request_id') and msg.request_id != pending_req:
            self.logger.warning(f"수신된 응답이 현재 대기중인 요청과 불일치하여 무시합니다: game_id={game_id}, msg.request_id={getattr(msg, 'request_id', None)}, pending={pending_req}")
            return
        if pending_model is not None and str(msg.model_id) != str(pending_model):
            self.logger.warning(f"수신된 응답의 model_id가 현재 턴 모델과 다릅니다. 무시: game_id={game_id}, msg.model_id={msg.model_id}, expected={pending_model}")
            return

        # AI별 응답 시간 누적
        model_id_key = str(msg.model_id)
        self.logger.info(f"AI 응답 누적 시도: model_id_key={model_id_key}, ai_response_times.keys()={list(meta['ai_response_times'].keys())}, response_time_ms={msg.response_time_ms}")
        if model_id_key in meta["ai_response_times"]:
            meta["ai_response_times"][model_id_key].append(msg.response_time_ms)
            self.logger.info(f"AI 응답 누적 성공: {model_id_key} → {meta['ai_response_times'][model_id_key]}")
        else:
            self.logger.warning(f"AI 응답 누적 실패: {model_id_key}가 ai_response_times에 없음")

        # AI_ExecutionLog 기록 (성공/실패 모든 경우)
        if self.db_available:
            if msg.success:
                # 성공 시: 확률배열 기록
                log_output = {
                    "probabilities": msg.probabilities
                }
            else:
                # 실패 시: 빈 확률배열 기록
                log_output = {
                    "probabilities": []
                }
                
            await self._record_ai_execution_log(
                msg.model_id, 
                meta["gameinfo_id"], 
                msg.response_time_ms, 
                msg.success, 
                log_output
            )
            
        if not msg.success:
            self.logger.error(f"AI 추론 실패: {msg}")
            await self.broadcast_progress(game_id, error="AI 추론 실패")
            await self.cleanup_game(game_id)
            return
            
        # 1. 확률 배열에서 최적 수 선택
        move = game.select_best_move_from_probabilities(msg.probabilities)
        if move is None:
            self.logger.error(f"유효한 수를 찾지 못함: {game_id}, 확률: {msg.probabilities}")
            await self.broadcast_progress(game_id, error="유효한 수를 찾지 못함")
            await self.cleanup_game(game_id)
            return
        
        # 2. 현재 턴 정보 저장 (변경 전)
        current_turn_info = {
            "turn_number": meta["turn_number"] + 1,  # 이번에 둘 수의 턴 번호
            "ai_id": int(msg.model_id)
        }
            
        # 3. 게임 상태 갱신
        result = game.play(move)
        meta["turn_number"] += 1
        meta["current_turn"] = (meta["current_turn"] + 1) % len(meta["model_ids"])
        
        self.logger.info(f"게임 {game_id} 턴 {meta['turn_number']} 진행: move={move}, result={result}")
        
        # 4. GameDetailLog 기록 (방금 둔 수 기준)
        if self.db_available:
            await self._record_game_detail_log(game_id, msg, move, current_turn_info, meta["gameinfo_id"])
    
        # 5. 진행상황 발행
        await self.broadcast_progress(game_id)
        
        # 6. 종료 여부 판단
        if result["game_over"]:
            # 게임 종료 시 추가 로깅
            winner = game.get_winner() if game.is_game_over() else None
            if winner:
                self.logger.info(f"게임 종료: {game_id}, 승자: Player {winner}")
            else:
                self.logger.info(f"게임 종료: {game_id}, 무승부")
            
            await self.cleanup_game(game_id)
        else:
            await self.proceed_turn(game_id)

    async def handle_model_load_response(self, msg: ModelLoadResponse):
        """모델 로드 응답 처리"""
        game_id = msg.game_id
        if not msg.success:
            self.logger.error(f"모델 로드 실패: {msg.model_id}, 게임: {game_id}, 메시지: {msg.message}")
            await self.broadcast_progress(game_id, error="모델 로드 실패")
            await self.cleanup_game(game_id)
            return
        self.logger.info(f"모델 로드 성공: {msg.model_id}, 게임: {game_id}, AI 서버: {msg.ai_server_id}")
        # 모델 로드가 성공했으니 해당 게임의 턴을 다시 진행
        await self.proceed_turn(game_id)

    async def proceed_turn(self, game_id: str):
        """한 턴 진행: AI 추론 요청 및 결과 반영"""
        game = self.games.get(game_id)
        meta = self.game_meta.get(game_id)
        if not game or not meta:
            self.logger.error(f"진행할 게임이 존재하지 않음: {game_id}")
            return

        turn_number = meta["turn_number"]
        current_turn = meta["current_turn"]
        model_ids = meta["model_ids"]
        model_id = model_ids[current_turn]
        model_urls = meta["model_urls"]
        model_url = None
        try:
            model_url = model_urls[current_turn]
        except Exception:
            model_url = None
        game_type = meta["game_type"]
        # board_state를 1차원 배열로 변환
        board = getattr(game, "board", None)
        if board is None:
            self.logger.error(f"게임 보드 상태가 없음: {game_id}")
            return
        if game_type == GameType.CHESS:
            # 체스: 64(보드) + 4(캐슬링) + 1(앙파상) = 69
            status = game.get_game_status()
            flat_board = [cell for row in game.board for cell in row]  # 64개
            castling = status["castling_rights"]
            castling_rights = [
                1 if castling["white_king_side"] else 0,
                1 if castling["white_queen_side"] else 0,
                1 if castling["black_king_side"] else 0,
                1 if castling["black_queen_side"] else 0,
            ]  # 4개
            ep = status["en_passant_target"]
            if ep is None:
                ep_idx = -1
            else:
                ep_idx = ep[0] * 8 + ep[1]
            board_state = flat_board + castling_rights + [ep_idx]  # 69개
        else:
            # 기존 방식 (오델로, 틱택토 등)
            if isinstance(board, list) and isinstance(board[0], list):
                board_state = [cell for row in board for cell in row]
            else:
                board_state = board

        # For Omok, many model wrappers expect the first element to indicate current player (1 or -1)
        if game_type == GameType.OMOK:
            try:
                # game.current_player is index (0 or 1) in BoardGame; model expects 1 for player0, -1 for player1
                current_player_value = 1 if getattr(game, 'current_player', 0) == 0 else -1
            except Exception:
                current_player_value = 1
            board_state = [current_player_value] + board_state
            self.logger.debug(f"Omok InferenceRequest board_state length (with player): {len(board_state)}")
        else:
            self.logger.debug(f"InferenceRequest board_state length: {len(board_state)}")

        # 서버 ID는 rabbitmq_client에서 config로 접근
        game_server_id = getattr(self.rabbitmq_client.config, "SERVER_ID", "game_server")

        # InferenceRequest 메시지 생성
        from message_schemas import InferenceRequest
        import datetime, uuid
        req = InferenceRequest(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.utcnow().isoformat(),
            game_id=game_id,
            model_id=model_id,
            board_state=board_state,
            current_turn=current_turn,
            turn_number=turn_number,
            game_type=game_type,
            game_server_id=game_server_id
        )
        # 저장 pending request id for validation when response arrives
        try:
            self.game_meta[game_id]["pending_request_id"] = req.request_id
            self.game_meta[game_id]["pending_model_id"] = model_id
        except Exception:
            pass
        self.logger.info(f"AI 추론 요청 발행: {game_id}, 턴: {turn_number}, 모델: {model_id}, request_id={req.request_id}")
        await self.rabbitmq_client.publish_inference_request(req, model_url=model_url)

    async def broadcast_progress(self, game_id: str, error: str = None):
        """게임 진행 상황 메시지 발행 (오류 메시지 포함 가능)"""
        game = self.games.get(game_id)
        meta = self.game_meta.get(game_id)
        # 오류 상황: 게임/메타 없음 또는 기타 에러
        if not game or not meta or error:
            from message_schemas import GameProgress
            import datetime, uuid
            progress = GameProgress(
                request_id=str(uuid.uuid4()),
                timestamp=datetime.datetime.utcnow().isoformat(),
                game_id=game_id,
                game_type=meta["game_type"] if meta else None,
                board_state=[],
                turn_number=meta["turn_number"] if meta and "turn_number" in meta else 0,
                current_turn=meta["current_turn"] if meta and "current_turn" in meta else 0,
                players=meta["players"] if meta and "players" in meta else [],
                last_move=None,
                is_finished=True,
                winner=None,
                is_success=False
            )
            if error:
                self.logger.error(f"게임 진행상황 발행(오류): {game_id}, error: {error}")
            else:
                self.logger.error(f"진행상황 발행 실패: 게임이 존재하지 않음: {game_id}")
            await self.rabbitmq_client.publish_game_progress(progress)
            return

        # board_state를 1차원 배열로 변환
        board = getattr(game, "board", None)
        if board is None:
            self.logger.error(f"게임 보드 상태가 없음: {game_id}")
            await self.broadcast_progress(game_id, error="보드 상태 없음")
            return
        if isinstance(board, list) and isinstance(board[0], list):
            board_state = [cell for row in board for cell in row]
        else:
            board_state = board
        # 마지막 수
        last_move = game.move_history[-1]["move"] if game.move_history else None

        # 평균 응답시간 계산 (게임 종료 시에만)
        is_finished = game.is_game_over()
        avg_response_times = None
        if is_finished:
            ai_response_times = meta.get("ai_response_times", {})
            model_ids = meta["model_ids"]
            avg_response_times = []
            for mid in model_ids:
                times = ai_response_times.get(str(mid), [])
                if times:
                    avg = sum(times) / len(times)
                else:
                    avg = 0
                avg_response_times.append(avg)

        # GameProgress 메시지 생성
        from message_schemas import GameProgress
        import datetime, uuid
        progress_kwargs = dict(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.utcnow().isoformat(),
            game_id=game_id,
            game_type=meta["game_type"],
            board_state=board_state,
            turn_number=meta["turn_number"],
            current_turn=meta["current_turn"],
            players=meta["players"],
            last_move=str(last_move) if last_move is not None else None,
            is_finished=is_finished,
            winner=game.get_winner() if is_finished else None,
            is_success=True
        )
        if is_finished:
            progress_kwargs["avg_response_times"] = avg_response_times

        progress = GameProgress(**progress_kwargs)
        self.logger.info(f"게임 진행상황 발행: {game_id}, 턴: {meta['turn_number']}, 종료: {progress.is_finished}")
        await self.rabbitmq_client.publish_game_progress(progress)

    async def cleanup_game(self, game_id: str):
        """게임 종료 후 상태 정리"""
        game = self.games.get(game_id)
        meta = self.game_meta.get(game_id)
        
        # GameResult 기록 및 AI 통계 업데이트 (게임 상태 정리 전에)
        if self.db_available and game and meta:
            try:
                # 승자 결정
                winner = game.get_winner() if game.is_game_over() else None
                winner_ai_id = None
                
                if winner is not None:
                    # 플레이어 목록에서 찾아서 해당하는 model_id 사용
                    player_index = meta["players"].index(winner)
                    winner_ai_id = int(meta["model_ids"][player_index])
                    self.logger.info(f"승자 매핑: player={winner} (index={player_index}) → ai_id={winner_ai_id}")
                
                # GameResult 기록
                await self._record_game_result(game_id, meta["gameinfo_id"], winner_ai_id)
                
                # AI 통계 업데이트
                await self._update_ai_statistics_for_game(game_id, meta, winner_ai_id)
                
            except Exception as e:
                self.logger.error(f"GameResult 기록 또는 AI 통계 업데이트 중 오류 발생: {game_id}, error={e}")

        # 기존 정리 로직
        if game_id in self.games:
            del self.games[game_id]
        if game_id in self.game_meta:
            del self.game_meta[game_id]
        self.logger.info(f"게임 상태 정리 완료: {game_id}")

    def get_game(self, game_id: str) -> BoardGame:
        return self.games.get(game_id)

    async def _record_game_info(self, game_id: str, model_ids: list, players: list = None) -> str:
        """GameInfo를 MongoDB에 기록"""
        start_time = time.time()
        
        try:
            # 입력 검증
            if not isinstance(model_ids, list) or len(model_ids) == 0:
                self.logger.error(f"유효하지 않은 model_ids: {model_ids}")
                return None
            players = [int(player) for player in players] if players else []
            model_ids = [int(model) for model in model_ids]
            # GameInfo 스키마 생성
            game_info_data = db.GameInfoSchema(
                player_ids=players,
                ai_ids=model_ids
            )
            
            # MongoDB에 삽입
            gameinfo_id = db.insert_game_info(game_info_data)
            
            duration = time.time() - start_time
            
            if gameinfo_id:
                self.logger.info(f"GameInfo 기록 성공: game_id={game_id}, gameinfo_id={gameinfo_id}, player_ids={players}, ai_ids={model_ids}, 소요시간={duration:.3f}초")
                return gameinfo_id
            else:
                self.logger.error(f"GameInfo 기록 실패: game_id={game_id}, player_ids={players}, ai_ids={model_ids}, 소요시간={duration:.3f}초")
                return None
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"GameInfo 기록 중 예외 발생: game_id={game_id}, error={e}, 소요시간={duration:.3f}초")
            return None

    async def _record_initial_board(self, game_id: str, gameinfo_id: str) -> bool:
        """초기 보드 상태를 GameDetailLog에 기록"""
        start_time = time.time()
        
        try:
            game = self.games.get(game_id)
            if not game:
                self.logger.error(f"초기 보드 기록 실패: 게임이 존재하지 않음: {game_id}")
                return False
                
            # 초기 보드 상태 가져오기
            board = getattr(game, "board", None)
            if board is None:
                self.logger.error(f"초기 보드 기록 실패: 보드 상태가 없음: {game_id}")
                return False
                
            # 보드 상태를 1차원 배열로 변환
            if isinstance(board, list) and isinstance(board[0], list):
                board_snapshot = [cell for row in board for cell in row]
            else:
                board_snapshot = board
                
            # 초기 보드 상태 GameDetailLog 스키마 생성
            game_detail_data = db.GameDetailLogSchema(
                response_time_ms=0,  # 초기 상태이므로 0
                board_snapshot={"board": board_snapshot},
                turn_count=0,  # 초기 상태이므로 턴 0
                move_data=None,  # 초기 상태이므로 None
                ai_id=None,  # 초기 상태이므로 None
                gameinfo_id=gameinfo_id
            )
            
            # MongoDB에 삽입
            detail_log_id = db.insert_game_detail_log(game_detail_data)
            
            duration = time.time() - start_time
            
            if detail_log_id:
                self.logger.info(f"초기 보드 기록 성공: game_id={game_id}, detail_log_id={detail_log_id}, 소요시간={duration:.3f}초")
                return True
            else:
                self.logger.error(f"초기 보드 기록 실패: game_id={game_id}, 소요시간={duration:.3f}초")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"초기 보드 기록 중 예외 발생: game_id={game_id}, error={e}, 소요시간={duration:.3f}초")
            return False

    async def _record_game_detail_log(self, game_id: str, msg: InferenceResponse, move, turn_info: dict, gameinfo_id: str) -> bool:
        """GameDetailLog를 MongoDB에 기록"""
        start_time = time.time()
        
        try:
            game = self.games.get(game_id)
            if not game:
                self.logger.error(f"GameDetailLog 기록 실패: 게임이 존재하지 않음: {game_id}")
                return False
                
            # 현재 보드 상태 가져오기 (move 적용 후)
            board = getattr(game, "board", None)
            if board is None:
                self.logger.error(f"GameDetailLog 기록 실패: 보드 상태가 없음: {game_id}")
                return False
                
            # 보드 상태를 1차원 배열로 변환
            if isinstance(board, list) and isinstance(board[0], list):
                board_snapshot = [cell for row in board for cell in row]
            else:
                board_snapshot = board
                
            # GameDetailLog 스키마 생성
            game_detail_data = db.GameDetailLogSchema(
                response_time_ms=msg.response_time_ms,
                board_snapshot={"board": board_snapshot},
                turn_count=turn_info["turn_number"],  # 방금 둔 수의 턴 번호
                move_data=str(move),
                ai_id=turn_info["ai_id"],  # 방금 수를 둔 AI ID
                gameinfo_id=gameinfo_id
            )
            
            # MongoDB에 삽입
            detail_log_id = db.insert_game_detail_log(game_detail_data)
            
            duration = time.time() - start_time
            
            if detail_log_id:
                self.logger.info(f"GameDetailLog 기록 성공: game_id={game_id}, detail_log_id={detail_log_id}, turn={turn_info['turn_number']}, ai_id={turn_info['ai_id']}, 소요시간={duration:.3f}초")
                return True
            else:
                self.logger.error(f"GameDetailLog 기록 실패: game_id={game_id}, turn={turn_info['turn_number']}, 소요시간={duration:.3f}초")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"GameDetailLog 기록 중 예외 발생: game_id={game_id}, error={e}, 소요시간={duration:.3f}초")
            return False

    async def _record_ai_execution_log(self, model_id: str, gameinfo_id: str, execution_time_ms: int, success: bool, log_output: dict = None) -> bool:
        """AI_ExecutionLog를 MongoDB에 기록"""
        start_time = time.time()
        
        try:
            # 상태 결정
            status = "success" if success else "failed"
            
            # log_output가 없으면 기본값 설정
            if log_output is None:
                log_output = {}
            
            # AI_ExecutionLog 스키마 생성
            ai_execution_data = db.AIExecutionLogSchema(
                execution_time_ms=execution_time_ms,
                status=status,
                log_output=log_output,
                ai_id=int(model_id),
                gameinfo_id=gameinfo_id
            )
            
            # MongoDB에 삽입
            execution_log_id = db.insert_ai_execution_log(ai_execution_data)
            
            duration = time.time() - start_time
            
            if execution_log_id:
                self.logger.info(f"AI_ExecutionLog 기록 성공: model_id={model_id}, gameinfo_id={gameinfo_id}, status={status}, 소요시간={duration:.3f}초")
                return True
            else:
                self.logger.error(f"AI_ExecutionLog 기록 실패: model_id={model_id}, gameinfo_id={gameinfo_id}")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"AI_ExecutionLog 기록 중 예외 발생: model_id={model_id}, gameinfo_id={gameinfo_id}, error={e}, 소요시간={duration:.3f}초")
            return False

    async def _record_game_result(self, game_id: str, gameinfo_id: str, winner_ai_id: int = None) -> bool:
        """GameResult를 MongoDB에 기록"""
        start_time = time.time()
        
        try:
            import datetime
            
            # GameResult 스키마 생성
            game_result_data = db.GameResultSchema(
                created_at=datetime.datetime.utcnow().isoformat(),
                winner_ai_id=winner_ai_id,  # None이면 무승부
                gameinfo_id=gameinfo_id
            )
            
            # MongoDB에 삽입
            result_id = db.insert_game_result(game_result_data)
            
            duration = time.time() - start_time
            
            if result_id:
                winner_info = f"winner_ai_id={winner_ai_id}" if winner_ai_id else "무승부"
                self.logger.info(f"GameResult 기록 성공: game_id={game_id}, result_id={result_id}, {winner_info}, 소요시간={duration:.3f}초")
                return True
            else:
                self.logger.error(f"GameResult 기록 실패: game_id={game_id}, winner_ai_id={winner_ai_id}")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"GameResult 기록 중 예외 발생: game_id={game_id}, error={e}, 소요시간={duration:.3f}초")
            return False

    async def _update_ai_statistics_for_game(self, game_id: str, meta: dict, winner_ai_id: int = None) -> bool:
        """게임 종료 시 참여한 모든 AI의 통계 업데이트"""
        start_time = time.time()
        
        try:
            # MySQL 연결 확인
            if not hasattr(db, 'mysql_conn') or db.mysql_conn is None:
                self.logger.warning(f"MySQL 연결이 없어 AI 통계 업데이트를 건너뜀: {game_id}")
                return False
            
            total_turns = meta["turn_number"]
            model_ids = meta["model_ids"]
            ai_response_times = meta.get("ai_response_times", {})
            
            # 각 AI별로 통계 업데이트
            for i, model_id in enumerate(model_ids):
                try:
                    ai_id = int(model_id)
                    
                    # 승부 결정
                    if winner_ai_id == ai_id:
                        win, draw, loss = 1, 0, 0  # 승리
                    elif winner_ai_id is None:
                        win, draw, loss = 0, 1, 0  # 무승부
                    else:
                        win, draw, loss = 0, 0, 1  # 패배
                    
                    # 평균 응답 시간 계산
                    response_times = ai_response_times.get(model_id, [])
                    if response_times:
                        avg_response_time = sum(response_times) // len(response_times)
                    else:
                        avg_response_time = 0
                    
                    # AI_Statistics 업데이트
                    success = db.update_ai_statistics(
                        ai_id=ai_id,
                        win=win,
                        draw=draw,
                        loss=loss,
                        turns=total_turns,
                        response_time_ms=avg_response_time
                    )
                    
                    if success:
                        result_info = "승리" if win else ("무승부" if draw else "패배")
                        self.logger.info(f"AI 통계 업데이트 성공: ai_id={ai_id}, 결과={result_info}, 평균응답시간={avg_response_time}ms, 총턴수={total_turns}")
                    else:
                        self.logger.error(f"AI 통계 업데이트 실패: ai_id={ai_id}")
                        
                except (ValueError, TypeError) as e:
                    self.logger.error(f"AI ID 변환 실패: {model_id}, error={e}")
                    continue
                except Exception as e:
                    self.logger.error(f"AI 통계 업데이트 중 오류: ai_id={model_id}, error={e}")
                    continue
            
            duration = time.time() - start_time
            self.logger.info(f"모든 AI 통계 업데이트 완료: game_id={game_id}, 소요시간={duration:.3f}초")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"AI 통계 업데이트 중 예외 발생: game_id={game_id}, error={e}, 소요시간={duration:.3f}초")
            return False
