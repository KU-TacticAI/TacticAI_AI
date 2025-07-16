from typing import Dict, Any
from Game.BoardGame.BoardGame import BoardGame
from message_schemas import GameRequest, InferenceRequest, InferenceResponse, ModelLoadRequest, ModelLoadResponse, GameProgress, GameType
import logging

class GameManager:
    def __init__(self, rabbitmq_client):
        self.rabbitmq_client = rabbitmq_client
        self.games: Dict[str, BoardGame] = {}  # game_id -> BoardGame 인스턴스
        self.game_meta: Dict[str, Dict[str, Any]] = {}  # game_id -> 메타정보(플레이어, 모델 등)
        self.logger = logging.getLogger(__name__)

    async def handle_game_request(self, msg: GameRequest):
        """게임 시작 요청 처리"""
        game_id = msg.game_id
        game_type = msg.game_type
        players = msg.players or [f"model_{i}" for i in range(len(msg.model_ids))]
        model_ids = msg.model_ids
        model_urls = msg.model_urls

        # 게임 타입별 클래스 매핑
        game_class_map = {
            GameType.CHESS: 'Chess',
            GameType.OTHELLO: 'Othello',
            GameType.TICTACTOE: 'TicTacToe',
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
        self.game_meta[game_id] = {
            "players": players,
            "model_ids": model_ids,
            "model_urls": model_urls,
            "turn_number": 0,
            "current_turn": 0,
            "game_type": game_type,
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
        # 2. 게임 상태 갱신
        result = game.play(move)
        meta["turn_number"] += 1
        meta["current_turn"] = (meta["current_turn"] + 1) % len(meta["model_ids"])
        self.logger.info(f"게임 {game_id} 턴 {meta['turn_number']} 진행: move={move}, result={result}")
        # 3. 진행상황 발행
        await self.broadcast_progress(game_id)
        # 4. 종료 여부 판단
        if result["game_over"]:
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
        self.logger.info(f"AI 추론 요청 발행: {game_id}, 턴: {turn_number}, 모델: {model_id}")
        await self.rabbitmq_client.publish_inference_request(req, model_url=model_url)

    async def broadcast_progress(self, game_id: str, error: str = None):
        """게임 진행 상황 메시지 발행 (오류 메시지 포함 가능)"""
        game = self.games.get(game_id)
        meta = self.game_meta.get(game_id)
        # 오류 상황: 게임/메타 없음 또는 기타 에러
        if not game or not meta or error:
            # 오류 상황에도 최소한의 정보로 진행상황 발행
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
        # GameProgress 메시지 생성
        from message_schemas import GameProgress
        import datetime, uuid
        progress = GameProgress(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.utcnow().isoformat(),
            game_id=game_id,
            game_type=meta["game_type"],
            board_state=board_state,
            turn_number=meta["turn_number"],
            current_turn=meta["current_turn"],
            players=meta["players"],
            last_move=str(last_move) if last_move is not None else None,
            is_finished=game.is_game_over(),
            winner=game.get_winner() if game.is_game_over() else None,
            is_success=True
        )
        self.logger.info(f"게임 진행상황 발행: {game_id}, 턴: {meta['turn_number']}, 종료: {progress.is_finished}")
        await self.rabbitmq_client.publish_game_progress(progress)

    async def cleanup_game(self, game_id: str):
        """게임 종료 후 상태 정리"""
        if game_id in self.games:
            del self.games[game_id]
        if game_id in self.game_meta:
            del self.game_meta[game_id]
        self.logger.info(f"게임 상태 정리 완료: {game_id}")

    def get_game(self, game_id: str) -> BoardGame:
        return self.games.get(game_id)
