import asyncio
import logging
from config import Config
from rabbitmq_client import RabbitMQClient
from game_manager import GameManager

async def main():
    # 1. 설정 및 로깅
    config = Config()
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
    logger = logging.getLogger(__name__)
    logger.info("게임 서버 시작")

    # 2. RabbitMQ 연결 (무한 재시도)
    rabbitmq = RabbitMQClient(config, config.SERVER_ID)
    while True:
        connected = await rabbitmq.connect()
        if connected:
            break
        logger.error("RabbitMQ 연결 실패. 5초 후 재시도합니다.")
        await asyncio.sleep(5)
    await rabbitmq.setup_topology()

    # 3. GameManager 생성
    manager = GameManager(rabbitmq)

    # 4. 콜백 등록
    callbacks = {
        "game_request": manager.handle_game_request,
        "inference_response": manager.handle_inference_response,
        "model_load_response": manager.handle_model_load_response,
    }
    await rabbitmq.start_consuming(callbacks)

    # 5. graceful shutdown 처리
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("서버 종료 신호 수신. 연결 해제 중...")
        await rabbitmq.disconnect()
        logger.info("서버 정상 종료.")

if __name__ == "__main__":
    asyncio.run(main())
